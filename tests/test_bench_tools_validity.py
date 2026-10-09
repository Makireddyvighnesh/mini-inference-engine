from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from minillm_l4.benchmarks.commands import run_cuda_graphs, run_fused_kernels, run_showcase, run_vllm_compare
from minillm_l4.benchmarks.core.metrics import percentile


@pytest.mark.parametrize("completed,expected,retries,valid", [
    (6, 6, 0, True), (5, 6, 0, False), (0, 6, 0, False),
    (7, 6, 0, False), (6, 6, 1, False), (5, 6, 2, False),
])
def test_shared_validity_and_exit(completed, expected, retries, valid, capsys):
    entry = {"completed": completed, "expected": expected, "alloc_retries": retries}
    entry.update(run_vllm_compare.case_validity(entry))
    assert entry["valid"] is valid
    assert (entry["invalid_reason"] is None) is valid
    run_vllm_compare.print_invalid_case("test case", entry)
    if valid:
        run_vllm_compare.exit_if_invalid([entry])
        assert capsys.readouterr().out == ""
    else:
        assert "INVALID" in capsys.readouterr().out
        if completed != expected:
            assert f"{completed}/{expected}" in entry["invalid_reason"]
        if retries:
            assert f"alloc_retries={retries}" in entry["invalid_reason"]
        with pytest.raises(SystemExit) as error:
            run_vllm_compare.exit_if_invalid([{"valid": True}, entry])
        assert error.value.code == 1


def test_metric_tools_import_without_engine_or_runners():
    script = """
import importlib.abc
import sys
class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('minillm_l4.engine', 'minillm_l4.benchmarks.runners')):
            raise AssertionError('unexpected engine import: ' + fullname)
sys.meta_path.insert(0, NoEngine())
from minillm_l4.benchmarks.commands import run_showcase, run_fused_kernels, run_cuda_graphs, run_vllm_compare
assert run_vllm_compare.case_validity({'completed': 2, 'expected': 3})['valid'] is False
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def showcase_result(values, statuses=None):
    statuses = statuses or ["completed"] * len(values)
    runs = [{"requests": [{"metrics": {
        "ttft_ms": value, "tpot_ms": None, "itl_ms": [], "status": status,
    }} for value, status in zip(values, statuses)], "summary": {
        "gpu_utilization_percent": {"count": 0}, "tokens_per_second": 10, "duration_ms": 100,
    }}]
    return SimpleNamespace(runs=runs, to_dict=lambda: {"runs": runs})


def test_showcase_p95_uses_interpolation_and_keeps_nulls():
    result = run_showcase.metrics(showcase_result([30, None, 10]), [object()] * 3)
    assert result["ttft_p50_ms"] == 20
    assert result["ttft_p95_ms"] == percentile([30, 10], 95) == 29
    assert result["valid"] is True
    empty = run_showcase.metrics(showcase_result([None], ["failed"]), [object()])
    assert empty["ttft_p95_ms"] is None
    assert empty["valid"] is False
    missing_run = run_showcase.metrics(showcase_result([10]), [object()], expected_runs=2)
    assert missing_run["valid"] is False
    assert missing_run["expected"] == 2


def comparison_run(rows):
    return {"duration_ms": 100, "requests": rows}


def comparison_row(*, tokens=1, expected=1, status="completed"):
    return {"id": "r0", "ttft_ms": 10 if tokens else None, "tpot_ms": None,
            "max_gap_ms": None, "tokens": tokens, "expected_tokens": expected,
            "token_ids": [1] * tokens, "status": status}


@pytest.mark.parametrize("rows,expected,completed", [
    ([comparison_row()], 1, 1),
    ([comparison_row(tokens=0)], 1, 0),
    ([comparison_row(status="failed")], 1, 0),
    ([comparison_row()], 2, 1),
    ([], 2, 0),
])
def test_comparison_counts_failures_and_missing_rows(rows, expected, completed):
    entry = run_vllm_compare.summarize([comparison_run(rows)], expected=expected)
    assert entry["completed"] == completed
    assert entry["expected"] == expected
    assert entry["valid"] is (completed == expected)


@pytest.fixture
def cpu_command_stubs(monkeypatch):
    state = SimpleNamespace(calls=0, closed=0, retries=0, invalid=None)
    cuda = run_showcase.torch.cuda
    monkeypatch.setattr(cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda, "get_device_name", lambda _: "CPU test stub")
    monkeypatch.setattr(cuda, "memory_stats", lambda: {"num_alloc_retries": state.retries})
    monkeypatch.setattr(cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(cuda, "empty_cache", lambda: None)

    class Runner:
        def __init__(self, **options):
            self.options = options

        def close(self):
            state.closed += 1

    class Harness:
        def __init__(self, config):
            self.repetitions = config.repetitions

        def run_trace(self, workload, runner):
            first = state.calls == 0
            state.calls += 1
            if first and state.invalid == "retries":
                state.retries += 1
            statuses = ["failed" if first and state.invalid == "failed" else "completed"] * len(workload.requests)
            values = [None if status == "failed" else 10 for status in statuses]
            if first and state.invalid == "missing":
                values, statuses = values[:-1], statuses[:-1]
            result = showcase_result(values, statuses)
            result.runs *= self.repetitions
            graph = runner.options.get("cuda_graphs", False)
            scheduler = {
                "execution_records": [{"kind": "decode", "batch_size": 1, "wall_ms": 10}],
                "graph_replays": int(graph), "eager_decode_steps": int(not graph),
                "decode_mode_used": "graph" if graph else "eager", "graph_captures_this_run": [],
                "graph_fallback_steps_by_reason": {}, "captured_buckets": [], "capture_ms": {},
                "graph_pool_memory_bytes_per_bucket": {}, "graph_pool_memory_bytes": 0,
                "graph_scratch_pages": 0, "graph_scratch_memory_bytes": 0,
                "graph_static_buffer_memory_bytes_per_bucket": {},
            }
            runner.run_summaries = [scheduler] * (self.repetitions + 1)
            runner.last_summary = scheduler
            return result

    for command in (run_showcase, run_fused_kernels, run_cuda_graphs):
        monkeypatch.setattr(command, "load_qwen_fp8", lambda **_: SimpleNamespace(model=object(), tokenizer=None))
        monkeypatch.setattr(command, "prompt", lambda tokenizer, length, tag: (1,) * length)
        monkeypatch.setattr(command, "paged", lambda model, requests, **options: Runner(**options))
        monkeypatch.setattr(command, "BenchmarkHarness", Harness)
    monkeypatch.setattr(run_fused_kernels, "install_fused_kernels", lambda _: True)
    monkeypatch.setattr(run_fused_kernels, "uninstall_fused_kernels", lambda _: True)
    return state


@pytest.mark.parametrize("command,section,summary_name,raw_key,case_count", [
    (run_showcase, "prefix", "showcase.json", "case_metrics", 2),
    (run_fused_kernels, "prefill", "fused_kernels.json", "case_metrics", 14),
    (run_cuda_graphs, "overhead", "cuda_graphs.json", "cuda_graphs", 6),
])
@pytest.mark.parametrize("invalid", ["failed", "missing", "retries", None])
def test_commands_save_raw_continue_and_exit_at_end(
    tmp_path, cpu_command_stubs, capsys, command, section, summary_name, raw_key, case_count, invalid,
):
    state = cpu_command_stubs
    state.invalid = invalid
    output = tmp_path / "output"
    argv = ["--output-dir", str(output), "--sections", section]
    if invalid:
        with pytest.raises(SystemExit) as error:
            command.main(argv)
        assert error.value.code == 1
    else:
        command.main(argv)
    assert state.calls == state.closed == case_count
    data = json.loads((output / summary_name).read_text())
    entries = [entry for name, rows in data["sections"].items() if name != "overhead" for entry in rows]
    assert len(entries) == case_count
    assert entries[0]["valid"] is (invalid is None)
    assert all(e["valid"] for e in entries[1:])
    raw_files = [p for p in output.glob("*.json") if p.name != summary_name]
    assert len(raw_files) == case_count
    raw = [json.loads(p.read_text()) for p in raw_files]
    assert all("runs" in r for r in raw)
    saved = [r[raw_key]["metrics"] if raw_key == "cuda_graphs" else r[raw_key] for r in raw]
    assert sum(not e["valid"] for e in saved) == int(invalid is not None)
    if section == "overhead":
        assert data["sections"]["overhead"][0]["valid"] is (invalid is None)
    printed = capsys.readouterr().out
    assert ("INVALID" in printed) is (invalid is not None)
    assert "Summary:" in printed


@pytest.mark.parametrize("invalid", ["incomplete", "retries", "missing", "status", None])
def test_report_flags_legacy_invalid_cases_and_omits_ratios(tmp_path, capsys, invalid):
    rows = [comparison_row(tokens=0 if invalid == "incomplete" else 1,
                           status="failed" if invalid == "status" else "completed")]
    ours = {**run_vllm_compare.summarize([comparison_run(rows)]), "runs": [comparison_run(rows)]}
    # Exercise old JSON files with no validity markers and misleading counts.
    ours.pop("valid")
    ours.pop("invalid_reason")
    if invalid == "status":
        ours["completed"] = 1
    ours["alloc_retries"] = int(invalid == "retries")
    theirs_rows = [comparison_row()]
    theirs = {**run_vllm_compare.summarize([comparison_run(theirs_rows)]), "runs": [comparison_run(theirs_rows)]}
    for engine, entry, config in (
        ("minillm", ours, {"fused_kernels": True}),
        ("vllm", theirs, {"vllm_version": "test", "cudagraph_mode": "test", "max_num_batched_tokens": 256}),
    ):
        (tmp_path / f"{engine}.json").write_text(json.dumps({"engine": engine, "config": config, "cases": {"case": entry}}))
    (tmp_path / "workloads.json").write_text(json.dumps([
        {"name": "case", "section": "prefill", "label": "test case", "requests": [{}] * (2 if invalid == "missing" else 1)},
    ]))
    args = SimpleNamespace(output_dir=tmp_path)
    if invalid:
        with pytest.raises(SystemExit) as error:
            run_vllm_compare.report(args)
        assert error.value.code == 1
    else:
        run_vllm_compare.report(args)
    text = (tmp_path / "comparison.md").read_text()
    printed = capsys.readouterr().out
    assert ("INVALID" in text) is (invalid is not None)
    assert ("INVALID" in printed) is (invalid is not None)
    table_row = next(line for line in text.splitlines() if line.startswith("| test case |"))
    assert (" / 1.00x" in table_row) is (invalid is None)
    assert ("## Output agreement" in text) is (invalid is None)


@pytest.mark.parametrize("mode", ["empty", "partial", "error", "complete"])
def test_vllm_retains_empty_partial_and_failed_streams(monkeypatch, mode):
    inputs, sampling = ModuleType("vllm.inputs"), ModuleType("vllm.sampling_params")
    inputs.TokensPrompt = lambda **kwargs: kwargs
    sampling.RequestOutputKind = SimpleNamespace(DELTA="delta")
    sampling.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling)

    class Engine:
        async def generate(self, *args, **kwargs):
            if mode != "empty":
                yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1, 2] if mode == "complete" else [1])])
            if mode == "error":
                raise RuntimeError("test stream failure")

    work = {"requests": [{"id": "r0", "arrival_ms": 0, "max_new_tokens": 2, "prompt_token_ids": [1]}]}
    run = asyncio.run(run_vllm_compare._vllm_case(Engine(), work, "test"))
    entry = run_vllm_compare.summarize([run], expected=1)
    assert entry["valid"] is (mode == "complete")
    row = run["requests"][0]
    assert row["tokens"] == len(row["token_ids"])
    if mode == "empty":
        assert row["ttft_ms"] is None
    if mode == "error":
        assert row["error"] == "RuntimeError: test stream failure"
        assert row["token_ids"] == [1]


def stub_module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize("invalid", ["failed", "retries", None])
def test_minillm_comparison_saves_all_cases_before_exit(tmp_path, monkeypatch, cpu_command_stubs, invalid):
    state = cpu_command_stubs
    state.invalid = invalid
    from minillm_l4.benchmarks.core import harness

    # Stub optional engine modules so this command test never imports engine code.
    stub_module(monkeypatch, "minillm_l4.benchmarks.runners.huggingface_baseline",
                MODEL_ID=run_vllm_compare.MODEL_ID, MODEL_REVISION=run_vllm_compare.MODEL_REVISION,
                load_qwen_fp8=run_showcase.load_qwen_fp8)
    stub_module(monkeypatch, "minillm_l4.benchmarks.runners.chunked_prefill",
                ChunkedPrefillPagedRunner=lambda model, **options: run_showcase.paged(model, (), **options))
    stub_module(monkeypatch, "minillm_l4.engine.kernels.fused", install_fused_kernels=lambda model: True)
    base_harness = run_showcase.BenchmarkHarness

    class Harness(base_harness):
        def run_trace(self, workload, runner):
            result = super().run_trace(workload, runner)
            for run in result.runs:
                run["duration_ms"] = 100
                for row in run["requests"]:
                    row["request_id"] = "r0"
                    row["metrics"]["requested_output_tokens"] = 1
                    row["outcome"] = {"generated_token_ids": [] if row["metrics"]["status"] == "failed" else [1]}
            return result

    monkeypatch.setattr(harness, "BenchmarkHarness", Harness)
    workloads = [{"name": name, "section": "prefill", "requests": [
        {"id": "r0", "prompt_token_ids": [1], "max_new_tokens": 1, "arrival_ms": 0},
    ]} for name in ("first", "second")]
    monkeypatch.setattr(run_vllm_compare, "build_workloads", lambda tok: workloads)
    args = SimpleNamespace(output_dir=tmp_path, sections=None, repetitions=2, warmup_repetitions=0)
    if invalid:
        with pytest.raises(SystemExit) as error:
            run_vllm_compare.run_minillm(args)
        assert error.value.code == 1
    else:
        run_vllm_compare.run_minillm(args)
    cases = json.loads((tmp_path / "minillm.json").read_text())["cases"]
    assert state.calls == state.closed == 2
    assert cases["first"]["valid"] is (invalid is None)
    assert cases["second"]["valid"] is True
    assert cases["first"]["expected"] == 2
    assert len(cases["first"]["runs"]) == 2


@pytest.mark.parametrize("invalid", [True, False])
def test_vllm_comparison_saves_all_cases_and_shuts_down_before_exit(tmp_path, monkeypatch, invalid):
    state = SimpleNamespace(calls=0, stopped=False)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=256, enable_chunked_prefill=True),
        compilation_config=SimpleNamespace(cudagraph_mode="test"),
        model_config=SimpleNamespace(quantization="fp8"),
    )

    class Engine:
        vllm_config = config

        def shutdown(self):
            state.stopped = True

    stub_module(monkeypatch, "vllm", __version__="test")
    stub_module(monkeypatch, "vllm.engine.arg_utils", AsyncEngineArgs=lambda **kwargs: kwargs)
    stub_module(monkeypatch, "vllm.v1.engine.async_llm",
                AsyncLLM=SimpleNamespace(from_engine_args=lambda args: Engine()))

    async def case(engine, work, label):
        state.calls += 1
        return comparison_run([comparison_row(tokens=0 if invalid and work["name"] == "first" else 1)])

    monkeypatch.setattr(run_vllm_compare, "_vllm_case", case)
    (tmp_path / "workloads.json").write_text(json.dumps([
        {"name": name, "section": "prefill", "requests": [{}]} for name in ("first", "second")
    ]))
    args = SimpleNamespace(output_dir=tmp_path, sections=None, repetitions=2, warmup_repetitions=0,
                           gpu_memory_utilization=0.9)
    if invalid:
        with pytest.raises(SystemExit) as error:
            asyncio.run(run_vllm_compare._run_vllm(args))
        assert error.value.code == 1
    else:
        asyncio.run(run_vllm_compare._run_vllm(args))
    assert state.calls == 4 and state.stopped
    cases = json.loads((tmp_path / "vllm.json").read_text())["cases"]
    assert cases["first"]["valid"] is (not invalid)
    assert cases["second"]["valid"] is True
    assert cases["first"]["expected"] == 2
    assert len(cases["first"]["runs"]) == 2
