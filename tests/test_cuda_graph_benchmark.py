from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from minillm_l4.benchmarks.commands import run_cuda_graphs as command


def test_case_metrics_use_step_time_and_measured_replay_share():
    runs = [{"requests": [{"metrics": {"ttft_ms": 10, "tpot_ms": 5, "itl_ms": [3, 7], "status": "completed"}}],
             "summary": {"gpu_utilization_percent": {"count": 2, "p50": 42}, "tokens_per_second": 100, "duration_ms": 100}}] * 3
    scheduler = [{"execution_records": [{"kind": "decode", "batch_size": 2, "wall_ms": 5},
                                        {"kind": "decode", "batch_size": 1, "wall_ms": 5}],
                  "graph_replays": 1, "eager_decode_steps": 1, "decode_mode_used": "mixed",
                  "graph_captures_this_run": []}] * 3
    result = command.case_metrics(SimpleNamespace(runs=runs), [object()], scheduler)
    assert result["decode_tokens_per_s"] == 300
    assert result["end_to_end_tokens_per_s"] == 100
    assert result["worst_pause_ms"] == 7
    assert result["graph_replay_share"] == 0.5
    assert result["graph_replays"] == result["eager_decode_steps"] == 3
    assert result["gpu_busy_percent"] == 42
    scheduler[0]["execution_records"][0]["wall_ms"] = None
    assert command.case_metrics(SimpleNamespace(runs=runs), [object()], scheduler)["decode_tokens_per_s"] is None


def test_command_workload_matrix_json_and_cleanup(tmp_path, monkeypatch):
    made, closed = [], []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda n: "test GPU")
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda: {"num_alloc_retries": 0})
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 2**30)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 2 * 2**30)
    emptied = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(True))
    monkeypatch.setattr(command, "load_qwen_fp8", lambda **kwargs: SimpleNamespace(model=object(), tokenizer=None))
    prompt_calls = []
    def prompt(tokenizer, length, tag):
        prompt_calls.append((length, tag))
        return tuple([1] * length)
    monkeypatch.setattr(command, "prompt", prompt)

    class Runner:
        def __init__(self, requests, options):
            self.requests, self.options = requests, options
            self.run_summaries = []
            self._graph_scratch_blocks = ()
            self.cache = SimpleNamespace(num_layers=1, num_kv_heads=1, head_dim=1, key_blocks=torch.zeros(1))
            self.allocator = SimpleNamespace(block_size=16)
        def close(self):
            closed.append(self.options)

    def paged(model, requests, **options):
        assert len(made) == len(closed)  # previous case was closed before creation
        made.append((requests, options))
        return Runner(requests, options)
    monkeypatch.setattr(command, "paged", paged)

    class Harness:
        def __init__(self, configuration):
            assert configuration.repetitions == 3 and configuration.warmup_repetitions == 1
            assert configuration.collect_gpu and configuration.respect_arrival_schedule
        def run_trace(self, workload, runner):
            graph = runner.options["cuda_graphs"] and workload.requests[0].max_new_tokens > 1
            summary = {"execution_records": [{"kind": "decode", "batch_size": len(workload.requests), "wall_ms": 10}],
                       "graph_replays": int(graph), "eager_decode_steps": int(not graph),
                       "decode_mode_used": "graph" if graph else "eager", "graph_captures_this_run": [],
                       "graph_fallback_steps_by_reason": {}, "captured_buckets": [1] if graph else [],
                       "capture_ms": {"1": 10} if graph else {}, "graph_pool_memory_bytes_per_bucket": {},
                       "graph_pool_memory_bytes": 100 if graph else 0, "graph_scratch_pages": 0,
                       "graph_scratch_memory_bytes": 0, "graph_static_buffer_memory_bytes_per_bucket": {}}
            runner.run_summaries = [summary] * 4
            runner.last_summary = summary
            runs = [{"requests": [{"metrics": {"ttft_ms": 10, "tpot_ms": 5, "itl_ms": [5], "status": "completed"}}
                                  for _ in workload.requests],
                     "summary": {"gpu_utilization_percent": {"count": 1, "p50": 42},
                                 "tokens_per_second": 100, "duration_ms": 100}}] * 3
            return SimpleNamespace(runs=runs, to_dict=lambda: {"runs": runs, "environment": {"test": True}})
    monkeypatch.setattr(command, "BenchmarkHarness", Harness)
    output = tmp_path / "results"
    command.main(["--output-dir", str(output)])
    data = json.loads((output / "cuda_graphs.json").read_text())
    assert data["hf_reference_check"] is False
    assert data["repetitions"] == 3 and data["warmup_repetitions"] == 1
    assert len(data["sections"]["decode"]) == 14
    assert len(data["sections"]["prefill"]) == 14
    assert len(data["sections"]["serving"]) == 10
    assert len(made) == len(closed) == len(emptied) == 38
    assert {r["batch"] for r in data["sections"]["decode"] if r["prompt_tokens"] == 512} == {1, 2, 4, 8, 16, 32}
    assert {n for n, tag in prompt_calls if tag == "prefill"} == set(command.PREFILL_LENGTHS)
    for entry in data["sections"]["serving"]:
        assert "worst_pause_ms" in entry and "graph_replay_share" in entry
    for section in ("decode", "prefill", "serving"):
        for entry in data["sections"][section]:
            case = json.loads((output / f"{section}_{entry['engine']}.json").read_text())
            assert len(case["cuda_graphs"]["measured_schedulers"]) == 3
            assert case["cuda_graphs"]["warmup_scheduler"]
    serve_requests = made[28][0]
    assert len(serve_requests) == 16
    assert [r.scheduled_arrival_ms for r in serve_requests] == [150 * i for i in range(16)]
    assert sorted(r.prompt_tokens for r in serve_requests) == sorted([128, 256, 512, 1024] * 4)
    long_requests = made[33][0]
    assert [r.prompt_tokens for r in long_requests] == [128, 6144] * 3
    assert [r.scheduled_arrival_ms for r in long_requests] == [400 * i for i in range(6)]
    assert all(r.max_new_tokens == 64 for r in long_requests)
    with pytest.raises(FileExistsError, match="not empty"):
        command.main(["--output-dir", str(output)])


def test_benchmark_refuses_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        command.main(["--output-dir", str(tmp_path / "new")])
    assert not (tmp_path / "new").exists()


def test_charts_render_both_themes_and_prefill_label(tmp_path):
    from minillm_l4.scripts.make_cuda_graph_charts import render
    data = {"sections": {
        "decode": [{"batch": n, "mode": mode, "prompt_tokens": 512, "tpot_p50_ms": 40 if mode == "eager" else 20}
                   for n in (1, 2, 4, 8, 16, 32) for mode in ("eager", "graph")],
        "prefill": [{"prompt_tokens": n, "mode": mode, "ttft_p50_ms": n / 10}
                    for n in (128, 8192) for mode in ("eager", "graph")],
        "serving": [{"workload": workload, "policy": policy, "mode": mode,
                     "tpot_p50_ms": 30, "output_tokens_per_s": 300}
                    for workload in ("serving", "longmix") for policy in ("whole", "mixed_adaptive")
                    for mode in ("eager", "graph")],
    }}
    render(data, tmp_path)
    assert len(list(tmp_path.glob("*.svg"))) == 8
    assert "no decode replay" in (tmp_path / "prefill-ttft-light.svg").read_text()
    assert "Graph" in (tmp_path / "serving-bars-dark.svg").read_text()
