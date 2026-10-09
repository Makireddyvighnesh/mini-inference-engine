from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import builtins
import io
import json
import pytest
import torch
import yaml
from transformers import Qwen3Config, Qwen3ForCausalLM

from minillm_l4.benchmarks.commands import run_prefill_decode_sweep as command
from minillm_l4.benchmarks.core.harness import BenchmarkHarness
from minillm_l4.benchmarks.core.schemas import HarnessConfig, RequestSpec, WorkloadSpec
from minillm_l4.benchmarks.runners.phase_sweep import MeteredRunner, StaticPagedBatchRunner, StaticPagedTraceRunner
from minillm_l4.engine.generation.manual import manual_greedy_generate


def model():
    torch.manual_seed(47)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=4, max_position_embeddings=64, use_sliding_window=False, sliding_window=None)).eval()


class Tokenizer:
    def encode(self, text, *, add_special_tokens=False):
        return [ord(c) % 30 + 1 for c in text]


@pytest.fixture
def reference_corpus(tmp_path, monkeypatch):
    corpus = tmp_path / "references"
    corpus.mkdir()
    for bucket in ("short", "medium", "long"):
        (corpus / f"{bucket}.json").write_text(json.dumps({
            "model_id": command.MODEL_ID, "model_revision": command.MODEL_REVISION,
            "requests": [{"request_id": f"fixture-{bucket}", "generated_token_ids": [1, 2]}],
        }) + "\n")
    real_corpus = command.REFERENCE_DIR
    monkeypatch.setattr(command, "REFERENCE_DIR", corpus)

    def hidden(path):
        return isinstance(path, (str, Path)) and Path(path).absolute().is_relative_to(real_corpus.absolute())

    # Emulate a checkout without the git-ignored corpus without touching results/.
    for module in (builtins, io):
        original_open = module.open

        def guarded_open(path, *args, _open=original_open, **kwargs):
            if hidden(path):
                raise FileNotFoundError(path)
            return _open(path, *args, **kwargs)

        monkeypatch.setattr(module, "open", guarded_open)
    original_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: False if hidden(path) else original_exists(path))
    assert not real_corpus.exists()
    with pytest.raises(FileNotFoundError):
        (real_corpus / "short.json").read_text()
    return corpus


def test_plan_covers_all_requested_lengths_and_keeps_1048_literal():
    data = command.configuration(command.parse_args([]))
    assert data["workloads"]["prompt_lengths"] == [128, 256, 512, 1024, 2048, 4096, 8192]
    assert data["workloads"]["generation_lengths"] == [128, 512, 1048]
    assert data["workloads"]["batch_sizes"] == [1, 2, 4, 8]
    assert len(command.planned_cases(data)) == 448
    prompts = command.build_prompts(Tokenizer(), data)
    assert all(len(tokens) == int(length) for length, tokens in prompts.items())
    for case in command.planned_cases(data):
        workload = command.build_workload(case, prompts, data)
        assert max(r.max_new_tokens for r in workload.requests) == case["generation_cap"]
        assert all(r.max_new_tokens <= 1048 for r in workload.requests)
        if case["mode"] in {"isolated", "prefill"}:
            assert len(workload.requests) == case["batch_limit"]
            assert all(r.prompt_tokens == case["prompt_tokens"] for r in workload.requests)


def test_mixed_traffic_is_identical_across_static_and_continuous_policies():
    data = command.configuration(command.parse_args([]))
    prompts = command.build_prompts(Tokenizer(), data)
    signatures = []
    for mode in ["static", "continuous", "chunked_128", "chunked_256"]:
        case = {"key": mode, "mode": mode, "prompt_tokens": 8192, "generation_cap": 1048, "batch_limit": 4}
        workload = command.build_workload(case, prompts, data)
        signatures.append([(r.prompt_token_ids, r.max_new_tokens, r.scheduled_arrival_ms) for r in workload.requests])
    assert all(signature == signatures[0] for signature in signatures)
    assert {len(r[0]) for r in signatures[0]} == {128, 8192}
    assert {r[1] for r in signatures[0]} == {128, 1048}


def test_mixed_request_population_is_matched_across_batch_limits():
    data = command.configuration(command.parse_args([]))
    prompts = command.build_prompts(Tokenizer(), data)
    signatures = []
    for batch in [1, 2, 4]:
        case = {"key": str(batch), "mode": "static", "prompt_tokens": 8192, "generation_cap": 1048, "batch_limit": batch}
        requests = command.build_workload(case, prompts, data).requests
        signatures.append([(r.prompt_token_ids, r.max_new_tokens, r.scheduled_arrival_ms) for r in requests])
    assert all(s == signatures[0] for s in signatures)
    assert len(signatures[0]) == 9


def test_static_paged_mixed_batch_matches_dense_reference_and_frees_pages():
    reference = model()
    backend = StaticPagedBatchRunner(deepcopy(reference), block_size=2, num_blocks=32,
        max_batch_size=3, max_prefill_tokens=16, device="cpu", enable_prefix=False, decode_sdpa_compat=True)
    requests = (RequestSpec("a", (1, 2, 3, 4), 5), RequestSpec("b", (5, 6), 2), RequestSpec("c", (7, 8, 9), 3))
    wrapped = MeteredRunner(backend, backend.model, "cpu")
    result = BenchmarkHarness(HarnessConfig(warmup_repetitions=0, repetitions=1, collect_gpu=False,
        collect_system_telemetry=False)).run_batched(WorkloadSpec(name="test", seed=47, device="cpu", requests=requests), 3, wrapped)
    assert result.summary["completed_requests"] == 3
    for r, row in zip(requests, result.runs[0]["requests"], strict=True):
        expected = manual_greedy_generate(reference, {"input_ids": torch.tensor([r.prompt_token_ids]),
            "attention_mask": torch.ones((1, r.prompt_tokens), dtype=torch.long)}, output_tokens=r.max_new_tokens)
        assert row["outcome"]["generated_token_ids"] == expected.row(0).tolist()
    summary = wrapped.run_summaries[0]
    assert summary["prefill"]["forward_count"] == 1
    assert summary["decode"]["forward_count"] == 4
    assert summary["decode"]["input_tokens"] == 7
    assert summary["prefill"]["cuda_elapsed_total_ms"] is None
    assert backend.allocator.free_block_count == backend.allocator.num_blocks
    assert all(s.resources_released for s in backend.last_lifecycles.values())
    backend.close()


def test_phase_decode_rate_uses_only_post_prefill_tokens():
    backend = StaticPagedBatchRunner(model(), block_size=2, num_blocks=16,
        max_batch_size=2, max_prefill_tokens=16, device="cpu", enable_prefix=False)
    wrapped = MeteredRunner(backend, backend.model, "cpu")
    requests = (RequestSpec("a", (1, 2, 3, 4), 4), RequestSpec("b", (1, 2, 3, 4), 4))
    result = BenchmarkHarness(HarnessConfig(warmup_repetitions=1, repetitions=2, collect_gpu=False,
        collect_system_telemetry=False)).run_batched(WorkloadSpec(name="phase", seed=47, device="cpu", requests=requests), 2, wrapped)
    phase = command.phase_summary(result, wrapped.run_summaries[-2:])
    assert phase["prefill_request_wall_ms_p50"] > 0
    assert phase["decode_window_wall_ms_p50"] > 0
    for run in result.runs:
        times = [e["timestamp_ns"] for e in run["events"] if e["event"] == "token_ready"]
        elapsed = (max(times) - min(times)) / 1e9
        assert elapsed > 0
    assert all(f["decode"]["input_tokens"] == 6 for f in wrapped.run_summaries)
    backend.close()


def test_static_trace_waits_for_selected_group_before_admitting_new_arrival():
    backend = StaticPagedBatchRunner(model(), block_size=2, num_blocks=16,
        max_batch_size=2, max_prefill_tokens=16, device="cpu", enable_prefix=False)
    requests = (RequestSpec("a", (1, 2, 3), 5), RequestSpec("b", (4, 5), 2, scheduled_arrival_ms=10.0))
    runner = StaticPagedTraceRunner(backend)
    result = BenchmarkHarness(HarnessConfig(warmup_repetitions=0, repetitions=1, respect_arrival_schedule=True,
        collect_gpu=False, collect_system_telemetry=False)).run_trace(WorkloadSpec(name="static", seed=47, device="cpu", requests=requests), runner)
    events = result.runs[0]["events"]
    a_done = next(e["timestamp_ns"] for e in events if e["request_id"] == "a" and e["event"] == "completion")
    b_admit = next(e["timestamp_ns"] for e in events if e["request_id"] == "b" and e["event"] == "admission")
    assert b_admit >= a_done
    assert len(runner.last_summary["batches"]) == 2
    backend.close()


def test_gpu_guard_only_observes_other_jobs(monkeypatch):
    monkeypatch.setattr(command.os, "getpid", lambda: 7)
    monkeypatch.setattr(command.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="7, python, 4000\n8, VLLM::EngineCore, 14000\n"))
    assert command.gpu_occupants() == [{"pid": 8, "process_name": "VLLM::EngineCore", "memory_mib": "14000"}]


@pytest.mark.parametrize("argv", [["--generation-lengths", "1049"], ["--batch-sizes", "0"], ["--prompt-lengths", "-1"], ["--repetitions", "0"]])
def test_invalid_configuration_fails_before_loading_weights(argv):
    with pytest.raises(ValueError):
        command.configuration(command.parse_args(argv))


def test_cpu_end_to_end_saves_separate_phases_exact_reference_and_resume(tmp_path, monkeypatch, reference_corpus):
    config = command.configuration(command.parse_args(["--prompt-lengths", "4", "8", "--generation-lengths", "6", "--batch-sizes", "1", "2", "--repetitions", "1", "--warmup-repetitions", "0"]))
    config["model"]["device"] = "cpu"
    config["workloads"]["mixed_short_output_tokens"] = 2
    config["workloads"]["mixed_arrival_interval_ms"] = 0.1
    config["engine"]["block_size"] = 2
    config["benchmark"].update(collect_gpu=False, collect_system_telemetry=False)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(command.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    reference_model = model()
    monkeypatch.setattr(command, "load_qwen_fp8", lambda **kwargs: SimpleNamespace(model=reference_model, metadata=lambda: {"test_fixture": True}))
    output, report = tmp_path / "out", tmp_path / "report.md"
    argv = ["--config", str(path), "--output-dir", str(output), "--markdown-output", str(report)]
    command.main(argv)
    manifest = json.loads((output / "phase_sweep_manifest.json").read_text())
    for bucket in ("short", "medium", "long"):
        copied = output / "source_snapshot/inputs/references" / f"{bucket}.json"
        assert copied.read_bytes() == (reference_corpus / f"{bucket}.json").read_bytes()
    assert len(manifest["completed_cases"]) == 24
    assert manifest["status"] == "completed"
    assert all(c["status"] == "pass" for c in manifest["completed_cases"])
    isolated = [c for c in manifest["completed_cases"] if c["mode"] == "isolated"]
    assert len(isolated) == 4
    assert all(c["phase_summary"]["decode_window_has_interleaved_prefill"] is False for c in isolated)
    assert "Isolated prefill and decode" in report.read_text()
    assert "Matched mixed traces" in report.read_text()
    before = (output / "hf_references.json").read_bytes()
    command.main([*argv, "--resume"])
    assert (output / "hf_references.json").read_bytes() == before
    assert len(json.loads((output / "phase_sweep_manifest.json").read_text())["completed_cases"]) == 24
