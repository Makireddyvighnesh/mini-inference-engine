from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from transformers import Qwen3Config, Qwen3ForCausalLM

from minillm_l4.benchmarks.commands import run_chunked_prefill as command
from minillm_l4.benchmarks.core.schemas import RequestSpec, WorkloadSpec
from minillm_l4.engine.generation.manual import manual_greedy_generate


def _fixture(tmp_path, monkeypatch):
    torch.manual_seed(47)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=4, max_position_embeddings=64, use_sliding_window=False,
        sliding_window=None,
    )).eval()
    requests = (
        RequestSpec("baseline-short-000", (1, 2, 3, 4), 3),
        RequestSpec("baseline-medium-000", (5, 6, 7, 8, 9), 3),
        RequestSpec("baseline-long-000", tuple(range(1, 18)), 3, scheduled_arrival_ms=0.1),
    )
    workload = WorkloadSpec(name="command-test", seed=47, device="cpu", requests=requests,
        model_id=command.MODEL_ID, model_revision=command.MODEL_REVISION)
    refs = tmp_path / "references"
    refs.mkdir()
    for bucket, request in zip(["short", "medium", "long"], requests, strict=True):
        generated = manual_greedy_generate(model, {
            "input_ids": torch.tensor([request.prompt_token_ids]),
            "attention_mask": torch.ones((1, request.prompt_tokens), dtype=torch.long),
        }, output_tokens=request.max_new_tokens)
        (refs / f"{bucket}.json").write_text(json.dumps({
            "model_id": command.MODEL_ID, "model_revision": command.MODEL_REVISION,
            "requests": {request.request_id: {
                "prompt_sha256": request.prompt_sha256, "max_new_tokens": request.max_new_tokens,
                "generated_token_ids": generated.row(0).tolist(),
            }},
        }))
    config = command.resolved_configuration(command.parse_args([]))
    config["model"]["device"] = "cpu"
    config["engine"].update(num_blocks=32, block_size=2, chunk_sizes=[0, 2, 3], max_batch_size=3, max_prefill_tokens=3)
    config["benchmark"].update(repetitions=2, warmup_repetitions=1, collect_gpu=False, collect_system_telemetry=False)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(command, "load_qwen_fp8", lambda **kwargs: SimpleNamespace(model=model, tokenizer=None, metadata=lambda: {"test_fixture": True}))
    monkeypatch.setattr(command, "build_trace_workloads", lambda *args, **kwargs: {"mixed": workload})
    argv = ["--config", str(path), "--workload", "mixed", "--reference-dir", str(refs),
            "--output-dir", str(tmp_path / "results"), "--markdown-output", str(tmp_path / "report.md")]
    return argv, refs, model


def test_command_saves_matched_raw_results_markdown_and_actual_source(tmp_path, monkeypatch):
    argv, refs, _ = _fixture(tmp_path, monkeypatch)
    original_refs = {p.name: p.read_bytes() for p in refs.glob("*.json")}
    command.main(argv)
    manifest = json.loads((tmp_path / "results/chunked_prefill_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert [case["chunk_size"] for case in manifest["cases"]] == [0, 2, 3]
    for case in manifest["cases"]:
        assert case["correctness"]["status"] == "pass" and case["matches_unchunked"]
        assert len(case["scheduler_runs"]) == 2
        data = json.loads(Path(case["result"]).read_text())
        assert len(data["runs"]) == 2 and len(data["warmup_durations_ms"]) == 1
        assert data["chunked_prefill"]["correctness"]["status"] == "pass"
        assert Path(case["result"]).with_name(Path(case["result"]).stem + "_events.jsonl").exists()
    assert "pass / pass" in (tmp_path / "report.md").read_text()
    provenance = json.loads((tmp_path / "results/run_provenance.json").read_text())
    source = "minillm_l4/engine/generation/chunked_prefill.py"
    import hashlib
    snapshot = tmp_path / "results/source_snapshot" / source
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == provenance["sha256"][source]
    assert snapshot.read_bytes() == (command.PROJECT_ROOT / "engine/generation/chunked_prefill.py").read_bytes()
    assert {p.name: p.read_bytes() for p in refs.glob("*.json")} == original_refs
    with pytest.raises(FileExistsError, match="empty output directory"):
        command.main(argv)


def test_reference_failure_is_saved_and_never_self_certified(tmp_path, monkeypatch):
    argv, refs, _ = _fixture(tmp_path, monkeypatch)
    path = refs / "long.json"
    data = json.loads(path.read_text())
    data["requests"]["baseline-long-000"]["generated_token_ids"][0] = -1
    path.write_text(json.dumps(data))
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="correctness gate failed"):
        command.main(argv)
    manifest = json.loads((tmp_path / "results/chunked_prefill_manifest.json").read_text())
    assert manifest["status"] == "failed_correctness"
    assert all(case["correctness"]["status"] == "fail" for case in manifest["cases"])
    assert "FAIL / pass" in (tmp_path / "report.md").read_text()
    assert path.read_bytes() == before


def test_failed_model_run_still_exports_diagnostics_and_missing_metrics(tmp_path, monkeypatch):
    argv, _, model = _fixture(tmp_path, monkeypatch)

    def fail(**kwargs):
        raise RuntimeError("test model failure")

    model.forward = fail
    with pytest.raises(RuntimeError, match="correctness gate failed"):
        command.main(argv)
    manifest = json.loads((tmp_path / "results/chunked_prefill_manifest.json").read_text())
    assert manifest["status"] == "failed_correctness"
    assert all(len(case["scheduler_runs"]) == 2 for case in manifest["cases"])
    assert all(s["status"] == "failed" and s["active_request_blocks_after_run"] == 0 for case in manifest["cases"] for s in case["scheduler_runs"])
    assert "FAIL / FAIL" in (tmp_path / "report.md").read_text()
    assert "—" in (tmp_path / "report.md").read_text()


def test_missing_or_wrong_reference_fails_without_creating_a_corpus(tmp_path):
    with pytest.raises(FileNotFoundError):
        command.validate_reference_identity(tmp_path / "missing")
    assert not (tmp_path / "missing").exists()
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "short.json").write_text(json.dumps({"model_id": "wrong", "model_revision": "wrong", "requests": {"a": {}}}))
    with pytest.raises(ValueError, match="pinned reference"):
        command.validate_reference_identity(refs)


def test_config_dry_run_includes_control_and_applies_chunk_budget_overrides(capsys):
    command.main(["--dry-run", "--chunk-sizes", "3", "3", "2", "--max-prefill-tokens", "1", "--enable-prefix"])
    payload = json.loads(capsys.readouterr().out)["configuration"]
    assert payload["phase"] == 8
    assert payload["engine"]["chunk_sizes"] == [0, 3, 2]
    assert payload["engine"]["max_prefill_tokens"] == 1
    assert payload["engine"]["enable_prefix"] is True


@pytest.mark.parametrize("argv", [["--chunk-sizes", "-1"], ["--max-prefill-tokens", "0"], ["--max-batch-size", "0"], ["--warmup-repetitions", "-1"], ["--repetitions", "0"]])
def test_invalid_cli_configuration_fails_before_model_loading(argv):
    with pytest.raises(ValueError):
        command.resolved_configuration(command.parse_args(argv))
