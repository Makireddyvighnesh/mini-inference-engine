from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import torch

from minillm_l4.benchmarks.core.harness import BenchmarkHarness, RequestEventRecorder
from minillm_l4.benchmarks.runners.huggingface_baseline import (
    HuggingFaceGreedyBatchRunner,
    build_hf_workload,
    resolve_model_source,
    verify_or_write_reference,
)
from minillm_l4.benchmarks.core.schemas import HarnessConfig, RequestSpec, WorkloadSpec


class FakeGenerateModel:
    def __init__(self) -> None:
        self.generation_config = SimpleNamespace(
            do_sample=True,
            max_new_tokens=None,
            min_new_tokens=None,
            use_cache=False,
            eos_token_id=2,
        )
        self.config = SimpleNamespace(_name_or_path="fake-qwen")
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        output_tokens = int(kwargs["generation_config"].max_new_tokens)
        streamer = kwargs["streamer"]
        streamer.put(input_ids.cpu())
        generated: list[torch.Tensor] = []
        for index in range(output_tokens):
            next_token = torch.full(
                (input_ids.shape[0],),
                10 + index,
                dtype=input_ids.dtype,
            )
            generated.append(next_token[:, None])
            streamer.put(next_token)
        streamer.end()
        return torch.cat((input_ids, torch.cat(generated, dim=1)), dim=1)


def _requests(count: int = 3) -> tuple[RequestSpec, ...]:
    return tuple(
        RequestSpec(
            request_id=f"request-{index}",
            prompt_token_ids=(1, 2, 3, 4),
            max_new_tokens=3,
            category="tiny",
        )
        for index in range(count)
    )


def test_huggingface_runner_records_streamed_tokens_and_static_batches() -> None:
    model = FakeGenerateModel()
    runner = HuggingFaceGreedyBatchRunner(model, device="cpu")
    workload = WorkloadSpec(
        name="baseline_fake",
        seed=7,
        requests=_requests(),
        model_id="fake-qwen",
        model_revision="fake-revision",
        dtype="fp8",
        device="cpu",
    )
    result = BenchmarkHarness(
        HarnessConfig(
            warmup_repetitions=1,
            repetitions=2,
            collect_gpu=False,
            collect_system_telemetry=False,
            timer_overhead_iterations=5,
        ),
        benchmark_name="minillm_l4_hf_baseline",
    ).run_batched(workload, 2, runner)

    assert result.to_dict()["benchmark"] == "minillm_l4_hf_baseline"
    assert result.diagnostics["batch_size"] == 2
    assert len(model.calls) == (1 + 2) * 2
    assert result.summary["completed_requests"] == 6
    assert result.summary["metrics"]["ttft_ms"]["count"] == 6
    assert result.summary["metrics"]["itl_ms"]["count"] == 12
    for run in result.runs:
        for request in run["requests"]:
            assert request["outcome"]["generated_token_ids"] == [10, 11, 12]
            events = [
                event
                for event in run["events"]
                if event["request_id"] == request["request_id"]
            ]
            assert [
                event["event"] for event in events if event["event"] == "token_ready"
            ] == ["token_ready", "token_ready", "token_ready"]
            assert {event["event"] for event in events} >= {
                "prefill_start",
                "prefill_end",
                "first_token_ready",
                "token_sent",
                "completion",
            }


def test_static_batch_rejects_mixed_prompt_lengths() -> None:
    model = FakeGenerateModel()
    runner = HuggingFaceGreedyBatchRunner(model, device="cpu")
    first = RequestSpec(
        request_id="first",
        prompt_token_ids=(1, 2),
        max_new_tokens=2,
    )
    second = RequestSpec(
        request_id="second",
        prompt_token_ids=(1, 2, 3),
        max_new_tokens=2,
    )
    recorders = [
        RequestEventRecorder(request.request_id, run_started_ns=0)
        for request in (first, second)
    ]

    try:
        runner((first, second), recorders)
    except ValueError as error:
        assert "equal prompt lengths" in str(error)
    else:
        raise AssertionError("mixed prompt lengths should be rejected")


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del text, add_special_tokens
        return list(range(1, 100))


def test_hf_workload_uses_tokenizer_and_preserves_exact_lengths(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "samples.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps(
                {
                    "sample_id": f"sample-{index}",
                    "category": "test",
                    "target_prompt_tokens": 5,
                    "target_output_tokens": 4,
                    "seed_text": "Explain the test.",
                }
            )
            for index in range(2)
        )
        + "\n",
        encoding="utf-8",
    )

    workload = build_hf_workload(
        FakeTokenizer(),
        dataset,
        bucket_name="tiny",
        prompt_tokens=5,
        output_tokens=3,
        count=2,
        seed=19,
        model_id="fake-qwen",
        revision="fake-revision",
        device="cpu",
    )

    assert [request.prompt_tokens for request in workload.requests] == [5, 5]
    assert [request.max_new_tokens for request in workload.requests] == [3, 3]
    assert workload.requests[0].metadata["source_sample_id"] == "sample-0"
    assert workload.metadata["tokenization"].startswith("Hugging Face tokenizer")


def test_reference_corpus_is_created_then_checked(tmp_path: Path) -> None:
    model = FakeGenerateModel()
    workload = WorkloadSpec(
        name="reference",
        seed=3,
        requests=_requests(1),
        model_id="fake-qwen",
        model_revision="fake-revision",
        dtype="fp8",
        device="cpu",
    )
    result = BenchmarkHarness(
        HarnessConfig(
            repetitions=2,
            warmup_repetitions=0,
            collect_gpu=False,
            collect_system_telemetry=False,
            timer_overhead_iterations=5,
        ),
        benchmark_name="minillm_l4_hf_baseline",
    ).run_batched(workload, 1, HuggingFaceGreedyBatchRunner(model, device="cpu"))
    reference_path = tmp_path / "reference.json"

    created = verify_or_write_reference(result, reference_path)
    checked = verify_or_write_reference(result, reference_path)

    assert created["status"] == "pass"
    assert created["reference_created"] is True
    assert checked["reference_checked"] is True
    assert checked["reference_match"] is True
    assert checked["reference_created"] is False


def test_reference_check_accepts_a_measured_request_subset(tmp_path: Path) -> None:
    model = FakeGenerateModel()
    full_workload = WorkloadSpec(
        name="reference",
        seed=3,
        requests=_requests(2),
        model_id="fake-qwen",
        model_revision="fake-revision",
        dtype="fp8",
        device="cpu",
    )
    config = HarnessConfig(
        repetitions=1,
        warmup_repetitions=0,
        collect_gpu=False,
        collect_system_telemetry=False,
        timer_overhead_iterations=5,
    )
    full_result = BenchmarkHarness(
        config,
        benchmark_name="minillm_l4_hf_baseline",
    ).run_batched(full_workload, 1, HuggingFaceGreedyBatchRunner(model, device="cpu"))
    reference_path = tmp_path / "reference.json"
    assert verify_or_write_reference(full_result, reference_path)["status"] == "pass"

    subset_workload = WorkloadSpec(
        name="reference",
        seed=3,
        requests=full_workload.requests[:1],
        model_id="fake-qwen",
        model_revision="fake-revision",
        dtype="fp8",
        device="cpu",
    )
    subset_result = BenchmarkHarness(
        config,
        benchmark_name="minillm_l4_hf_baseline",
    ).run_batched(subset_workload, 1, HuggingFaceGreedyBatchRunner(model, device="cpu"))
    checked = verify_or_write_reference(subset_result, reference_path)

    assert checked["status"] == "pass"
    assert checked["reference_match"] is True
    assert checked["reference_scope"] == "measured_subset"


def test_local_model_path_resolution_checks_required_files(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"fixture")

    source = resolve_model_source(
        "fake/model",
        "0123456789abcdef0123456789abcdef01234567",
        model_path=tmp_path,
    )

    assert source == str(tmp_path.resolve())
