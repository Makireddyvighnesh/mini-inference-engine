from __future__ import annotations

from copy import deepcopy

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from minillm_l4.benchmarks.core.harness import BenchmarkHarness
from minillm_l4.benchmarks.core.schemas import HarnessConfig, RequestSpec, WorkloadSpec
from minillm_l4.benchmarks.runners.continuous_prefix import ContinuousPrefixPagedRunner
from minillm_l4.benchmarks.commands.run_prefix_trace import build_shared_prefix_workload
from minillm_l4.configs.loader import load_yaml_config
from minillm_l4.engine.generation.manual import manual_greedy_generate
from minillm_l4.engine.scheduler import ContinuousBatchScheduler, ScheduledRequest


def _model() -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=4,
        max_position_embeddings=64, use_sliding_window=False,
        sliding_window=None,
    )).eval()


def _run(runner: ContinuousPrefixPagedRunner, requests: tuple[RequestSpec, ...]):
    workload = WorkloadSpec(
        name="shared-prefix-trace", seed=47, requests=requests,
        device="cpu", arrival_pattern="fixed_rate",
    )
    return BenchmarkHarness(HarnessConfig(
        repetitions=1, warmup_repetitions=0, respect_arrival_schedule=True,
        collect_gpu=False, collect_system_telemetry=False,
    )).run_trace(workload, runner)


def test_continuous_prefix_matches_reference_with_mixed_arrivals_and_decode() -> None:
    torch.manual_seed(47)
    reference = _model()
    runner = ContinuousPrefixPagedRunner(
        deepcopy(reference), block_size=2, num_blocks=24,
        max_batch_size=3, max_prefill_tokens=2, device="cpu",
        decode_mode="graph",  # safely falls back for dynamic membership
    )
    requests = (
        RequestSpec("a", (1, 2, 3, 4, 5, 6), 5, scheduled_arrival_ms=0),
        RequestSpec("b", (1, 2, 3, 4, 7, 8), 3, scheduled_arrival_ms=0),
        RequestSpec("c", (1, 2, 3, 4, 9, 10), 4, scheduled_arrival_ms=0.1),
        RequestSpec("d", (11, 12, 13), 2, scheduled_arrival_ms=0.2),
    )
    result = _run(runner, requests)
    assert result.summary["completed_requests"] == 4
    for request, row in zip(requests, result.runs[0]["requests"], strict=True):
        expected = manual_greedy_generate(
            reference,
            {"input_ids": torch.tensor([request.prompt_token_ids]),
             "attention_mask": torch.ones((1, request.prompt_tokens), dtype=torch.long)},
            output_tokens=request.max_new_tokens,
        )
        assert row["outcome"]["generated_token_ids"] == expected.row(0).tolist()
    summary = runner.last_summary
    assert summary is not None
    assert summary["hits"] >= 2
    assert summary["reused_tokens"] >= 8
    assert summary["prefills_while_decoding"] >= 1
    assert summary["budget_deferred_admissions"] >= 1
    assert summary["maximum_decode_batch_size"] >= 2
    assert summary["decode_mode_used"] == "eager"
    assert summary["graph_fallback_reason"]
    assert summary["active_request_blocks_after_run"] == 0
    assert all(item.resources_released for item in runner.last_lifecycles.values())
    runner.close()
    assert runner.allocator.free_block_count == runner.allocator.num_blocks


def test_uncached_control_matches_cached_trace() -> None:
    torch.manual_seed(48)
    model = _model()
    requests = (
        RequestSpec("a", (1, 2, 3, 4, 5, 6), 3),
        RequestSpec("b", (1, 2, 3, 4, 7, 8), 3),
    )
    cached = ContinuousPrefixPagedRunner(
        deepcopy(model), block_size=2, num_blocks=16,
        max_batch_size=2, max_prefill_tokens=8, device="cpu",
    )
    uncached = ContinuousPrefixPagedRunner(
        deepcopy(model), block_size=2, num_blocks=16,
        max_batch_size=2, max_prefill_tokens=8, device="cpu", enable_prefix=False,
    )
    rows = []
    for runner in (cached, uncached):
        rows.append([
            item["outcome"]["generated_token_ids"]
            for item in _run(runner, requests).runs[0]["requests"]
        ])
        runner.close()
    assert rows[0] == rows[1]
    assert cached.last_summary["hits"] >= 1
    assert uncached.last_summary["hits"] == 0


def test_capacity_rejection_and_cancellation_release_resources() -> None:
    runner = ContinuousPrefixPagedRunner(
        _model(), block_size=2, num_blocks=4,
        max_batch_size=2, max_prefill_tokens=8, device="cpu",
        cancel_request_ids=("cancel",),
    )
    result = _run(runner, (
        RequestSpec("cancel", (1, 2, 3, 4), 2),
        RequestSpec("too-large", tuple(range(1, 10)), 2),
        RequestSpec("fits", (1, 2, 3, 4), 2),
    ))
    statuses = [row["outcome"]["status"] for row in result.runs[0]["requests"]]
    assert statuses == ["cancelled", "failed", "completed"]
    assert all(item.resources_released for item in runner.last_lifecycles.values())
    runner.close()
    assert runner.allocator.free_block_count == runner.allocator.num_blocks


def test_capacity_deferred_request_keeps_fifo_order() -> None:
    items = tuple(ScheduledRequest(str(index), 0, index) for index in range(3))
    scheduler = ContinuousBatchScheduler(
        items, max_batch_size=2, max_prefill_tokens=4,
    )
    scheduler.admit(0)
    first = scheduler.next_batch({str(index): 2 for index in range(3)}, max_requests=1)
    scheduler.defer_front(first)
    assert [item.request_id for item in scheduler.next_batch(
        {str(index): 2 for index in range(3)}
    )] == ["0", "1"]


def test_live_capacity_pressure_defers_admission_until_decode_releases_pages() -> None:
    runner = ContinuousPrefixPagedRunner(
        _model(), block_size=2, num_blocks=3,
        max_batch_size=2, max_prefill_tokens=8, device="cpu",
    )
    result = _run(runner, (
        RequestSpec("first", (1, 2, 3, 4), 3),
        RequestSpec("second", (5, 6), 2),
    ))
    assert result.summary["completed_requests"] == 2
    assert runner.last_summary["deferred_admissions"] >= 1
    assert runner.last_summary["active_request_blocks_after_run"] == 0
    runner.close()
    assert runner.allocator.free_block_count == runner.allocator.num_blocks


def test_shared_prefix_workload_is_deterministic_and_respects_reuse_rate() -> None:
    class Tokenizer:
        vocab_size = 256

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert not add_special_tokens
            return [ord(character) % 256 for character in text]

    options = dict(
        rate=0.5, count=4, prefix_tokens=8, suffix_tokens=2,
        output_tokens=4, arrival_interval_ms=0.5, seed=17, device="cpu",
    )
    first = build_shared_prefix_workload(Tokenizer(), **options)
    second = build_shared_prefix_workload(Tokenizer(), **options)
    assert first.to_dict() == second.to_dict()
    assert [request.category for request in first.requests] == [
        "shared_prefix", "shared_prefix", "unique_prefix", "unique_prefix"
    ]
    assert first.requests[0].prompt_token_ids[:8] == first.requests[1].prompt_token_ids[:8]
    assert first.requests[0].prompt_token_ids[8:] != first.requests[1].prompt_token_ids[8:]
    assert [request.max_new_tokens for request in first.requests] == [4, 2, 4, 2]


def test_prefix_benchmark_yaml_loads() -> None:
    from pathlib import Path

    config = Path(__file__).resolve().parents[1] / "configs/workloads/qwen3_fp8_prefix.yaml"
    payload = load_yaml_config(config, expected_phase=7)
    assert payload["workload"]["reuse_rates"] == [0.0, 0.5, 1.0]


def test_repetitions_reset_prefix_state_without_leaking_request_pages() -> None:
    runner = ContinuousPrefixPagedRunner(
        _model(), block_size=2, num_blocks=12,
        max_batch_size=2, max_prefill_tokens=8, device="cpu",
    )
    workload = WorkloadSpec(
        name="repeat-prefix", seed=17, device="cpu",
        requests=(
            RequestSpec("a", (1, 2, 3, 4), 2),
            RequestSpec("b", (1, 2, 5, 6), 2),
        ),
    )
    result = BenchmarkHarness(HarnessConfig(
        warmup_repetitions=1, repetitions=2,
        respect_arrival_schedule=True, collect_gpu=False,
        collect_system_telemetry=False,
    )).run_trace(workload, runner)
    assert len(result.runs) == 2
    assert len(runner.run_summaries) == 3
    assert [summary["hits"] for summary in runner.run_summaries] == [1, 1, 1]
    assert all(summary["active_request_blocks_after_run"] == 0
               for summary in runner.run_summaries)
    assert all(run["summary"]["completed_requests"] == 2 for run in result.runs)
    runner.close()
    assert runner.allocator.free_block_count == runner.allocator.num_blocks
