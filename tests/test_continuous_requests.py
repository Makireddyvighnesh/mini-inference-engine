from types import SimpleNamespace
import time

import torch
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

from minillm_l4.benchmarks.core.harness import BenchmarkHarness
from minillm_l4.benchmarks.core.schemas import HarnessConfig, RequestSpec, WorkloadSpec
from minillm_l4.benchmarks.runners.continuous_requests import (
    ContinuousRequestTraceRunner,
)
from minillm_l4.engine.generation.manual import manual_greedy_generate
from minillm_l4.engine.scheduler import ContinuousBatchScheduler, ScheduledRequest


def tiny_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        use_sliding_window=False,
        sliding_window=None,
    )


class ContinuousFakeModel:
    """A small deterministic model that updates a real DynamicCache."""

    def __init__(self, *, call_delay_seconds: float = 0.0) -> None:
        self.config = tiny_config()
        self.config._name_or_path = "continuous-fake"
        self.calls: list[dict] = []
        self.call_delay_seconds = call_delay_seconds

    def __call__(self, **kwargs):
        if self.call_delay_seconds:
            time.sleep(self.call_delay_seconds)
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        batch_size, input_tokens = input_ids.shape
        cache = kwargs.get("past_key_values")
        if cache is None:
            cache = DynamicCache(config=self.config)
        for layer_index in range(self.config.num_hidden_layers):
            keys = torch.zeros(
                (
                    batch_size,
                    self.config.num_key_value_heads,
                    input_tokens,
                    self.config.head_dim,
                ),
                dtype=torch.float32,
                device=input_ids.device,
            )
            cache.update(keys, keys, layer_index)

        next_ids = (input_ids[:, -1] + 1) % self.config.vocab_size
        logits = torch.zeros(
            (batch_size, 1, self.config.vocab_size),
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits.scatter_(2, next_ids[:, None, None], 1.0)
        return SimpleNamespace(logits=logits, past_key_values=cache)


def test_continuous_scheduler_respects_request_and_prefill_budgets() -> None:
    scheduler = ContinuousBatchScheduler(
        (
            ScheduledRequest("a", 0.0, 0),
            ScheduledRequest("b", 0.0, 1),
            ScheduledRequest("c", 0.0, 2),
        ),
        max_batch_size=3,
        max_prefill_tokens=5,
        max_wait_ms=2.0,
    )
    assert [item.request_id for item in scheduler.admit(0.0)] == ["a", "b", "c"]
    assert scheduler.should_wait_for_batch(1.0, active_count=0) is True
    assert scheduler.should_wait_for_batch(2.0, active_count=0) is False
    assert scheduler.should_wait_for_batch(1.0, active_count=1) is False

    first = scheduler.next_batch(
        {"a": 2, "b": 3, "c": 2},
        max_requests=3,
    )
    assert [item.request_id for item in first] == ["a", "b"]
    assert scheduler.ready_count == 1
    assert [item.request_id for item in scheduler.next_batch({"c": 2})] == ["c"]


def test_continuous_runner_admits_new_request_while_decode_is_in_flight() -> None:
    requests = (
        RequestSpec("a", (1, 2), 4, scheduled_arrival_ms=0.0),
        RequestSpec("b", (5, 6, 7), 1, scheduled_arrival_ms=0.0),
        RequestSpec("c", (10, 11), 2, scheduled_arrival_ms=5.0),
    )
    workload = WorkloadSpec(
        name="continuous-mixed",
        seed=17,
        requests=requests,
        device="cpu",
        arrival_pattern="fixed_rate",
    )
    runner = ContinuousRequestTraceRunner(
        ContinuousFakeModel(call_delay_seconds=0.01),
        max_batch_size=3,
        max_prefill_tokens=8,
        max_wait_ms=0.0,
        device="cpu",
    )

    result = BenchmarkHarness(
        HarnessConfig(
            warmup_repetitions=0,
            repetitions=1,
            respect_arrival_schedule=True,
            collect_gpu=False,
            collect_system_telemetry=False,
        )
    ).run_trace(workload, runner)

    records = {
        record["request_id"]: record for record in result.runs[0]["requests"]
    }
    assert records["a"]["outcome"]["generated_token_ids"] == [3, 4, 5, 6]
    assert records["b"]["outcome"]["generated_token_ids"] == [8]
    assert records["c"]["outcome"]["generated_token_ids"] == [12, 13]
    assert result.summary["completed_requests"] == 3
    assert runner.last_summary is not None
    assert runner.last_summary["maximum_active_batch_size"] == 2
    assert runner.last_summary["maximum_concurrent_requests"] == 2
    assert runner.last_summary["maximum_prefill_input_tokens"] == 5
    assert runner.last_summary["prefill_batches_while_decoding"] >= 1
    assert runner.last_summary["requests_prefilled_while_decoding"] >= 1
    assert runner.last_summary["prefill_batch_count"] == 2
    assert runner.last_summary["decode_iteration_count"] >= 3
    assert any(
        record["kind"] == "decode"
        and record["request_ids"] == ["a", "c"]
        for record in runner.last_summary["batch_records"]
    )
    assert runner.last_lifecycles["b"].history == [
        "waiting",
        "prefill",
        "decoding",
        "finished",
    ]
    assert all(
        lifecycle.resources_released
        for lifecycle in runner.last_lifecycles.values()
    )


def test_continuous_runner_cleans_up_finished_rows() -> None:
    requests = (
        RequestSpec("short", (1,), 2),
        RequestSpec("long", (2,), 3),
    )
    workload = WorkloadSpec(
        name="cleanup",
        seed=17,
        requests=requests,
        device="cpu",
    )
    runner = ContinuousRequestTraceRunner(
        ContinuousFakeModel(),
        max_batch_size=2,
        max_prefill_tokens=8,
        device="cpu",
    )
    result = BenchmarkHarness(
        HarnessConfig(
            warmup_repetitions=0,
            repetitions=1,
            respect_arrival_schedule=True,
            collect_gpu=False,
            collect_system_telemetry=False,
        )
    ).run_trace(workload, runner)

    assert result.summary["completed_requests"] == 2
    decode_records = [
        record
        for record in runner.last_summary["batch_records"]  # type: ignore[index]
        if record["kind"] == "decode"
    ]
    assert decode_records[0]["request_ids"] == ["short", "long"]
    assert all(record["request_ids"] == ["long"] for record in decode_records[1:])


def test_continuous_runner_matches_real_qwen_cache_execution() -> None:
    model = Qwen3ForCausalLM(tiny_config()).eval()
    requests = (
        RequestSpec("a", (1, 2, 3), 4, scheduled_arrival_ms=0.0),
        RequestSpec("b", (4, 5), 2, scheduled_arrival_ms=0.1),
    )
    workload = WorkloadSpec(
        name="tiny-qwen3-continuous",
        seed=17,
        requests=requests,
        device="cpu",
        arrival_pattern="fixed_rate",
    )
    runner = ContinuousRequestTraceRunner(
        model,
        max_batch_size=2,
        max_prefill_tokens=8,
        device="cpu",
    )
    result = BenchmarkHarness(
        HarnessConfig(
            warmup_repetitions=0,
            repetitions=1,
            respect_arrival_schedule=True,
            collect_gpu=False,
            collect_system_telemetry=False,
        )
    ).run_trace(workload, runner)

    expected: list[tuple[int, ...]] = []
    for request in requests:
        inputs = {
            "input_ids": torch.tensor([request.prompt_token_ids]),
            "attention_mask": torch.ones(
                (1, request.prompt_tokens), dtype=torch.long
            ),
            "position_ids": torch.arange(request.prompt_tokens).reshape(1, -1),
        }
        reference = manual_greedy_generate(
            model,
            inputs,
            output_tokens=request.max_new_tokens,
        )
        expected.append(tuple(int(value) for value in reference.row(0).tolist()))

    actual = [
        tuple(int(value) for value in record["outcome"]["generated_token_ids"])
        for record in result.runs[0]["requests"]
    ]
    assert actual == expected
