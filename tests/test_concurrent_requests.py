from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config

from minillm_l4.benchmarks.core.harness import BenchmarkHarness
from minillm_l4.benchmarks.core.schemas import HarnessConfig, RequestSpec, WorkloadSpec
from minillm_l4.benchmarks.runners.concurrent_requests import StaticRequestTraceRunner
from minillm_l4.engine.request import RequestLifecycle, RequestState, RequestStateError
from minillm_l4.engine.scheduler import ScheduledRequest, StaticBatchScheduler


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


class MixedShapeFakeModel:
    def __init__(self) -> None:
        self.config = tiny_config()
        self.config._name_or_path = "mixed-shape-fake"
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        batch_size, input_tokens = input_ids.shape
        cache = kwargs["past_key_values"]
        for layer_index in range(self.config.num_hidden_layers):
            states = torch.zeros(
                (
                    batch_size,
                    self.config.num_key_value_heads,
                    input_tokens,
                    self.config.head_dim,
                ),
                dtype=torch.float32,
            )
            cache.update(states, states, layer_index)
        next_ids = (input_ids[:, -1] + 1) % self.config.vocab_size
        logits = torch.zeros((batch_size, 1, self.config.vocab_size))
        logits.scatter_(2, next_ids[:, None, None], 1.0)
        return SimpleNamespace(logits=logits, past_key_values=cache)


def test_request_lifecycle_rejects_invalid_transitions() -> None:
    lifecycle = RequestLifecycle("request")
    lifecycle.transition(RequestState.PREFILL)
    lifecycle.transition(RequestState.DECODING)
    lifecycle.transition(RequestState.FINISHED)
    lifecycle.release_resources()

    assert lifecycle.history == ["waiting", "prefill", "decoding", "finished"]
    assert lifecycle.resources_released is True
    with pytest.raises(RequestStateError, match="invalid request transition"):
        lifecycle.transition(RequestState.PREFILL)


def test_static_scheduler_admits_by_arrival_and_batches_fifo() -> None:
    scheduler = StaticBatchScheduler(
        (
            ScheduledRequest("a", 0.0, 0),
            ScheduledRequest("b", 5.0, 1),
            ScheduledRequest("c", 5.0, 2),
        ),
        max_batch_size=2,
    )

    assert [item.request_id for item in scheduler.admit(0.0)] == ["a"]
    assert [item.request_id for item in scheduler.next_batch()] == ["a"]
    assert scheduler.next_arrival_ms == 5.0
    assert [item.request_id for item in scheduler.admit(5.0)] == ["b", "c"]
    assert [item.request_id for item in scheduler.next_batch()] == ["b", "c"]
    assert scheduler.empty
    assert scheduler.maximum_queue_depth == 2


def test_trace_runner_handles_mixed_lengths_cancellation_and_cleanup() -> None:
    requests = (
        RequestSpec("a", (1, 2), 2, category="short"),
        RequestSpec("b", (5, 6, 7), 3, category="long"),
        RequestSpec("cancel", (8,), 2, category="short"),
    )
    workload = WorkloadSpec(
        name="mixed",
        seed=17,
        requests=requests,
        device="cpu",
    )
    runner = StaticRequestTraceRunner(
        MixedShapeFakeModel(),
        max_batch_size=3,
        device="cpu",
        cancel_request_ids=("cancel",),
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
    assert records["a"]["outcome"]["generated_token_ids"] == [3, 4]
    assert records["b"]["outcome"]["generated_token_ids"] == [8, 9, 10]
    assert records["cancel"]["outcome"]["status"] == "cancelled"
    assert runner.last_lifecycles["a"].history == [
        "waiting",
        "prefill",
        "decoding",
        "finished",
    ]
    assert all(
        lifecycle.resources_released
        for lifecycle in runner.last_lifecycles.values()
    )
    assert runner.last_summary is not None
    assert runner.last_summary["prompt_padding_slots"] == 1
    assert runner.last_summary["output_padding_slots"] == 1
    assert runner.last_summary["cancelled_requests"] == 1
    assert runner.last_summary["batches"][0]["cache_released"] is True
    assert result.summary["completed_requests"] == 2
    assert result.summary["cancelled_requests"] == 1


def test_trace_runner_stress_finishes_every_request_without_starvation() -> None:
    requests = tuple(
        RequestSpec(
            request_id=f"request-{index:03d}",
            prompt_token_ids=tuple(range(1, 2 + index % 4)),
            max_new_tokens=1 + index % 5,
            category=f"shape-{index % 4}",
        )
        for index in range(60)
    )
    cancelled = tuple(request.request_id for request in requests[::11])
    workload = WorkloadSpec(
        name="stress",
        seed=17,
        requests=requests,
        device="cpu",
    )
    runner = StaticRequestTraceRunner(
        MixedShapeFakeModel(),
        max_batch_size=8,
        device="cpu",
        cancel_request_ids=cancelled,
    )

    result = BenchmarkHarness(
        HarnessConfig(
            warmup_repetitions=0,
            repetitions=2,
            respect_arrival_schedule=True,
            collect_gpu=False,
            collect_system_telemetry=False,
        )
    ).run_trace(workload, runner)

    assert result.summary["completed_requests"] == 2 * (60 - len(cancelled))
    assert result.summary["cancelled_requests"] == 2 * len(cancelled)
    assert set(runner.last_lifecycles) == {request.request_id for request in requests}
    assert all(lifecycle.terminal for lifecycle in runner.last_lifecycles.values())
    assert all(
        lifecycle.resources_released
        for lifecycle in runner.last_lifecycles.values()
    )
    assert runner.last_summary is not None
    assert runner.last_summary["batch_count"] == 7
