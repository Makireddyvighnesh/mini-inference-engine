from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config

from minillm_l4.benchmarks.core.harness import RequestEventRecorder
from minillm_l4.benchmarks.core.schemas import RequestSpec
from minillm_l4.benchmarks.runners.kv_cache import KvCacheBatchRunner
from minillm_l4.engine.generation.manual import manual_greedy_generate
from minillm_l4.engine.generation.recompute import recompute_greedy_generate
from minillm_l4.engine.kv_cache import (
    ContiguousKvCache,
    KvCacheCapacityError,
    KvCacheStateError,
)


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


class CacheAwareFakeModel:
    """Update a real Transformers cache while producing deterministic logits."""

    def __init__(self) -> None:
        self.config = tiny_config()
        self.config._name_or_path = "fake-cache-model"
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        batch_size, input_tokens = input_ids.shape
        cache = kwargs.get("past_key_values")
        if kwargs["use_cache"]:
            assert cache is not None
            for layer_index in range(self.config.num_hidden_layers):
                states = torch.full(
                    (
                        batch_size,
                        self.config.num_key_value_heads,
                        input_tokens,
                        self.config.head_dim,
                    ),
                    float(layer_index + 1),
                    dtype=torch.float32,
                    device=input_ids.device,
                )
                cache.update(states, states + 0.5, layer_index)
        else:
            assert cache is None

        next_ids = (input_ids[:, -1] + 1) % self.config.vocab_size
        logits = torch.zeros(
            (batch_size, 1, self.config.vocab_size),
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits.scatter_(2, next_ids[:, None, None], 1.0)
        return SimpleNamespace(logits=logits, past_key_values=cache)

    def generate(self, **kwargs):
        del kwargs
        raise AssertionError("KV paths must not call model.generate")


def sample_inputs() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[3, 4, 5], [6, 7, 8]], dtype=torch.long),
        "attention_mask": torch.ones((2, 3), dtype=torch.long),
    }


def test_contiguous_cache_tracks_shapes_capacity_bytes_and_reset() -> None:
    cache = ContiguousKvCache(
        tiny_config(),
        owner_ids=("request-a", "request-b"),
        capacity_tokens=8,
    )
    backend = cache.backend_cache
    for layer_index in range(2):
        states = torch.zeros((2, 2, 3, 4), dtype=torch.float32)
        backend.update(states, states, layer_index)
    cache.commit_append(3, backend)

    snapshot = cache.snapshot(free_device_bytes_before=4096)

    assert snapshot["backend"] == "DynamicCache"
    assert snapshot["position_tokens_per_sequence"] == 3
    assert snapshot["capacity_tokens_per_sequence"] == 8
    assert snapshot["aggregate_used_token_slots"] == 6
    assert snapshot["aggregate_capacity_token_slots"] == 16
    assert snapshot["utilization"] == 3 / 8
    assert snapshot["layer_count"] == 2
    assert snapshot["allocated_bytes"] == 768
    assert snapshot["bytes_per_sequence_token"] == 128
    assert snapshot["layers"][0]["key_shape"] == [2, 2, 3, 4]
    assert snapshot["layers"][0]["value_shape"] == [2, 2, 3, 4]
    assert snapshot["estimated_additional_token_slots_from_free_memory"] == 32

    cache.reset()
    assert cache.position == 0
    assert cache.backend_cache is not backend
    assert int(cache.backend_cache.get_seq_length()) == 0
    assert cache.snapshot()["allocated_bytes"] == 0


def test_cache_enforces_ownership_capacity_release_and_isolation() -> None:
    first = ContiguousKvCache(
        tiny_config(),
        owner_ids=("first",),
        capacity_tokens=4,
    )
    second = ContiguousKvCache(
        tiny_config(),
        owner_ids=("second",),
        capacity_tokens=4,
    )
    assert first.backend_cache is not second.backend_cache
    with pytest.raises(KvCacheStateError, match="owner mismatch"):
        first.assert_owners(("second",))
    with pytest.raises(KvCacheCapacityError, match="capacity"):
        first.prepare_append(5)
    assert second.position == 0

    first.release()
    assert first.lifecycle == "released"
    with pytest.raises(KvCacheStateError, match="released"):
        _ = first.backend_cache
    first.release()


def test_explicit_cache_and_recompute_generate_identical_tokens() -> None:
    cached_model = CacheAwareFakeModel()
    cache = ContiguousKvCache(
        cached_model.config,
        owner_ids=("a", "b"),
        capacity_tokens=6,
    )
    cached = manual_greedy_generate(
        cached_model,
        sample_inputs(),
        output_tokens=4,
        kv_cache=cache,
    )

    recompute_model = CacheAwareFakeModel()
    recomputed = recompute_greedy_generate(
        recompute_model,
        sample_inputs(),
        output_tokens=4,
    )

    assert cached.runtime == "manual_contiguous_cache"
    assert recomputed.runtime == "recompute_eager"
    assert cached.token_ids.tolist() == [[6, 7, 8, 9], [9, 10, 11, 12]]
    assert cached.token_ids.tolist() == recomputed.token_ids.tolist()
    assert cache.position == 6
    assert [call["input_ids"].shape[1] for call in cached_model.calls] == [3, 1, 1, 1]
    assert [call["input_ids"].shape[1] for call in recompute_model.calls] == [3, 4, 5, 6]
    assert all(call["use_cache"] is True for call in cached_model.calls)
    assert all(call["use_cache"] is False for call in recompute_model.calls)
    assert all(
        call["past_key_values"] is cache.backend_cache
        for call in cached_model.calls
    )


def test_kv_runner_reports_cache_accounting_and_recompute_work() -> None:
    requests = tuple(
        RequestSpec(
            request_id=f"request-{index}",
            prompt_token_ids=(1, 2, 3),
            max_new_tokens=3,
        )
        for index in range(2)
    )

    cached_runner = KvCacheBatchRunner(
        CacheAwareFakeModel(),
        mode="contiguous",
        device="cpu",
        capacity_tokens=8,
    )
    cached_recorders = [
        RequestEventRecorder(request.request_id, run_started_ns=0)
        for request in requests
    ]
    cached_outcomes = cached_runner(requests, cached_recorders)

    recompute_runner = KvCacheBatchRunner(
        CacheAwareFakeModel(),
        mode="recompute",
        device="cpu",
    )
    recompute_recorders = [
        RequestEventRecorder(request.request_id, run_started_ns=0)
        for request in requests
    ]
    recompute_outcomes = recompute_runner(requests, recompute_recorders)

    assert [outcome.generated_token_ids for outcome in cached_outcomes] == [
        outcome.generated_token_ids for outcome in recompute_outcomes
    ]
    assert cached_runner.last_cache_snapshot is not None
    assert cached_runner.last_cache_snapshot["owner_ids"] == [
        "request-0",
        "request-1",
    ]
    assert cached_runner.last_cache_snapshot["position_tokens_per_sequence"] == 5
    assert cached_outcomes[0].metadata["cache_mode"] == "contiguous"
    assert cached_outcomes[0].metadata[
        "model_input_tokens_processed_per_sequence"
    ] == 5
    assert recompute_outcomes[0].metadata["cache_mode"] == "recompute"
    assert recompute_outcomes[0].metadata[
        "model_input_tokens_processed_per_sequence"
    ] == 12
    assert all(
        [event.event for event in recorder.events].count("completion") == 1
        for recorder in (*cached_recorders, *recompute_recorders)
    )


def test_kv_runner_rejects_capacity_before_decode_overflow() -> None:
    request = RequestSpec(
        request_id="request",
        prompt_token_ids=(1, 2, 3),
        max_new_tokens=3,
    )
    runner = KvCacheBatchRunner(
        CacheAwareFakeModel(),
        mode="contiguous",
        device="cpu",
        capacity_tokens=3,
    )
    recorder = RequestEventRecorder(request.request_id, run_started_ns=0)

    with pytest.raises(KvCacheCapacityError, match="capacity"):
        runner((request,), (recorder,))
