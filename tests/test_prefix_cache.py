from __future__ import annotations

from copy import deepcopy
from time import perf_counter_ns

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from minillm_l4.benchmarks.core.harness import RequestEventRecorder
from minillm_l4.benchmarks.core.schemas import RequestSpec
from minillm_l4.benchmarks.runners.prefix_cache import PrefixCachedPagedRunner
from minillm_l4.engine.generation.manual import manual_greedy_generate
from minillm_l4.engine.kv_cache import (
    PagedKvAllocator,
    PagedKvCache,
    PagedKvStateError,
    PagedPrefixCache,
)
from minillm_l4.engine.kv_cache import prefix as prefix_module


def _cache() -> PagedKvCache:
    return PagedKvCache(
        PagedKvAllocator(num_blocks=8, block_size=2),
        num_layers=1,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
    )


def _write(cache: PagedKvCache, owner: str, values: list[float]) -> None:
    tensor = torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1, 2)
    cache.append(owner, ((tensor, tensor + 10),))


def test_prefix_blocks_survive_source_release_and_reject_mutation() -> None:
    cache = _cache()
    cache.allocator.allocate("source")
    _write(cache, "source", [1, 2, 3, 4, 5, 6, 7, 8])
    prefixes = PagedPrefixCache(cache)
    assert prefixes.publish("source", (1, 2, 3, 4)) == 4
    cache.release("source")
    assert cache.allocator.allocated_block_count == 2

    assert prefixes.attach("request", (1, 2, 3, 4, 9), output_tokens=2) == 4
    assert cache.allocator.get_block_table("request").block_ids == (0, 1)
    assert cache.allocator.block_refcount(0) == 2
    with pytest.raises(PagedKvStateError, match="shared prefix"):
        cache.write_layer_segment(
            "request", 0, torch.ones((1, 1, 1, 2)),
            torch.ones((1, 1, 1, 2)), start_token=0,
        )
    cache.append(
        "request",
        ((torch.full((1, 1, 1, 2), 9.0), torch.full((1, 1, 1, 2), 19.0)),),
    )
    assert cache.allocator.get_block_table("request").block_ids == (0, 1, 2)
    cache.release("request")
    assert cache.allocator.allocated_block_count == 2
    prefixes.clear()
    assert cache.allocator.free_block_count == 8


def test_prefix_lookup_uses_exact_tokens_and_evicts_under_capacity() -> None:
    cache = _cache()
    prefixes = PagedPrefixCache(cache, max_entries=1)
    cache.allocator.allocate("a")
    _write(cache, "a", [1, 2, 3, 4, 5, 6, 7, 8])
    prefixes.publish("a", (1, 2, 3, 4))
    cache.release("a")
    assert prefixes.attach("b", (1, 2, 9, 4, 5), output_tokens=1) == 2
    _write(cache, "b", [9, 10, 11, 12, 13, 14])
    prefixes.publish("b", (1, 2, 9, 4, 5))
    cache.release("b")
    assert prefixes.evictions == 1
    assert prefixes.attach("c", (1, 2, 3, 4, 5), output_tokens=1) == 2
    cache.release("c")
    prefixes.clear()
    assert cache.allocator.free_block_count == 8


def test_eviction_keeps_blocks_alive_for_an_active_request() -> None:
    cache = PagedKvCache(
        PagedKvAllocator(num_blocks=3, block_size=2),
        num_layers=1,
        num_kv_heads=1,
        head_dim=2,
    )
    cache.allocator.allocate("source")
    _write(cache, "source", [1, 2, 3, 4, 5, 6, 7, 8])
    prefixes = PagedPrefixCache(cache)
    prefixes.publish("source", (1, 2, 3, 4))
    cache.release("source")
    assert prefixes.attach("active", (1, 2, 3, 4, 5), output_tokens=1) == 4
    before = cache.gather_layer("active", 0)[0].clone()
    prefixes.clear()
    assert cache.allocator.allocated_block_count == 2
    assert torch.equal(cache.gather_layer("active", 0)[0], before)
    _write(cache, "active", [9, 10])
    cache.release("active")
    assert cache.allocator.free_block_count == 3


def test_hash_collision_never_reuses_wrong_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prefix_module, "_block_digest", lambda tokens: b"same-digest")
    cache = _cache()
    prefixes = PagedPrefixCache(cache)
    cache.allocator.allocate("source")
    _write(cache, "source", [1, 2, 3, 4, 5, 6, 7, 8])
    prefixes.publish("source", (1, 2, 3, 4))
    cache.release("source")
    assert prefixes.attach("different", (8, 9, 3, 4, 5), output_tokens=1) == 0
    cache.release("different")
    assert prefixes.attach("partial", (1, 2, 9, 4, 5), output_tokens=1) == 2
    cache.release("partial")
    prefixes.clear()
    assert cache.allocator.free_block_count == cache.allocator.num_blocks


def _tiny_model() -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(
        Qwen3Config(
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
    ).eval()


def test_prefix_cached_generation_matches_uncached_model() -> None:
    torch.manual_seed(47)
    reference = _tiny_model()
    runner = PrefixCachedPagedRunner(
        deepcopy(reference), block_size=2, num_blocks=16, device="cpu"
    )
    prompts = (
        (1, 2, 3, 4, 5, 6),
        (1, 2, 3, 4, 5, 6),
        (1, 2, 3, 4, 7, 8),
    )
    reused = []
    for index, prompt in enumerate(prompts):
        spec = RequestSpec(f"request-{index}", prompt, 3)
        recorder = RequestEventRecorder(spec.request_id, run_started_ns=perf_counter_ns())
        actual = runner((spec,), (recorder,))[0]
        expected = manual_greedy_generate(
            reference,
            {
                "input_ids": torch.tensor([prompt]),
                "attention_mask": torch.ones((1, len(prompt)), dtype=torch.long),
            },
            output_tokens=3,
        )
        assert actual.generated_token_ids == tuple(int(value) for value in expected.row(0))
        reused.append(actual.metadata["prefix_cache"]["cached_prefix_tokens"])
        assert runner.allocator.active_sequence_count == runner.prefixes.entry_count
    assert reused == [0, 4, 4]
    runner.close()
    assert runner.allocator.free_block_count == runner.allocator.num_blocks


def test_attach_counts_hits_only_when_the_request_is_admitted() -> None:
    cache = _cache()
    prefixes = PagedPrefixCache(cache)
    cache.allocator.allocate("source")
    _write(cache, "source", [1, 2, 3, 4, 5, 6, 7, 8])
    prefixes.publish("source", (1, 2, 3, 4))
    cache.release("source")

    # A scheduler may attach, release, and retry a deferred request.
    assert prefixes.attach("request", (1, 2, 3, 4, 9), output_tokens=2) == 4
    cache.release("request")
    reused = prefixes.attach("request", (1, 2, 3, 4, 9), output_tokens=2)
    assert (prefixes.hits, prefixes.misses, prefixes.reused_tokens) == (0, 0, 0)

    prefixes.record_admission(reused)
    prefixes.record_admission(0)
    assert (prefixes.hits, prefixes.misses, prefixes.reused_tokens) == (1, 1, 4)


def test_eviction_skips_entries_pinned_by_active_requests() -> None:
    cache = PagedKvCache(
        PagedKvAllocator(num_blocks=4, block_size=2),
        num_layers=1, num_kv_heads=1, head_dim=2,
    )
    prefixes = PagedPrefixCache(cache)
    cache.allocator.allocate("a")
    _write(cache, "a", [1, 2, 3, 4, 5, 6, 7, 8])
    prefixes.publish("a", (1, 2, 3, 4))
    cache.release("a")
    assert prefixes.attach("active", (1, 2, 3, 4, 9), output_tokens=1) == 4  # pins entry A
    cache.allocator.allocate("b")
    _write(cache, "b", [9, 10, 11, 12, 13, 14, 15, 16])
    prefixes.publish("b", (5, 6, 7, 8))
    cache.release("b")
    assert cache.allocator.free_block_count == 0

    # Entry A is least recently used but pinned; only B's eviction frees space.
    assert prefixes.evict_until_free(1)
    assert prefixes.entry_count == 1 and prefixes.evictions == 1
    assert not prefixes.evict_until_free(3)  # nothing freeable remains
    assert prefixes.entry_count == 1
    cache.release("active")
    assert prefixes.attach("next", (1, 2, 3, 4, 9), output_tokens=1) == 4


def test_allocator_snapshot_counts_shared_physical_blocks_once() -> None:
    allocator = PagedKvAllocator(num_blocks=8, block_size=2)
    allocator.allocate("source", token_count=4)
    allocator.share_prefix("source", "reader", 4)
    allocator.append("reader", 1)

    snapshot = allocator.snapshot()

    assert snapshot["allocated_blocks"] == 3
    assert snapshot["used_token_slots"] == 5
    assert snapshot["reserved_token_slots"] == 6
    assert snapshot["wasted_token_slots"] == 1
    assert snapshot["logical_token_slots"] == 9
