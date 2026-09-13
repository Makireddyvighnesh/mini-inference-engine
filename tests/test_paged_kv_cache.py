from __future__ import annotations

import random

import pytest
import torch
from transformers import Qwen3Config

from minillm_l4.engine.kv_cache import (
    PagedKvAllocator,
    PagedKvCache,
    PagedKvOutOfMemoryError,
    PagedKvShapeError,
    PagedKvStateError,
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


def _layer_segment(
    *,
    layer_index: int,
    token_count: int,
    start_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.arange(
        2 * token_count * 4,
        dtype=torch.float32,
    ).reshape(1, 2, token_count, 4)
    keys = values + start_value + layer_index * 1000.0
    return keys, keys + 0.5


def _segments(
    *,
    token_count: int,
    start_value: float,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple(
        _layer_segment(
            layer_index=layer_index,
            token_count=token_count,
            start_value=start_value,
        )
        for layer_index in range(2)
    )


def test_allocator_grows_fragmented_tables_and_accounts_for_internal_waste() -> None:
    allocator = PagedKvAllocator(num_blocks=4, block_size=4)

    first = allocator.allocate("first", token_count=5)
    second = allocator.allocate("second", token_count=2)
    assert first.block_ids == (0, 1)
    assert second.block_ids == (2,)
    assert first.physical_location(4) == (1, 0)
    assert allocator.snapshot()["wasted_token_slots"] == 5

    allocator.release("first")
    replacement = allocator.allocate("replacement", token_count=7)
    assert replacement.block_ids == (0, 1)

    allocator.release("replacement")
    grown = allocator.append("second", token_count=3)
    assert grown.token_count == 5
    assert grown.block_ids == (2, 0)
    assert grown.physical_location(4) == (0, 0)
    assert allocator.snapshot()["allocated_blocks"] == 2


def test_allocator_rejects_oom_without_partial_allocation() -> None:
    allocator = PagedKvAllocator(num_blocks=2, block_size=4)
    allocator.allocate("active", token_count=5)

    assert allocator.can_append("active", 3) is True
    assert allocator.can_allocate(5) is False
    before = allocator.get_block_table("active")
    with pytest.raises(PagedKvOutOfMemoryError, match="free"):
        allocator.allocate("blocked", token_count=1)
    after = allocator.get_block_table("active")
    assert after == before
    assert allocator.free_block_count == 0

    allocator.append("active", token_count=3)
    assert allocator.get_block_table("active").token_count == 8
    with pytest.raises(PagedKvOutOfMemoryError, match="free"):
        allocator.append("active", token_count=1)


def test_allocator_randomized_alloc_append_release_has_no_aliases_or_leaks() -> None:
    allocator = PagedKvAllocator(num_blocks=12, block_size=4)
    active_tokens: dict[str, int] = {}
    randomizer = random.Random(17)

    for step in range(250):
        if active_tokens and randomizer.random() < 0.35:
            sequence_id = randomizer.choice(tuple(active_tokens))
            allocator.release(sequence_id)
            del active_tokens[sequence_id]
        elif active_tokens and randomizer.random() < 0.55:
            sequence_id = randomizer.choice(tuple(active_tokens))
            token_count = randomizer.randint(1, 5)
            if allocator.can_append(sequence_id, token_count):
                allocator.append(sequence_id, token_count)
                active_tokens[sequence_id] += token_count
        else:
            sequence_id = f"request-{step}"
            token_count = randomizer.randint(0, 12)
            if allocator.can_allocate(token_count):
                allocator.allocate(sequence_id, token_count)
                active_tokens[sequence_id] = token_count

        tables = [allocator.get_block_table(sequence_id) for sequence_id in active_tokens]
        all_blocks = [block_id for table in tables for block_id in table.block_ids]
        assert len(all_blocks) == len(set(all_blocks))
        assert len(all_blocks) + allocator.free_block_count == allocator.num_blocks
        assert sum(active_tokens.values()) == allocator.snapshot()["used_token_slots"]

    for sequence_id in tuple(active_tokens):
        allocator.release(sequence_id)
    assert allocator.active_sequence_count == 0
    assert allocator.allocated_block_count == 0
    assert allocator.free_block_count == allocator.num_blocks


def test_paged_cache_gathers_variable_lengths_and_exposes_indirection() -> None:
    allocator = PagedKvAllocator(num_blocks=6, block_size=4)
    cache = PagedKvCache(
        allocator,
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )
    allocator.allocate("long", token_count=0)
    allocator.allocate("short", token_count=0)
    long_segment = _segments(token_count=5, start_value=10.0)
    short_segment = _segments(token_count=2, start_value=100.0)
    cache.append("long", long_segment)
    cache.append("short", short_segment)

    long_keys, long_values = cache.gather_layer("long", 0)
    short_keys, short_values = cache.gather_layer("short", 0)
    assert torch.equal(long_keys, long_segment[0][0])
    assert torch.equal(long_values, long_segment[0][1])
    assert torch.equal(short_keys, short_segment[0][0])
    assert torch.equal(short_values, short_segment[0][1])

    batch = cache.gather_batch(("long", "short"))
    assert len(batch) == 2
    assert batch[0][0].shape == (2, 2, 5, 4)
    assert torch.equal(batch[0][0][0:1], long_segment[0][0])
    assert torch.equal(batch[0][0][1:2, :, -2:, :], short_segment[0][0])
    assert torch.count_nonzero(batch[0][0][1:2, :, :3, :]) == 0
    assert torch.equal(
        cache.attention_mask(("long", "short")),
        torch.tensor([[1, 1, 1, 1, 1], [0, 0, 0, 1, 1]]),
    )
    assert torch.equal(
        cache.block_table_tensor(("long", "short")),
        torch.tensor([[0, 1], [2, -1]]),
    )

    snapshot = allocator.snapshot()
    assert snapshot["layout"] == "paged_fixed_blocks"
    assert snapshot["used_token_slots"] == 7
    assert snapshot["reserved_token_slots"] == 12
    assert snapshot["wasted_token_slots"] == 5
    assert snapshot["internal_fragmentation"] == pytest.approx(5 / 12)


def test_paged_cache_round_trips_into_transformers_dynamic_cache() -> None:
    allocator = PagedKvAllocator(num_blocks=4, block_size=4)
    cache = PagedKvCache(
        allocator,
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )
    allocator.allocate("request-a")
    cache.append("request-a", _segments(token_count=5, start_value=7.0))

    dense_cache = cache.as_dynamic_cache(("request-a",), model_config=tiny_config())
    assert int(dense_cache.get_seq_length()) == 5
    assert len(dense_cache.layers) == 2
    assert dense_cache.layers[0].keys.shape == (1, 2, 5, 4)
    assert torch.equal(dense_cache.layers[1].values, cache.gather_layer("request-a", 1)[1])


def test_paged_cache_imports_new_right_aligned_tokens_from_dense_cache() -> None:
    allocator = PagedKvAllocator(num_blocks=8, block_size=4)
    cache = PagedKvCache(
        allocator,
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )
    allocator.allocate("long")
    allocator.allocate("short")
    cache.append("long", _segments(token_count=5, start_value=10.0))
    cache.append("short", _segments(token_count=2, start_value=100.0))

    dense_cache = cache.as_dynamic_cache(("long", "short"), model_config=tiny_config())
    appended_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_index in range(2):
        one_token = _layer_segment(
            layer_index=layer_index,
            token_count=1,
            start_value=500.0,
        )
        keys = torch.cat((one_token[0], one_token[0] + 50.0), dim=0)
        values = torch.cat((one_token[1], one_token[1] + 50.0), dim=0)
        dense_cache.update(keys, values, layer_index)
        appended_layers.append((keys, values))

    cache.append_from_dynamic_cache(
        ("long", "short"),
        dense_cache,
        previous_token_counts=(5, 2),
        appended_token_counts=(1, 1),
    )
    assert cache.allocator.get_block_table("long").token_count == 6
    assert cache.allocator.get_block_table("short").token_count == 3
    assert torch.equal(cache.gather_layer("long", 0)[0][:, :, -1:, :], appended_layers[0][0][0:1])
    assert torch.equal(cache.gather_layer("short", 0)[0][:, :, -1:, :], appended_layers[0][0][1:2])


def test_paged_cache_rejects_wrong_shapes_and_released_sequences() -> None:
    allocator = PagedKvAllocator(num_blocks=2, block_size=4)
    cache = PagedKvCache(
        allocator,
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        device="cpu",
    )
    allocator.allocate("request")
    invalid = list(_segments(token_count=2, start_value=1.0))
    invalid[0] = (
        torch.zeros((1, 1, 2, 4)),
        invalid[0][1],
    )
    with pytest.raises(PagedKvShapeError, match="does not match"):
        cache.append("request", invalid)

    cache.release("request")
    with pytest.raises(PagedKvStateError, match="no active"):
        cache.gather_layer("request", 0)
