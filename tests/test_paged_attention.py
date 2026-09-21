from __future__ import annotations

import math

import pytest
import torch

from minillm_l4.engine.kv_cache import (
    PagedKvAllocator,
    PagedKvCache,
    paged_attention,
    select_decode_block_tokens,
    select_decode_split_count,
    triton_is_available,
)


def _dense_reference(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    starts: tuple[int, ...],
    groups: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    repeated_keys = keys.repeat_interleave(groups, dim=1)
    repeated_values = values.repeat_interleave(groups, dim=1)
    for row, start in enumerate(starts):
        outputs: list[torch.Tensor] = []
        for query_index in range(query.shape[2]):
            end = start + query_index + 1
            scores = torch.bmm(
                query[row, :, query_index, :].unsqueeze(1),
                repeated_keys[row, :, :end, :].transpose(1, 2),
            ).squeeze(1) * (1.0 / math.sqrt(query.shape[-1]))
            weights = torch.softmax(scores, dim=-1)
            outputs.append(
                torch.matmul(
                    weights.unsqueeze(1),
                    repeated_values[row, :, :end, :],
                ).squeeze(1)
            )
        rows.append(torch.stack(outputs, dim=0))
    return torch.stack(rows, dim=0)


def test_paged_attention_reads_fragmented_pages_without_dense_batch_gather() -> None:
    torch.manual_seed(11)
    allocator = PagedKvAllocator(num_blocks=8, block_size=2)
    cache = PagedKvCache(
        allocator,
        num_layers=1,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )
    allocator.allocate("a", token_count=1)
    allocator.allocate("interleaving-request", token_count=2)
    allocator.append("a", token_count=4)
    allocator.allocate("b", token_count=3)
    assert allocator.get_block_table("a").block_ids == (0, 2, 3)

    keys = torch.randn(2, 2, 5, 4)
    values = torch.randn(2, 2, 5, 4)
    cache.write_layer_segment("a", 0, keys[0:1], values[0:1], start_token=0)
    cache.write_layer_segment("b", 0, keys[1:2, :, :3, :], values[1:2, :, :3, :], start_token=0)

    query = torch.randn(2, 4, 2, 4)
    starts = (3, 1)
    actual = paged_attention(
        query,
        cache,
        ("a", "b"),
        layer_index=0,
        query_start_positions=starts,
        num_key_value_groups=2,
    )
    expected = _dense_reference(
        query,
        keys,
        values,
        starts=starts,
        groups=2,
    )
    assert actual.shape == (2, 2, 4, 4)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_dense_prefill_batch_scatter_matches_each_physical_page() -> None:
    torch.manual_seed(19)
    allocator = PagedKvAllocator(num_blocks=12, block_size=2)
    cache = PagedKvCache(
        allocator,
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )
    allocator.allocate("interleaving", token_count=2)
    allocator.allocate("a", token_count=5)
    allocator.allocate("b", token_count=5)
    allocator.release("interleaving")

    layer_kv = tuple(
        (
            torch.randn(2, 2, 5, 4),
            torch.randn(2, 2, 5, 4),
        )
        for _ in range(2)
    )
    cache.write_dense_batch(
        ("a", "b"),
        layer_kv,
        block_tables=cache.block_table_tensor(
            ("a", "b"),
            device="cpu",
        ).to(dtype=torch.int32),
    )

    for layer_index, (expected_keys, expected_values) in enumerate(layer_kv):
        actual_keys, actual_values = cache.gather_layer("a", layer_index)
        assert torch.equal(actual_keys, expected_keys[0:1])
        assert torch.equal(actual_values, expected_values[0:1])
        actual_keys, actual_values = cache.gather_layer("b", layer_index)
        assert torch.equal(actual_keys, expected_keys[1:2])
        assert torch.equal(actual_values, expected_values[1:2])


@pytest.mark.parametrize(
    ("max_sequence_length", "expected_split_count"),
    [(128, 1), (1024, 2), (2048, 2), (2049, 4), (4096, 4), (4097, 8)],
)
def test_long_context_split_count_uses_fixed_graph_buckets(
    max_sequence_length: int,
    expected_split_count: int,
) -> None:
    assert select_decode_split_count(max_sequence_length) == expected_split_count


@pytest.mark.parametrize(
    ("max_sequence_length", "batch_size", "expected_split_count"),
    [
        (2048, 1, 8),
        (2048, 4, 4),
        (2048, 8, 1),
        (3072, 1, 8),
        (3072, 8, 8),
    ],
)
def test_decode_split_count_uses_profiled_batch_shape(
    max_sequence_length: int,
    batch_size: int,
    expected_split_count: int,
) -> None:
    assert (
        select_decode_split_count(max_sequence_length, batch_size=batch_size)
        == expected_split_count
    )


@pytest.mark.parametrize(
    ("max_sequence_length", "expected_block_tokens"),
    [(1023, 16), (1024, 64), (4608, 64)],
)
def test_decode_tile_selector_uses_profiled_long_context_tile(
    max_sequence_length: int,
    expected_block_tokens: int,
) -> None:
    assert select_decode_block_tokens(max_sequence_length) == expected_block_tokens


@pytest.mark.skipif(not triton_is_available(), reason="requires CUDA and Triton")
@pytest.mark.parametrize("block_size", [8, 16, 32, 64])
@pytest.mark.parametrize("split_count", [1, 2, 4])
@pytest.mark.parametrize("use_gqa_reuse", [False, True])
@pytest.mark.parametrize("block_tokens", [16, 64])
def test_triton_decode_matches_torch_paged_attention(
    block_size: int,
    split_count: int,
    use_gqa_reuse: bool,
    block_tokens: int,
) -> None:
    torch.manual_seed(37)
    device = torch.device("cuda")
    allocator = PagedKvAllocator(num_blocks=32, block_size=block_size)
    cache = PagedKvCache(
        allocator,
        num_layers=1,
        num_kv_heads=2,
        head_dim=128,
        dtype=torch.bfloat16,
        device=device,
    )
    sequence_ids = ("a", "b")
    lengths = (block_size * 2 + 3, block_size + 1)
    for sequence_id, length in zip(sequence_ids, lengths, strict=True):
        allocator.allocate(sequence_id, token_count=length)
        keys = torch.randn(
            1,
            2,
            length,
            128,
            dtype=torch.bfloat16,
            device=device,
        )
        values = torch.randn_like(keys)
        cache.write_layer_segment(
            sequence_id,
            0,
            keys,
            values,
            start_token=0,
        )

    query = torch.randn(
        2,
        4,
        1,
        128,
        dtype=torch.bfloat16,
        device=device,
    )
    positions = tuple(length - 1 for length in lengths)
    expected = paged_attention(
        query,
        cache,
        sequence_ids,
        layer_index=0,
        query_start_positions=positions,
        num_key_value_groups=2,
        backend="torch",
    )
    actual = paged_attention(
        query,
        cache,
        sequence_ids,
        layer_index=0,
        query_start_positions=positions,
        num_key_value_groups=2,
        block_tables=cache.block_table_tensor(sequence_ids, device=device).to(
            dtype=torch.int32
        ),
        sequence_lengths=torch.tensor(lengths, dtype=torch.int32, device=device),
        decode_split_count=split_count,
        decode_max_sequence_length=max(lengths),
        backend="triton",
        decode_use_gqa_reuse=use_gqa_reuse,
        decode_block_tokens=block_tokens,
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
