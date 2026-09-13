"""Fused Triton kernel for one-token paged-KV decode attention."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only without Triton installed
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _paged_decode_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        block_table_ptr,
        sequence_length_ptr,
        output_ptr,
        scale,
        query_stride_batch: tl.constexpr,
        query_stride_head: tl.constexpr,
        query_stride_token: tl.constexpr,
        query_stride_dim: tl.constexpr,
        key_stride_block: tl.constexpr,
        key_stride_head: tl.constexpr,
        key_stride_token: tl.constexpr,
        key_stride_dim: tl.constexpr,
        value_stride_block: tl.constexpr,
        value_stride_head: tl.constexpr,
        value_stride_token: tl.constexpr,
        value_stride_dim: tl.constexpr,
        table_stride_batch: tl.constexpr,
        output_stride_batch: tl.constexpr,
        output_stride_head: tl.constexpr,
        output_stride_token: tl.constexpr,
        output_stride_dim: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
    ):
        program_id = tl.program_id(0)
        batch_index = program_id // NUM_HEADS
        query_head = program_id % NUM_HEADS
        kv_head = query_head // (NUM_HEADS // NUM_KV_HEADS)

        dim_offsets = tl.arange(0, BLOCK_DIM)
        dim_mask = dim_offsets < HEAD_DIM
        query_offsets = (
            batch_index * query_stride_batch
            + query_head * query_stride_head
            + dim_offsets * query_stride_dim
        )
        query = tl.load(query_ptr + query_offsets, mask=dim_mask, other=0.0)

        sequence_length = tl.load(sequence_length_ptr + batch_index)
        tile_count = tl.cdiv(sequence_length, BLOCK_TOKENS)
        running_max = -float("inf")
        running_sum = 0.0
        accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
        token_offsets = tl.arange(0, BLOCK_TOKENS)

        for tile_index in tl.range(0, tile_count):
            logical_tokens = tile_index * BLOCK_TOKENS + token_offsets
            token_mask = logical_tokens < sequence_length
            logical_blocks = logical_tokens // BLOCK_SIZE
            offsets_in_block = logical_tokens % BLOCK_SIZE
            physical_blocks = tl.load(
                block_table_ptr
                + batch_index * table_stride_batch
                + logical_blocks,
                mask=token_mask,
                other=0,
            )

            key_offsets = (
                physical_blocks[:, None] * key_stride_block
                + kv_head * key_stride_head
                + offsets_in_block[:, None] * key_stride_token
                + dim_offsets[None, :] * key_stride_dim
            )
            keys = tl.load(
                key_ptr + key_offsets,
                mask=token_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            scores = tl.sum(keys * query[None, :], axis=1) * scale
            scores = tl.where(token_mask, scores, -float("inf"))

            block_max = tl.max(scores, axis=0)
            new_max = tl.maximum(running_max, block_max)
            previous_weight = tl.exp(running_max - new_max)
            probabilities = tl.exp(scores - new_max)
            probabilities = tl.where(token_mask, probabilities, 0.0)

            value_offsets = (
                physical_blocks[:, None] * value_stride_block
                + kv_head * value_stride_head
                + offsets_in_block[:, None] * value_stride_token
                + dim_offsets[None, :] * value_stride_dim
            )
            values = tl.load(
                value_ptr + value_offsets,
                mask=token_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            accumulator = accumulator * previous_weight + tl.sum(
                probabilities[:, None] * values,
                axis=0,
            )
            running_sum = (
                running_sum * previous_weight + tl.sum(probabilities, axis=0)
            )
            running_max = new_max

        output = accumulator / running_sum
        output_offsets = (
            batch_index * output_stride_batch
            + query_head * output_stride_head
            + dim_offsets * output_stride_dim
        )
        tl.store(output_ptr + output_offsets, output, mask=dim_mask)


def triton_is_available() -> bool:
    """Return whether Triton and a CUDA runtime are available."""

    return triton is not None and torch.cuda.is_available()


def can_use_triton_paged_decode(
    query: torch.Tensor,
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
    block_tables: torch.Tensor | None,
    sequence_lengths: torch.Tensor | None,
) -> bool:
    """Check the narrow shape/device contract supported by the fused kernel."""

    return bool(
        triton_is_available()
        and query.is_cuda
        and key_blocks.is_cuda
        and value_blocks.is_cuda
        and query.ndim == 4
        and int(query.shape[2]) == 1
        and query.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and key_blocks.dtype == query.dtype
        and value_blocks.dtype == query.dtype
        and block_tables is not None
        and block_tables.is_cuda
        and block_tables.ndim == 2
        and sequence_lengths is not None
        and sequence_lengths.is_cuda
        and sequence_lengths.ndim == 1
    )


def triton_paged_decode_attention(
    query: torch.Tensor,
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
    block_tables: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Decode one token per request directly from physical KV pages.

    ``query`` is ``[batch, query_heads, 1, head_dim]`` and each cache tensor is
    ``[physical_blocks, kv_heads, block_size, head_dim]``. The result uses the
    Qwen attention-interface layout ``[batch, 1, query_heads, head_dim]``.
    """

    if not can_use_triton_paged_decode(
        query,
        key_blocks,
        value_blocks,
        block_tables,
        sequence_lengths,
    ):
        raise ValueError("inputs do not satisfy the Triton paged-decode contract")
    batch_size, num_heads, _, head_dim = (int(value) for value in query.shape)
    num_kv_heads = int(key_blocks.shape[1])
    block_size = int(key_blocks.shape[2])
    if value_blocks.shape != key_blocks.shape:
        raise ValueError("key and value page tensors must have identical shapes")
    if int(block_tables.shape[0]) != batch_size:
        raise ValueError("block-table batch does not match query batch")
    if int(sequence_lengths.shape[0]) != batch_size:
        raise ValueError("sequence lengths do not match query batch")
    if num_heads % num_kv_heads != 0:
        raise ValueError("query heads must be divisible by KV heads")
    if block_size < 1 or block_size > 128:
        raise ValueError("Triton paged decode supports block sizes from 1 to 128")

    output = torch.empty_like(query)
    block_dim = triton.next_power_of_2(head_dim)
    # Compute tiling is deliberately independent from physical page size.
    # Besides making block sizes 8/16/32/64 follow the same reduction order,
    # this mirrors production kernels where storage pages and compute tiles
    # are separate concepts.
    block_tokens = 16
    grid = (batch_size * num_heads,)
    _paged_decode_kernel[grid](
        query,
        key_blocks,
        value_blocks,
        block_tables,
        sequence_lengths,
        output,
        float(scale),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key_blocks.stride(0),
        key_blocks.stride(1),
        key_blocks.stride(2),
        key_blocks.stride(3),
        value_blocks.stride(0),
        value_blocks.stride(1),
        value_blocks.stride(2),
        value_blocks.stride(3),
        block_tables.stride(0),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        NUM_HEADS=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        BLOCK_DIM=block_dim,
        BLOCK_TOKENS=block_tokens,
        num_warps=4,
    )
    return output.transpose(1, 2)


__all__ = [
    "can_use_triton_paged_decode",
    "triton_is_available",
    "triton_paged_decode_attention",
]
