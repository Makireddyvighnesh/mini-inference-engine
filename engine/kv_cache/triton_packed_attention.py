"""Triton varlen prefill attention over physical KV pages.

This is intentionally a narrow educational kernel for Qwen-style full
attention.  One Triton program handles one flat query token and one query
head.  It uses the token's request index and local position to walk only that
request's logical pages, so requests with different prompt lengths need no
padding and cannot attend to one another.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on the runtime environment
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _packed_prefill_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        token_to_sequence_ptr,
        token_position_ptr,
        block_table_ptr,
        output_ptr,
        scale,
        query_stride_token: tl.constexpr,
        query_stride_head: tl.constexpr,
        query_stride_dim: tl.constexpr,
        key_stride_block: tl.constexpr,
        key_stride_head: tl.constexpr,
        key_stride_token: tl.constexpr,
        key_stride_dim: tl.constexpr,
        value_stride_block: tl.constexpr,
        value_stride_head: tl.constexpr,
        value_stride_token: tl.constexpr,
        value_stride_dim: tl.constexpr,
        token_to_sequence_stride: tl.constexpr,
        token_position_stride: tl.constexpr,
        table_stride_sequence: tl.constexpr,
        num_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        block_size: tl.constexpr,
        head_dim: tl.constexpr,
        block_dim: tl.constexpr,
        block_tokens: tl.constexpr,
    ):
        program_id = tl.program_id(0)
        token_index = program_id // num_heads
        query_head = program_id % num_heads
        kv_head = query_head // (num_heads // num_kv_heads)

        dim_offsets = tl.arange(0, block_dim)
        dim_mask = dim_offsets < head_dim
        query_offsets = (
            token_index * query_stride_token
            + query_head * query_stride_head
            + dim_offsets * query_stride_dim
        )
        query = tl.load(query_ptr + query_offsets, mask=dim_mask, other=0.0)

        sequence_index = tl.load(
            token_to_sequence_ptr
            + token_index * token_to_sequence_stride,
        )
        query_position = tl.load(
            token_position_ptr + token_index * token_position_stride,
        )
        valid_tokens = query_position + 1

        running_max = -float("inf")
        running_sum = 0.0
        accumulator = tl.zeros((block_dim,), dtype=tl.float32)
        token_offsets = tl.arange(0, block_tokens)
        tile_count = tl.cdiv(valid_tokens, block_tokens)

        for tile_index in tl.range(0, tile_count):
            logical_tokens = tile_index * block_tokens + token_offsets
            token_mask = logical_tokens < valid_tokens
            logical_blocks = logical_tokens // block_size
            offsets_in_block = logical_tokens % block_size
            physical_blocks = tl.load(
                block_table_ptr
                + sequence_index * table_stride_sequence
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

            tile_max = tl.max(scores, axis=0)
            new_max = tl.maximum(running_max, tile_max)
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

        output_offsets = (
            token_index * query_stride_token
            + query_head * query_stride_head
            + dim_offsets * query_stride_dim
        )
        tl.store(
            output_ptr + output_offsets,
            accumulator / running_sum,
            mask=dim_mask,
        )

def triton_is_available() -> bool:
    """Return whether Triton and a CUDA runtime are available."""

    return triton is not None and torch.cuda.is_available()


def can_use_triton_packed_prefill(
    query: torch.Tensor,
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
    token_to_sequence: torch.Tensor,
    token_positions: torch.Tensor,
    block_tables: torch.Tensor,
) -> bool:
    """Check the packed-prefill kernel's narrow device and shape contract."""

    return bool(
        triton_is_available()
        and query.is_cuda
        and key_blocks.is_cuda
        and value_blocks.is_cuda
        and token_to_sequence.is_cuda
        and token_positions.is_cuda
        and block_tables.is_cuda
        and query.ndim == 3
        and query.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and key_blocks.dtype == query.dtype
        and value_blocks.dtype == query.dtype
        and token_to_sequence.ndim == 1
        and token_positions.ndim == 1
        and int(token_to_sequence.shape[0]) == int(query.shape[0])
        and int(token_positions.shape[0]) == int(query.shape[0])
        and block_tables.ndim == 2
    )


def triton_packed_prefill_attention(
    query: torch.Tensor,
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
    token_to_sequence: torch.Tensor,
    token_positions: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Compute causal varlen attention for flat ``[tokens, heads, dim]`` Q."""

    if not can_use_triton_packed_prefill(
        query,
        key_blocks,
        value_blocks,
        token_to_sequence,
        token_positions,
        block_tables,
    ):
        raise ValueError("inputs do not satisfy the Triton packed-prefill contract")
    if key_blocks.shape != value_blocks.shape:
        raise ValueError("key and value page tensors must have identical shapes")
    total_tokens, num_heads, head_dim = (int(value) for value in query.shape)
    num_kv_heads = int(key_blocks.shape[1])
    block_size = int(key_blocks.shape[2])
    if int(block_tables.shape[0]) < 1:
        raise ValueError("block_tables must contain at least one request")
    if num_heads % num_kv_heads != 0:
        raise ValueError("query heads must be divisible by KV heads")
    if block_size < 1 or block_size > 128:
        raise ValueError("Triton packed prefill supports block sizes from 1 to 128")
    if total_tokens < 1:
        raise ValueError("packed prefill requires at least one token")

    output = torch.empty_like(query)
    block_dim = triton.next_power_of_2(head_dim)
    # Keep the online-softmax grouping aligned with the physical page. This
    # matches the readable reference reduction and avoids avoidable greedy
    # argmax drift when a model has nearly tied logits.
    block_tokens = block_size
    grid = (total_tokens * num_heads,)
    _packed_prefill_kernel[grid](
        query,
        key_blocks,
        value_blocks,
        token_to_sequence,
        token_positions,
        block_tables,
        output,
        float(scale),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_blocks.stride(0),
        key_blocks.stride(1),
        key_blocks.stride(2),
        key_blocks.stride(3),
        value_blocks.stride(0),
        value_blocks.stride(1),
        value_blocks.stride(2),
        value_blocks.stride(3),
        token_to_sequence.stride(0),
        token_positions.stride(0),
        block_tables.stride(0),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        head_dim=head_dim,
        block_dim=block_dim,
        block_tokens=block_tokens,
        num_warps=4,
    )
    return output


__all__ = [
    "can_use_triton_packed_prefill",
    "triton_is_available",
    "triton_packed_prefill_attention",
]
