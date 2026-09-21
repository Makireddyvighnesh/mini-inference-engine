"""Triton scatter writes for packed prefill K/V states."""

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
    def _write_packed_kv_kernel(
        key_source_ptr,
        value_source_ptr,
        token_to_sequence_ptr,
        token_position_ptr,
        block_table_ptr,
        key_destination_ptr,
        value_destination_ptr,
        source_stride_head: tl.constexpr,
        source_stride_token: tl.constexpr,
        source_stride_dim: tl.constexpr,
        table_stride_sequence: tl.constexpr,
        key_destination_stride_block: tl.constexpr,
        key_destination_stride_head: tl.constexpr,
        key_destination_stride_token: tl.constexpr,
        key_destination_stride_dim: tl.constexpr,
        value_destination_stride_block: tl.constexpr,
        value_destination_stride_head: tl.constexpr,
        value_destination_stride_token: tl.constexpr,
        value_destination_stride_dim: tl.constexpr,
        token_to_sequence_stride: tl.constexpr,
        token_position_stride: tl.constexpr,
        num_kv_heads: tl.constexpr,
        block_size: tl.constexpr,
        head_dim: tl.constexpr,
        block_dim: tl.constexpr,
    ):
        program_id = tl.program_id(0)
        token_index = program_id // num_kv_heads
        kv_head = program_id % num_kv_heads
        dim_offsets = tl.arange(0, block_dim)
        dim_mask = dim_offsets < head_dim

        sequence_index = tl.load(
            token_to_sequence_ptr
            + token_index * token_to_sequence_stride,
        )
        token_position = tl.load(
            token_position_ptr + token_index * token_position_stride,
        )
        logical_block = token_position // block_size
        offset_in_block = token_position % block_size
        physical_block = tl.load(
            block_table_ptr
            + sequence_index * table_stride_sequence
            + logical_block,
        )

        source_offsets = (
            kv_head * source_stride_head
            + token_index * source_stride_token
            + dim_offsets * source_stride_dim
        )
        key_values = tl.load(
            key_source_ptr + source_offsets,
            mask=dim_mask,
            other=0.0,
        )
        value_values = tl.load(
            value_source_ptr + source_offsets,
            mask=dim_mask,
            other=0.0,
        )

        key_offsets = (
            physical_block * key_destination_stride_block
            + kv_head * key_destination_stride_head
            + offset_in_block * key_destination_stride_token
            + dim_offsets * key_destination_stride_dim
        )
        value_offsets = (
            physical_block * value_destination_stride_block
            + kv_head * value_destination_stride_head
            + offset_in_block * value_destination_stride_token
            + dim_offsets * value_destination_stride_dim
        )
        tl.store(key_destination_ptr + key_offsets, key_values, mask=dim_mask)
        tl.store(value_destination_ptr + value_offsets, value_values, mask=dim_mask)


def triton_is_available() -> bool:
    """Return whether Triton and CUDA are available."""

    return triton is not None and torch.cuda.is_available()


def can_use_triton_packed_kv_write(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    token_to_sequence: torch.Tensor,
    token_positions: torch.Tensor,
    block_tables: torch.Tensor,
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
) -> bool:
    """Check the packed K/V scatter-write shape and device contract."""

    return bool(
        triton_is_available()
        and key_states.is_cuda
        and value_states.is_cuda
        and token_to_sequence.is_cuda
        and token_positions.is_cuda
        and block_tables.is_cuda
        and key_blocks.is_cuda
        and value_blocks.is_cuda
        and key_states.ndim == 4
        and value_states.shape == key_states.shape
        and key_states.shape[0] == 1
        and key_states.dtype == value_states.dtype
        and key_states.dtype == key_blocks.dtype == value_blocks.dtype
        and token_to_sequence.ndim == 1
        and token_positions.ndim == 1
        and token_to_sequence.shape == token_positions.shape
        and block_tables.ndim == 2
        and key_blocks.shape == value_blocks.shape
    )


def triton_write_packed_kv(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    token_to_sequence: torch.Tensor,
    token_positions: torch.Tensor,
    block_tables: torch.Tensor,
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
) -> None:
    """Scatter flat current-layer K/V states into their physical pages."""

    if not can_use_triton_packed_kv_write(
        key_states,
        value_states,
        token_to_sequence,
        token_positions,
        block_tables,
        key_blocks,
        value_blocks,
    ):
        raise ValueError("inputs do not satisfy the Triton packed K/V write contract")
    if int(key_states.shape[2]) != int(token_to_sequence.shape[0]):
        raise ValueError("packed K/V token count must match token metadata")
    if int(key_states.shape[1]) != int(key_blocks.shape[1]):
        raise ValueError("packed K/V head count must match the cache")

    total_tokens = int(key_states.shape[2])
    num_kv_heads = int(key_states.shape[1])
    head_dim = int(key_states.shape[3])
    block_size = int(key_blocks.shape[2])
    block_dim = triton.next_power_of_2(head_dim)
    grid = (total_tokens * num_kv_heads,)
    _write_packed_kv_kernel[grid](
        key_states,
        value_states,
        token_to_sequence,
        token_positions,
        block_tables,
        key_blocks,
        value_blocks,
        key_states.stride(1),
        key_states.stride(2),
        key_states.stride(3),
        block_tables.stride(0),
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
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        head_dim=head_dim,
        block_dim=block_dim,
        num_warps=4,
    )


__all__ = [
    "can_use_triton_packed_kv_write",
    "triton_is_available",
    "triton_write_packed_kv",
]
