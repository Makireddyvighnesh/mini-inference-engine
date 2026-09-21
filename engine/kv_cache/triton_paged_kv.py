"""Triton KV writes for one-token paged decode."""

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
    def _write_decode_kv_kernel(
        key_source_ptr,
        value_source_ptr,
        sequence_lengths_ptr,
        block_table_ptr,
        key_destination_ptr,
        value_destination_ptr,
        source_stride_batch: tl.constexpr,
        source_stride_head: tl.constexpr,
        source_stride_token: tl.constexpr,
        source_stride_dim: tl.constexpr,
        table_stride_batch: tl.constexpr,
        key_destination_stride_block: tl.constexpr,
        key_destination_stride_head: tl.constexpr,
        key_destination_stride_token: tl.constexpr,
        key_destination_stride_dim: tl.constexpr,
        value_destination_stride_block: tl.constexpr,
        value_destination_stride_head: tl.constexpr,
        value_destination_stride_token: tl.constexpr,
        value_destination_stride_dim: tl.constexpr,
        source_batch_count: tl.constexpr,
        block_size: tl.constexpr,
        head_dim: tl.constexpr,
        block_dim: tl.constexpr,
    ):
        program_id = tl.program_id(0)
        batch_index = program_id // source_batch_count
        kv_head = program_id % source_batch_count
        dim_offsets = tl.arange(0, block_dim)
        dim_mask = dim_offsets < head_dim

        sequence_length = tl.load(sequence_lengths_ptr + batch_index)
        logical_position = sequence_length - 1
        logical_block = logical_position // block_size
        offset_in_block = logical_position % block_size
        physical_block = tl.load(
            block_table_ptr
            + batch_index * table_stride_batch
            + logical_block,
        )

        source_offsets = (
            batch_index * source_stride_batch
            + kv_head * source_stride_head
            + source_stride_token * 0
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


def can_use_triton_decode_kv_write(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    block_tables: torch.Tensor | None,
    sequence_lengths: torch.Tensor | None,
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
) -> bool:
    """Check the device-side one-token KV-write contract."""

    return bool(
        triton_is_available()
        and key_states.is_cuda
        and value_states.is_cuda
        and key_blocks.is_cuda
        and value_blocks.is_cuda
        and block_tables is not None
        and block_tables.is_cuda
        and sequence_lengths is not None
        and sequence_lengths.is_cuda
        and key_states.ndim == 4
        and value_states.shape == key_states.shape
        and int(key_states.shape[2]) == 1
        and key_states.dtype == value_states.dtype == key_blocks.dtype == value_blocks.dtype
        and block_tables.ndim == 2
        and sequence_lengths.ndim == 1
        and int(key_states.shape[0]) == int(block_tables.shape[0])
        and int(key_states.shape[0]) == int(sequence_lengths.shape[0])
        and int(key_states.shape[1]) == int(key_blocks.shape[1])
        and int(key_states.shape[3]) == int(key_blocks.shape[3])
        and 1 <= int(key_blocks.shape[2]) <= 128
    )


def triton_write_decode_kv(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    block_tables: torch.Tensor,
    sequence_lengths: torch.Tensor,
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
) -> None:
    """Write one newly appended token per request directly into its page."""

    if not can_use_triton_decode_kv_write(
        key_states,
        value_states,
        block_tables,
        sequence_lengths,
        key_blocks,
        value_blocks,
    ):
        raise ValueError("inputs do not satisfy the Triton decode KV-write contract")

    batch_size = int(key_states.shape[0])
    num_kv_heads = int(key_states.shape[1])
    head_dim = int(key_states.shape[3])
    block_size = int(key_blocks.shape[2])
    block_dim = triton.next_power_of_2(head_dim)
    _write_decode_kv_kernel[(batch_size * num_kv_heads,)](
        key_states,
        value_states,
        sequence_lengths,
        block_tables,
        key_blocks,
        value_blocks,
        key_states.stride(0),
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
        num_kv_heads,
        block_size,
        head_dim,
        block_dim,
        num_warps=4,
    )


__all__ = [
    "can_use_triton_decode_kv_write",
    "triton_is_available",
    "triton_write_decode_kv",
]
