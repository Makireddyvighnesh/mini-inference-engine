"""Validate paged decode metadata before kernels can read or write pages."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def validate_decode_metadata(
    block_tables: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    num_blocks: int,
    block_size: int,
    query_positions: torch.Tensor | None = None,
    max_sequence_length: int | None = None,
    reserved_lengths: Sequence[int] | None = None,
) -> None:
    """Check only live table entries; allow arbitrary metadata tensor strides.

    Eager calls raise ordinary Python errors without poisoning the CUDA
    context. During capture, asynchronous device assertions become graph
    nodes and validate the *current* metadata on every replay. Model adapters
    validate once, before layer zero writes KV, rather than once per layer.
    """

    for name, tensor, ndim in (
        ("block_tables", block_tables, 2),
        ("sequence_lengths", sequence_lengths, 1),
    ):
        if tensor.ndim != ndim or tensor.shape[0] != batch_size:
            raise ValueError(f"{name} has an invalid shape for the decode batch")
        if tensor.device != device or tensor.dtype not in {torch.int32, torch.int64}:
            raise ValueError(f"{name} must be an integer tensor on the query device")
    if query_positions is not None and (
        query_positions.ndim != 1 or query_positions.shape[0] != batch_size
        or query_positions.device != device
        or query_positions.dtype not in {torch.int32, torch.int64}
    ):
        raise ValueError("query positions must be an integer vector on the query device")
    if reserved_lengths is not None and len(reserved_lengths) != batch_size:
        raise ValueError("reserved lengths must contain one value per sequence")

    if not torch.cuda.is_current_stream_capturing():
        lengths = sequence_lengths.detach().cpu().tolist()
        positions = None if query_positions is None else query_positions.detach().cpu().tolist()
        tables = block_tables.detach().cpu().tolist()
        for row, length in enumerate(lengths):
            if length < 1 or length > block_tables.shape[1] * block_size:
                raise ValueError("sequence_lengths exceed the available block-table capacity")
            if reserved_lengths is not None and length > reserved_lengths[row]:
                raise ValueError("sequence_lengths exceed the reserved KV length")
            position = length - 1 if positions is None else positions[row]
            if position < 0 or position >= length:
                raise ValueError("query position must satisfy 0 <= position < sequence_length")
            read_length = length if positions is None else position + 1
            if max_sequence_length is not None and read_length > max_sequence_length:
                raise ValueError("max_sequence_length does not cover the attention read length")
            live = tables[row][: (length + block_size - 1) // block_size]
            if any(block < 0 or block >= num_blocks for block in live):
                raise ValueError("block_tables contain an invalid physical block ID")
        return

    # Never copy device values to the host while capturing. These checks are
    # deliberately part of the replay, not just the capture-time validation.
    torch._assert_async(
        ((sequence_lengths > 0) & (sequence_lengths <= block_tables.shape[1] * block_size)).all(),
        "sequence_lengths exceed the available block-table capacity",
    )
    if reserved_lengths is not None:
        for row, limit in enumerate(reserved_lengths):
            torch._assert_async(
                sequence_lengths[row] <= limit, "sequence_lengths exceed reserved KV length",
            )
    if query_positions is not None:
        torch._assert_async(
            ((query_positions >= 0) & (query_positions < sequence_lengths)).all(),
            "query position must satisfy 0 <= position < sequence_length",
        )
    if max_sequence_length is not None:
        read_lengths = sequence_lengths if query_positions is None else query_positions + 1
        torch._assert_async((read_lengths <= max_sequence_length).all(), "max_sequence_length is too small")
    columns = torch.arange(block_tables.shape[1], device=device)
    live = columns[None, :] * block_size < sequence_lengths[:, None]
    torch._assert_async(
        ((~live) | ((block_tables >= 0) & (block_tables < num_blocks))).all(),
        "block_tables contain an invalid physical block ID",
    )
