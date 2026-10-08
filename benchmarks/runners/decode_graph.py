"""Static decode metadata for graphs with changing request membership.

Packing is deliberately host-side and CPU-testable. Graphs never consult
request IDs or the allocator; they see only these fixed-address tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


DEFAULT_GRAPH_BATCH_SIZES = (1, 2, 4, 8, 16, 32)


def validate_graph_batch_sizes(sizes: Sequence[int]) -> tuple[int, ...]:
    try:
        sizes = tuple(sizes)
    except TypeError as error:
        raise ValueError("graph_batch_sizes must be a sequence of positive integers") from error
    if (not sizes or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in sizes)
            or tuple(sorted(set(sizes))) != sizes):
        raise ValueError("graph_batch_sizes must be non-empty, strictly increasing positive integers")
    return sizes


def select_graph_bucket(rows: int, sizes: Sequence[int]) -> int | None:
    if rows < 1:
        return None
    return next((size for size in sizes if size >= rows), None)


@dataclass
class DecodeGraphBuffers:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    sequence_lengths: torch.Tensor
    block_tables: torch.Tensor
    attention_mask: torch.Tensor
    scratch_blocks: tuple[int, ...]
    block_size: int

    @classmethod
    def allocate(cls, batch_size: int, table_width: int, scratch_blocks: Sequence[int],
                 *, device: str | torch.device, block_size: int = 16) -> DecodeGraphBuffers:
        scratch = tuple(scratch_blocks[:batch_size])
        if batch_size < 1 or table_width < 1 or block_size < 1:
            raise ValueError("graph batch size, table width, and block size must be positive")
        if len(scratch) != batch_size or len(set(scratch)) != batch_size or any(n < 0 for n in scratch):
            raise ValueError("each graph row requires a distinct non-negative scratch block")
        buffers = cls(
            torch.empty((batch_size, 1), dtype=torch.long, device=device),
            torch.empty((batch_size, 1), dtype=torch.long, device=device),
            torch.empty(batch_size, dtype=torch.int32, device=device),
            torch.empty((batch_size, table_width), dtype=torch.int32, device=device),
            torch.ones((batch_size, 1), dtype=torch.long, device=device),
            scratch,
            block_size,
        )
        buffers.pack((), (), ())  # valid all-scratch batch for warmup/capture
        return buffers

    def pack(self, tokens: Sequence[int], positions: Sequence[int],
             block_rows: Sequence[Sequence[int]]) -> None:
        """Copy real rows, and reset every inactive row on every step.

        Real tokens come from the scheduler's last host readback. Unused table
        entries are -1 and never read. Padding rows write/read only offset zero
        of their own scratch page, so neither real rows nor padding rows alias.
        """
        batch, width = self.block_tables.shape
        real = len(tokens)
        if real > batch or len(positions) != real or len(block_rows) != real:
            raise ValueError("real row counts must agree and fit the graph bucket")
        inputs = torch.zeros((batch, 1), dtype=torch.long)
        starts = torch.zeros((batch, 1), dtype=torch.long)
        lengths = torch.ones(batch, dtype=torch.int32)
        tables = torch.full((batch, width), -1, dtype=torch.int32)
        tables[:, 0] = torch.tensor(self.scratch_blocks, dtype=torch.int32)
        scratch = set(self.scratch_blocks)
        for row, (token, position, blocks) in enumerate(zip(tokens, positions, block_rows, strict=True)):
            live_blocks = (position + 1 + self.block_size - 1) // self.block_size
            if position < 0 or len(blocks) > width or len(blocks) < live_blocks:
                raise ValueError("decode position/block row exceeds the fixed graph table capacity")
            if any(block < 0 or block in scratch for block in blocks):
                raise ValueError("real rows must use valid blocks disjoint from scratch pages")
            inputs[row, 0] = token
            starts[row, 0] = position
            lengths[row] = position + 1
            tables[row, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        self.input_ids.copy_(inputs)
        self.position_ids.copy_(starts)
        self.sequence_lengths.copy_(lengths)
        self.block_tables.copy_(tables)

@dataclass
class DecodeGraph:
    graph: torch.cuda.CUDAGraph
    buffers: DecodeGraphBuffers
    output_tokens: torch.Tensor
    logit_gaps: torch.Tensor | None
    capture_ms: float
    pool_memory_bytes: int


def graph_pool_memory_bytes(graph: torch.cuda.CUDAGraph) -> int:
    """Reserved bytes in this graph's private allocator pool, excluding KV.

    Use the snapshot's segment pool IDs rather than global allocation deltas,
    which would also charge autotuning and unrelated caching allocations.
    """
    pool = tuple(graph.pool())
    return sum(segment["total_size"] for segment in torch.cuda.memory_snapshot()
               if tuple(segment.get("segment_pool_id", ())) == pool)
