"""Metadata for one mixed prefill + decode forward over a flat token stream.

vLLM-style iteration batching puts every live decode token and every scheduled
prompt chunk into one flat ``[total_tokens]`` batch.  Projections, norms, and
the MLP run once over all rows; only attention depends on the row type:

* decode rows (one new token, history in pages) use the paged decode kernel;
* prefill rows (a prompt chunk at ``start_position``) attend causally to their
  request's cached pages plus the chunk itself.

Example with two decoding requests and two prompt chunks::

    flat tokens:   [d0][d1][p2 p2 p2 ... (128)][p3 p3 ... (512)]
    decode rows:   flat 0 -> r0 at position 70, flat 1 -> r1 at position 40
    prefill:       r2 flat [2, 130) positions [0, 128)
                   r3 flat [130, 642) positions [512, 1024)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MixedPrefillSegment:
    """A contiguous prompt chunk inside the flat token stream."""

    sequence_id: str
    flat_start: int
    flat_end: int
    start_position: int

    @property
    def token_count(self) -> int:
        return self.flat_end - self.flat_start

    @property
    def end_position(self) -> int:
        return self.start_position + self.token_count


@dataclass(frozen=True)
class MixedBatchMetadata:
    """Row layout and decode-kernel settings for one mixed forward."""

    total_tokens: int
    decode_rows: torch.Tensor
    decode_sequence_ids: tuple[str, ...]
    decode_start_positions: tuple[int, ...]
    decode_block_tables: torch.Tensor | None
    decode_sequence_lengths: torch.Tensor | None
    prefill_segments: tuple[MixedPrefillSegment, ...]
    # Prefill-token scatter metadata (one Triton write per layer) and each
    # segment's physical page IDs for gathering its attention prefix.  The
    # table width is fixed to the pool size so kernel strides never change.
    prefill_token_to_sequence: torch.Tensor | None = None
    prefill_token_positions: torch.Tensor | None = None
    prefill_block_tables: torch.Tensor | None = None
    prefill_page_ids: tuple[torch.Tensor, ...] = ()
    decode_attention_backend: str = "auto"
    decode_split_count: int | None = None
    decode_max_sequence_length: int | None = None
    decode_block_tokens: int = 16
    decode_sdpa_compat: bool = False

    def __post_init__(self) -> None:
        if self.total_tokens < 1:
            raise ValueError("a mixed batch needs at least one token")
        decode_count = len(self.decode_sequence_ids)
        if int(self.decode_rows.numel()) != decode_count or len(self.decode_start_positions) != decode_count:
            raise ValueError("decode rows, sequence IDs, and start positions must align")
        owners = list(self.decode_sequence_ids) + [segment.sequence_id for segment in self.prefill_segments]
        if len(set(owners)) != len(owners):
            raise ValueError("a sequence may appear only once in a mixed batch")
        covered = decode_count + sum(segment.token_count for segment in self.prefill_segments)
        if covered != self.total_tokens:
            raise ValueError("decode rows and prefill segments must cover every flat token exactly once")
        if any(segment.token_count < 1 or segment.start_position < 0 for segment in self.prefill_segments):
            raise ValueError("prefill segments must be non-empty with a non-negative start")

    @property
    def decode_count(self) -> int:
        return len(self.decode_sequence_ids)


__all__ = ["MixedBatchMetadata", "MixedPrefillSegment"]
