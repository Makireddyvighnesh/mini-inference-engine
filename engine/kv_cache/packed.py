"""Metadata for packed (ragged) prompt batches.

The model-facing prefill representation is a single flat token stream.  The
metadata describes where each request starts and ends, and gives a kernel the
request/local-position identity of every flat token.  It is the small piece
that replaces a padded ``[batch, max_prompt]`` tensor for variable-length
prefill.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PackedSequenceMetadata:
    """Ragged-batch indexing metadata for one packed prefill.

    All tensor fields are device-resident so an attention kernel can consume
    them without a CPU lookup in the hot path.

    ``cu_seqlens`` has the same meaning as in common varlen-attention APIs:
    request ``r`` owns flat token indices
    ``[cu_seqlens[r], cu_seqlens[r + 1])``.  A token at local position ``p``
    may attend only to local positions ``0..p`` in that request.
    """

    sequence_ids: tuple[str, ...]
    sequence_lengths: tuple[int, ...]
    cu_seqlens: torch.Tensor
    token_to_sequence: torch.Tensor
    token_positions: torch.Tensor
    last_token_indices: torch.Tensor

    def __post_init__(self) -> None:
        if not self.sequence_ids:
            raise ValueError("sequence_ids must not be empty")
        if len(set(self.sequence_ids)) != len(self.sequence_ids):
            raise ValueError("sequence_ids must be unique")
        if len(self.sequence_lengths) != len(self.sequence_ids):
            raise ValueError("sequence_lengths must match sequence_ids")
        if any(int(length) < 1 for length in self.sequence_lengths):
            raise ValueError("sequence lengths must be positive")

        batch_size = len(self.sequence_ids)
        total_tokens = sum(int(length) for length in self.sequence_lengths)
        expected_cu = batch_size + 1
        if self.cu_seqlens.ndim != 1 or int(self.cu_seqlens.shape[0]) != expected_cu:
            raise ValueError(
                f"cu_seqlens must have shape [{expected_cu}], got "
                f"{tuple(self.cu_seqlens.shape)}"
            )
        if self.token_to_sequence.shape != (total_tokens,):
            raise ValueError("token_to_sequence must contain one entry per flat token")
        if self.token_positions.shape != (total_tokens,):
            raise ValueError("token_positions must contain one entry per flat token")
        if self.last_token_indices.shape != (batch_size,):
            raise ValueError("last_token_indices must contain one entry per request")
        if self.cu_seqlens.dtype not in {torch.int32, torch.int64}:
            raise TypeError("cu_seqlens must use an integer dtype")
        if self.token_to_sequence.dtype not in {torch.int32, torch.int64}:
            raise TypeError("token_to_sequence must use an integer dtype")
        if self.token_positions.dtype not in {torch.int32, torch.int64}:
            raise TypeError("token_positions must use an integer dtype")
        if self.last_token_indices.dtype not in {torch.int32, torch.int64}:
            raise TypeError("last_token_indices must use an integer dtype")

        expected_lengths = torch.tensor(
            self.sequence_lengths,
            dtype=self.cu_seqlens.dtype,
            device=self.cu_seqlens.device,
        )
        actual_lengths = self.cu_seqlens[1:] - self.cu_seqlens[:-1]
        if not torch.equal(actual_lengths, expected_lengths):
            raise ValueError("cu_seqlens does not match sequence_lengths")

    @property
    def batch_size(self) -> int:
        return len(self.sequence_ids)

    @property
    def total_tokens(self) -> int:
        return sum(self.sequence_lengths)

    @classmethod
    def from_lengths(
        cls,
        sequence_ids: Sequence[str],
        sequence_lengths: Sequence[int],
        *,
        device: str | torch.device = "cpu",
    ) -> "PackedSequenceMetadata":
        """Build deterministic flat-token metadata from request lengths."""

        normalized_ids = tuple(str(value) for value in sequence_ids)
        normalized_lengths = tuple(int(value) for value in sequence_lengths)
        if not normalized_ids:
            raise ValueError("sequence_ids must not be empty")
        if len(normalized_ids) != len(normalized_lengths):
            raise ValueError("sequence_ids and sequence_lengths must have equal lengths")
        if any(not value for value in normalized_ids):
            raise ValueError("sequence_ids must not contain empty values")
        if any(value < 1 for value in normalized_lengths):
            raise ValueError("sequence_lengths must be positive")

        normalized_device = torch.device(device)
        lengths_tensor = torch.tensor(
            normalized_lengths,
            dtype=torch.int32,
            device=normalized_device,
        )
        cu_seqlens = torch.zeros(
            (len(normalized_lengths) + 1,),
            dtype=torch.int32,
            device=normalized_device,
        )
        cu_seqlens[1:] = torch.cumsum(lengths_tensor, dim=0)

        request_indices = torch.arange(
            len(normalized_lengths),
            dtype=torch.int32,
            device=normalized_device,
        )
        token_to_sequence = torch.repeat_interleave(request_indices, lengths_tensor)
        token_positions = torch.cat(
            tuple(
                torch.arange(
                    length,
                    dtype=torch.int32,
                    device=normalized_device,
                )
                for length in normalized_lengths
            ),
            dim=0,
        )
        last_token_indices = cu_seqlens[1:] - 1
        return cls(
            sequence_ids=normalized_ids,
            sequence_lengths=normalized_lengths,
            cu_seqlens=cu_seqlens,
            token_to_sequence=token_to_sequence,
            token_positions=token_positions,
            last_token_indices=last_token_indices,
        )

    def request_slice(self, request_index: int) -> slice:
        """Return the flat-token slice belonging to one request."""

        index = int(request_index)
        if index < 0 or index >= self.batch_size:
            raise IndexError("request_index is outside the packed batch")
        start = int(self.cu_seqlens[index].item())
        end = int(self.cu_seqlens[index + 1].item())
        return slice(start, end)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe metadata useful in benchmark manifests."""

        return {
            "sequence_ids": list(self.sequence_ids),
            "sequence_lengths": list(self.sequence_lengths),
            "cu_seqlens": [int(value) for value in self.cu_seqlens.cpu().tolist()],
            "total_tokens": self.total_tokens,
            "padding_tokens_avoided": (
                len(self.sequence_lengths) * max(self.sequence_lengths)
                - self.total_tokens
            ),
        }


__all__ = ["PackedSequenceMetadata"]
