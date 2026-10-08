"""One Qwen3 forward over decode tokens and prompt chunks together.

Each row keeps the numerics of its single-purpose path: projections, norms,
the MLP, and ``lm_head`` are row-independent on the L4 FP8 path, decode rows
use the same paged decode kernel as a decode-only step, and prefill rows use
the same aligned SDPA call as whole-prompt prefill.  The mixed step therefore
produces the tokens that separate decode and prefill forwards would, with one
launch sequence instead of several.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from minillm_l4.engine.kv_cache.mixed import MixedBatchMetadata, MixedPrefillSegment
from minillm_l4.engine.kv_cache.paged import PagedKvCache


@dataclass(frozen=True)
class MixedRow:
    """One scheduled unit: a decode token or a prompt chunk."""

    sequence_id: str
    token_ids: tuple[int, ...]
    start_position: int
    decode: bool
    needs_logits: bool = True


@dataclass(frozen=True)
class MixedForwardOutput:
    """Next-token logits: decode rows in order, then completed prompt chunks."""

    decode_logits: torch.Tensor | None
    prefill_logits: dict[str, torch.Tensor]


def build_mixed_metadata(
    rows: Sequence[MixedRow],
    cache: PagedKvCache,
    *,
    device: torch.device,
    decode_attention_backend: str = "auto",
    decode_split_count: int | None = None,
    decode_max_sequence_length: int | None = None,
    decode_block_tokens: int = 16,
    decode_sdpa_compat: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, MixedBatchMetadata]:
    """Flatten rows (decode first) and describe them; pages must be reserved."""

    ordered = [row for row in rows if row.decode] + [row for row in rows if not row.decode]
    if not ordered:
        raise ValueError("a mixed forward needs at least one row")
    token_ids: list[int] = []
    positions: list[int] = []
    segments: list[MixedPrefillSegment] = []
    decode_ids: list[str] = []
    decode_starts: list[int] = []
    for row in ordered:
        if row.decode and len(row.token_ids) != 1:
            raise ValueError("a decode row carries exactly one token")
        reserved = cache.allocator.get_block_table(row.sequence_id).token_count
        if reserved != row.start_position + len(row.token_ids):
            raise ValueError(f"pages for {row.sequence_id!r} must be reserved through this row")
        flat_start = len(token_ids)
        token_ids.extend(row.token_ids)
        positions.extend(range(row.start_position, row.start_position + len(row.token_ids)))
        if row.decode:
            decode_ids.append(row.sequence_id)
            decode_starts.append(row.start_position)
        else:
            segments.append(MixedPrefillSegment(row.sequence_id, flat_start, len(token_ids), row.start_position))
    decode_count = len(decode_ids)
    prefill_extra = {}
    if segments:
        width = cache.allocator.num_blocks
        tables = torch.zeros((len(segments), width), dtype=torch.int32)
        page_ids = []
        sequence_index, token_positions = [], []
        for index, segment in enumerate(segments):
            blocks = cache.allocator.get_block_table(segment.sequence_id).block_ids
            tables[index, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
            page_ids.append(torch.tensor(blocks, dtype=torch.long, device=device))
            sequence_index += [index] * segment.token_count
            token_positions += list(range(segment.start_position, segment.end_position))
        prefill_extra = dict(
            prefill_token_to_sequence=torch.tensor(sequence_index, dtype=torch.int32, device=device),
            prefill_token_positions=torch.tensor(token_positions, dtype=torch.int32, device=device),
            prefill_block_tables=tables.to(device),
            prefill_page_ids=tuple(page_ids),
        )
    metadata = MixedBatchMetadata(
        total_tokens=len(token_ids),
        decode_rows=torch.arange(decode_count, dtype=torch.long, device=device),
        decode_sequence_ids=tuple(decode_ids),
        decode_start_positions=tuple(decode_starts),
        decode_block_tables=(
            cache.block_table_tensor(tuple(decode_ids), device=device).to(dtype=torch.int32)
            if decode_ids else None
        ),
        decode_sequence_lengths=(
            torch.tensor([start + 1 for start in decode_starts], dtype=torch.int32, device=device)
            if decode_ids else None
        ),
        prefill_segments=tuple(segments),
        **prefill_extra,
        decode_attention_backend=decode_attention_backend,
        decode_split_count=decode_split_count,
        decode_max_sequence_length=decode_max_sequence_length,
        decode_block_tokens=decode_block_tokens,
        decode_sdpa_compat=decode_sdpa_compat,
    )
    return (
        torch.tensor(token_ids, dtype=torch.long, device=device),
        torch.tensor(positions, dtype=torch.long, device=device),
        metadata,
    )


def qwen3_mixed_forward(
    model: Any,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    metadata: MixedBatchMetadata,
    *,
    paged_kv_cache: PagedKvCache,
    prefill_logit_ids: Sequence[str] = (),
) -> MixedForwardOutput:
    """Run every layer once over the flat batch; K/V land in each row's pages.

    ``prefill_logit_ids`` names prompt chunks that finish their prompt; only
    those (and all decode rows) are projected to vocabulary logits.
    """

    base_model = getattr(model, "model", None)
    layers = getattr(base_model, "layers", None)
    if layers is None or getattr(model, "lm_head", None) is None:
        raise TypeError("qwen3_mixed_forward requires a Qwen3ForCausalLM model")
    if input_ids.ndim != 1 or int(input_ids.shape[0]) != metadata.total_tokens:
        raise ValueError("input_ids must be flat and match the mixed metadata")

    hidden_states = base_model.embed_tokens(input_ids)
    position_ids = positions.unsqueeze(0)
    position_embeddings = base_model.rotary_emb(hidden_states, position_ids)
    for decoder_layer in layers[: model.config.num_hidden_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=position_embeddings,
            paged_kv_cache=paged_kv_cache,
            paged_sequence_ids=metadata.decode_sequence_ids,
            paged_mixed_metadata=metadata,
        )
    hidden_states = base_model.norm(hidden_states)

    decode_logits = None
    if metadata.decode_count:
        # One projection for all decode rows, as a decode-only step does.
        decode_logits = model.lm_head(hidden_states.index_select(0, metadata.decode_rows))
    wanted = set(prefill_logit_ids)
    prefill_logits = {
        # One row per completed prompt, as a single-request prefill projects.
        segment.sequence_id: model.lm_head(hidden_states[segment.flat_end - 1 : segment.flat_end])
        for segment in metadata.prefill_segments
        if segment.sequence_id in wanted
    }
    return MixedForwardOutput(decode_logits=decode_logits, prefill_logits=prefill_logits)


__all__ = ["MixedForwardOutput", "MixedRow", "build_mixed_metadata", "qwen3_mixed_forward"]
