"""Qwen3 adapter for the project-owned direct paged attention path."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from .mixed import MixedBatchMetadata
from .packed import PackedSequenceMetadata
from .decode_metadata import validate_decode_metadata
from .paged import PagedKvCache
from .paged_attention import _triton_query_positions, packed_paged_attention, paged_attention
from .triton_packed_kv import (
    can_use_triton_packed_kv_write,
    triton_write_packed_kv,
)
from .triton_paged_kv import (
    can_use_triton_decode_kv_write,
    triton_write_decode_kv,
)


class PagedQwen3Attention(nn.Module):
    """Run Qwen3 projections while storing/reading K/V through page tables.

    The original attention module remains in ``inner`` and owns all learned
    projections and normalization parameters.  With no ``paged_kv_cache``
    keyword this wrapper delegates to the original implementation, preserving
    a safe eager/dense fallback.  With the keyword, it replaces only the
    cache-and-attention portion with the project-owned page-wise routine.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.config = inner.config
        self.layer_idx = inner.layer_idx
        self.head_dim = inner.head_dim
        self.num_key_value_groups = inner.num_key_value_groups
        self.scaling = inner.scaling
        self.sliding_window = getattr(inner, "sliding_window", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        page_cache = kwargs.pop("paged_kv_cache", None)
        packed_metadata = kwargs.pop("paged_packed_metadata", None)
        mixed_metadata = kwargs.pop("paged_mixed_metadata", None)
        if page_cache is None:
            if packed_metadata is not None:
                raise ValueError(
                    "paged_packed_metadata requires paged_kv_cache"
                )
            return self.inner(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
        if not isinstance(page_cache, PagedKvCache):
            raise TypeError("paged_kv_cache must be a PagedKvCache")
        if self.sliding_window is not None:
            raise NotImplementedError(
                "the direct adapter currently supports full-attention Qwen3 layers only"
            )
        if mixed_metadata is not None:
            if not isinstance(mixed_metadata, MixedBatchMetadata):
                raise TypeError("paged_mixed_metadata must be MixedBatchMetadata")
            return self._mixed_forward(hidden_states, position_embeddings, page_cache, mixed_metadata), None

        sequence_ids = tuple(
            str(sequence_id)
            for sequence_id in kwargs.pop("paged_sequence_ids", ())
        )
        if not sequence_ids:
            raise ValueError("paged_sequence_ids must be supplied with paged_kv_cache")
        query_start_positions = kwargs.pop("paged_query_start_positions", None)
        block_tables = kwargs.pop("paged_block_tables", None)
        sequence_lengths = kwargs.pop("paged_sequence_lengths", None)
        attention_backend = str(kwargs.pop("paged_attention_backend", "auto"))
        decode_split_count = kwargs.pop("paged_decode_split_count", None)
        decode_max_sequence_length = kwargs.pop(
            "paged_decode_max_sequence_length", None
        )
        decode_block_tokens = int(kwargs.pop("paged_decode_block_tokens", 16))
        decode_use_gqa_reuse = kwargs.pop("paged_decode_use_gqa_reuse", False)
        decode_sdpa_compat = bool(kwargs.pop("paged_decode_sdpa_compat", False))
        decode_graph = bool(kwargs.pop("paged_decode_graph", False))

        if packed_metadata is not None:
            if not isinstance(packed_metadata, PackedSequenceMetadata):
                raise TypeError(
                    "paged_packed_metadata must be PackedSequenceMetadata"
                )
            if hidden_states.ndim != 2:
                raise ValueError(
                    "packed paged prefill requires hidden_states with shape "
                    "[total_tokens, hidden_size]"
                )
            if sequence_ids != packed_metadata.sequence_ids:
                raise ValueError(
                    "paged_sequence_ids must match packed metadata sequence_ids"
                )
            if int(hidden_states.shape[0]) != packed_metadata.total_tokens:
                raise ValueError(
                    "packed hidden-state rows must match packed metadata token count"
                )

            total_tokens = int(hidden_states.shape[0])
            query_states = self.inner.q_norm(
                self.inner.q_proj(hidden_states).view(
                    total_tokens,
                    -1,
                    self.head_dim,
                )
            )
            key_states = self.inner.k_norm(
                self.inner.k_proj(hidden_states).view(
                    total_tokens,
                    -1,
                    self.head_dim,
                )
            )
            value_states = self.inner.v_proj(hidden_states).view(
                total_tokens,
                -1,
                self.head_dim,
            )

            # Keep a synthetic batch dimension for the shared RoPE helper,
            # then flatten back to [total_tokens, heads, head_dim] for the
            # packed attention kernel.
            query_states = query_states.permute(1, 0, 2).unsqueeze(0)
            key_states = key_states.permute(1, 0, 2).unsqueeze(0)
            value_states = value_states.permute(1, 0, 2).unsqueeze(0)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
            )

            if block_tables is None:
                block_tables = page_cache.block_table_tensor(
                    sequence_ids,
                    device=hidden_states.device,
                ).to(dtype=torch.int32)
            if can_use_triton_packed_kv_write(
                key_states,
                value_states,
                packed_metadata.token_to_sequence,
                packed_metadata.token_positions,
                block_tables,
                page_cache.key_blocks[self.layer_idx],
                page_cache.value_blocks[self.layer_idx],
            ):
                triton_write_packed_kv(
                    key_states,
                    value_states,
                    packed_metadata.token_to_sequence,
                    packed_metadata.token_positions,
                    block_tables,
                    page_cache.key_blocks[self.layer_idx],
                    page_cache.value_blocks[self.layer_idx],
                )
            else:
                for request_index, sequence_id in enumerate(sequence_ids):
                    token_slice = packed_metadata.request_slice(request_index)
                    page_cache.write_layer_segment(
                        sequence_id,
                        self.layer_idx,
                        key_states[:, :, token_slice, :],
                        value_states[:, :, token_slice, :],
                        start_token=0,
                    )

            attn_output = packed_paged_attention(
                query_states.squeeze(0).permute(1, 0, 2).contiguous(),
                page_cache,
                packed_metadata,
                layer_index=self.layer_idx,
                scale=self.scaling,
                num_key_value_groups=self.num_key_value_groups,
                block_tables=block_tables,
                backend=attention_backend,
                prefill_key_states=key_states,
                prefill_value_states=value_states,
            )
            attn_output = attn_output.reshape(total_tokens, -1).contiguous()
            attn_output = self.inner.o_proj(attn_output)
            return attn_output, None

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.inner.q_norm(
            self.inner.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.inner.k_norm(
            self.inner.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.inner.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
        )

        attn_output = self._paged_attend(
            query_states, key_states, value_states, page_cache, sequence_ids,
            query_start_positions=query_start_positions, block_tables=block_tables,
            sequence_lengths=sequence_lengths, attention_backend=attention_backend,
            decode_split_count=decode_split_count,
            decode_max_sequence_length=decode_max_sequence_length,
            decode_block_tokens=decode_block_tokens,
            decode_use_gqa_reuse=decode_use_gqa_reuse,
            decode_sdpa_compat=decode_sdpa_compat,
            decode_graph=decode_graph,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.inner.o_proj(attn_output)
        return attn_output, None



    def _mixed_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        page_cache: PagedKvCache,
        metadata: MixedBatchMetadata,
    ) -> torch.Tensor:
        """Attention for a flat batch of decode tokens and prompt chunks.

        Projections run once over every row.  Decode rows go through the
        decode path's own writer and paged kernel; each prompt chunk writes its
        K/V into pages and attends to its request's prefix with the aligned
        SDPA call used by whole-prompt prefill.
        """

        # Imported lazily: the generation package imports this module.
        from minillm_l4.engine.generation.chunked_prefill import aligned_prefill_sdpa

        if hidden_states.ndim != 2 or int(hidden_states.shape[0]) != metadata.total_tokens:
            raise ValueError("mixed attention expects flat hidden states covering the metadata")
        total = metadata.total_tokens
        query = self.inner.q_norm(self.inner.q_proj(hidden_states).view(total, -1, self.head_dim))
        key = self.inner.k_norm(self.inner.k_proj(hidden_states).view(total, -1, self.head_dim))
        value = self.inner.v_proj(hidden_states).view(total, -1, self.head_dim)
        # [tokens, heads, dim] -> [1, heads, tokens, dim] for the shared RoPE helper.
        query = query.permute(1, 0, 2).unsqueeze(0)
        key = key.permute(1, 0, 2).unsqueeze(0)
        value = value.permute(1, 0, 2).unsqueeze(0)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        output = query.new_empty((total, query.shape[1] * self.head_dim))
        if metadata.decode_count:
            rows = metadata.decode_rows

            def as_decode_batch(states: torch.Tensor) -> torch.Tensor:
                # [1, heads, tokens, dim] -> [decode rows, heads, 1, dim]
                return states[0].index_select(1, rows).permute(1, 0, 2).unsqueeze(2).contiguous()

            decode_output = self._paged_attend(
                as_decode_batch(query), as_decode_batch(key), as_decode_batch(value),
                page_cache, metadata.decode_sequence_ids,
                query_start_positions=metadata.decode_start_positions,
                block_tables=metadata.decode_block_tables,
                sequence_lengths=metadata.decode_sequence_lengths,
                attention_backend=metadata.decode_attention_backend,
                decode_split_count=metadata.decode_split_count,
                decode_max_sequence_length=metadata.decode_max_sequence_length,
                decode_block_tokens=metadata.decode_block_tokens,
                decode_use_gqa_reuse=False,
                decode_sdpa_compat=metadata.decode_sdpa_compat,
            )
            output.index_copy_(0, rows, decode_output.reshape(metadata.decode_count, -1))
        segments = metadata.prefill_segments
        key_pages, value_pages = page_cache.key_blocks[self.layer_idx], page_cache.value_blocks[self.layer_idx]
        if segments:
            prefill = slice(metadata.decode_count, total)
            if metadata.prefill_block_tables is not None and can_use_triton_packed_kv_write(
                key[:, :, prefill, :], value[:, :, prefill, :], metadata.prefill_token_to_sequence,
                metadata.prefill_token_positions, metadata.prefill_block_tables, key_pages, value_pages,
            ):
                triton_write_packed_kv(
                    key[:, :, prefill, :], value[:, :, prefill, :], metadata.prefill_token_to_sequence,
                    metadata.prefill_token_positions, metadata.prefill_block_tables, key_pages, value_pages,
                )
            else:
                for segment in segments:
                    chunk = slice(segment.flat_start, segment.flat_end)
                    page_cache.write_layer_segment(
                        segment.sequence_id, self.layer_idx, key[:, :, chunk, :], value[:, :, chunk, :],
                        start_token=segment.start_position,
                    )
        for index, segment in enumerate(segments):
            chunk = slice(segment.flat_start, segment.flat_end)
            chunk_key, chunk_value = key[:, :, chunk, :], value[:, :, chunk, :]
            if segment.start_position and metadata.prefill_page_ids:
                pages = metadata.prefill_page_ids[index]

                def gather(blocks: torch.Tensor) -> torch.Tensor:
                    # [pages, heads, page_tokens, dim] -> [1, heads, tokens, dim]
                    picked = blocks.index_select(0, pages).permute(1, 0, 2, 3)
                    flat = picked.reshape(picked.shape[0], -1, picked.shape[3])
                    return flat[:, : segment.end_position].unsqueeze(0).contiguous()

                prefix_key, prefix_value = gather(key_pages), gather(value_pages)
            elif segment.start_position:
                prefix_key, prefix_value = page_cache.gather_layer(segment.sequence_id, self.layer_idx)
            else:
                prefix_key, prefix_value = chunk_key, chunk_value
            attended, _ = aligned_prefill_sdpa(
                self, query[:, :, chunk, :], prefix_key, prefix_value, None, scaling=self.scaling,
            )
            output[chunk] = attended.reshape(segment.token_count, -1)
        return self.inner.o_proj(output)

    def _paged_attend(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        page_cache: PagedKvCache,
        sequence_ids: tuple[str, ...],
        *,
        query_start_positions: Any,
        block_tables: torch.Tensor | None,
        sequence_lengths: torch.Tensor | None,
        attention_backend: str,
        decode_split_count: int | None,
        decode_max_sequence_length: int | None,
        decode_block_tokens: int,
        decode_use_gqa_reuse: Any,
        decode_sdpa_compat: bool,
        decode_graph: bool = False,
    ) -> torch.Tensor:
        """Write rectangular-batch K/V into pages and attend from the pages.

        Shared by the decode/rectangular path and the decode rows of a mixed
        forward, so both use exactly the same writer and attention kernel.
        """

        if len(sequence_ids) != int(query_states.shape[0]):
            raise ValueError("paged_sequence_ids must match the hidden-state batch size")
        # Graph rows have fixed synthetic IDs, not allocator owners. Reservation
        # and ownership are checked by the serving runner before packing. Never
        # freeze an owner's current length into a replay-time device assertion.
        if decode_graph:
            if page_cache.allocator.shared_block_count:
                raise ValueError("graph decode does not support shared KV blocks")
            if attention_backend != "triton" or not isinstance(query_start_positions, torch.Tensor):
                raise ValueError("graph decode requires Triton and device query positions")
            if not query_start_positions.is_cuda:
                raise ValueError("graph decode requires CUDA query positions")
        if query_start_positions is None:
            starts: Sequence[int] | torch.Tensor | None = None
        else:
            starts = query_start_positions
        # Every layer writes the same logical positions through the same block
        # tables, so checking shared-block ownership once per forward (layer 0)
        # is sufficient and keeps the per-step host cost off the other layers.
        if self.layer_idx == 0 and page_cache.allocator.shared_block_count:
            for row, sequence_id in enumerate(sequence_ids):
                if isinstance(starts, torch.Tensor):
                    start = int(starts.reshape(-1)[row].item())
                elif starts is None:
                    start = (
                        page_cache.allocator.get_block_table(sequence_id).token_count
                        - int(key_states.shape[-2])
                    )
                else:
                    start = int(starts[row])
                page_cache.allocator.assert_writable_range(
                    sequence_id, start, int(key_states.shape[-2])
                )
        use_fast_writer = can_use_triton_decode_kv_write(
            key_states,
            value_states,
            block_tables,
            sequence_lengths,
            page_cache.key_blocks[self.layer_idx],
            page_cache.value_blocks[self.layer_idx],
        )
        if decode_graph and not use_fast_writer:
            raise ValueError("graph decode requires the Triton one-token KV writer")
        if use_fast_writer:
            write_positions = _triton_query_positions(
                starts, sequence_ids=sequence_ids, cache=page_cache,
                query_tokens=1, device=query_states.device,
            )
            # All layers consume the same metadata. Validate once before any
            # KV write; graph capture records live device assertions here.
            if self.layer_idx == 0:
                validate_decode_metadata(
                    block_tables, sequence_lengths, device=query_states.device,
                    batch_size=len(sequence_ids), num_blocks=page_cache.allocator.num_blocks,
                    block_size=page_cache.allocator.block_size, query_positions=write_positions,
                    max_sequence_length=(
                        decode_max_sequence_length if decode_graph or (decode_split_count and decode_split_count > 1) else None
                    ),
                    reserved_lengths=None if decode_graph else tuple(
                        page_cache.allocator.get_block_table(sid).token_count for sid in sequence_ids
                    ),
                )
            triton_write_decode_kv(
                key_states,
                value_states,
                block_tables,
                sequence_lengths,
                page_cache.key_blocks[self.layer_idx],
                page_cache.value_blocks[self.layer_idx],
                query_start_positions=write_positions,
                _validate_metadata=False,
            )
        else:
            for row, sequence_id in enumerate(sequence_ids):
                if isinstance(starts, torch.Tensor):
                    if starts.ndim == 2 and starts.shape[-1] == 1:
                        start = int(starts[row, 0].item())
                    else:
                        start = int(starts[row].item())
                elif starts is None:
                    start = None
                else:
                    start = int(starts[row])
                # ``None`` makes paged_attention infer the suffix position, but a
                # layer write needs an explicit logical location.  In that case a
                # query suffix is necessarily the final query_tokens positions.
                if start is None:
                    start = (
                        page_cache.allocator.get_block_table(sequence_id).token_count
                        - int(key_states.shape[-2])
                    )
                page_cache.write_layer_segment(
                    sequence_id,
                    self.layer_idx,
                    key_states[row : row + 1],
                    value_states[row : row + 1],
                    start_token=start,
                )

        return paged_attention(
            query_states,
            page_cache,
            sequence_ids,
            layer_index=self.layer_idx,
            query_start_positions=write_positions if use_fast_writer else starts,
            scale=self.scaling,
            num_key_value_groups=self.num_key_value_groups,
            causal=True,
            block_tables=block_tables,
            sequence_lengths=sequence_lengths,
            decode_split_count=decode_split_count,
            decode_max_sequence_length=decode_max_sequence_length,
            decode_use_gqa_reuse=decode_use_gqa_reuse,
            decode_block_tokens=decode_block_tokens,
            decode_sdpa_compat=decode_sdpa_compat,
            backend=attention_backend,
            _validate_metadata=not use_fast_writer,
        )


def install_paged_qwen3_attention(model: Any) -> tuple[PagedQwen3Attention, ...]:
    """Replace Qwen3 self-attention modules with page-aware wrappers.

    Installation is idempotent, which lets a benchmark reuse one loaded model
    across warmups and repetitions.  The original module is retained inside
    every wrapper and remains the fallback when no page cache is supplied.
    """

    base_model = getattr(model, "model", None)
    layers = getattr(base_model, "layers", None)
    if layers is None:
        raise TypeError("paged Qwen3 integration requires model.model.layers")
    wrappers: list[PagedQwen3Attention] = []
    for layer_index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        if isinstance(attention, PagedQwen3Attention):
            wrapper = attention
        else:
            if attention is None:
                raise TypeError(f"model layer {layer_index} has no self_attn module")
            wrapper = PagedQwen3Attention(attention)
            layer.self_attn = wrapper
        wrappers.append(wrapper)
    return tuple(wrappers)


__all__ = ["PagedQwen3Attention", "install_paged_qwen3_attention"]
