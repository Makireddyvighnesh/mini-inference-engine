"""Qwen3 adapter for the project-owned direct paged attention path."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from .paged import PagedKvCache
from .paged_attention import paged_attention


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
        if page_cache is None:
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

        if len(sequence_ids) != int(hidden_states.shape[0]):
            raise ValueError("paged_sequence_ids must match the hidden-state batch size")
        if query_start_positions is None:
            starts: Sequence[int] | torch.Tensor | None = None
        else:
            starts = query_start_positions
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

        attn_output = paged_attention(
            query_states,
            page_cache,
            sequence_ids,
            layer_index=self.layer_idx,
            query_start_positions=starts,
            scale=self.scaling,
            num_key_value_groups=self.num_key_value_groups,
            causal=True,
            block_tables=block_tables,
            sequence_lengths=sequence_lengths,
            backend=attention_backend,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.inner.o_proj(attn_output)
        return attn_output, None


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
