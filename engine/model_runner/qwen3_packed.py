"""Qwen3 packed/ragged prefill execution.

The regular Transformers model API represents a batch as a rectangular
``[batch, sequence]`` tensor.  This module drives the Qwen3 decoder one layer
at a time with a flat ``[total_prompt_tokens, hidden_size]`` tensor instead.
The page-aware attention wrapper consumes the accompanying request metadata
and stores K/V directly in each request's physical pages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from minillm_l4.engine.kv_cache.packed import PackedSequenceMetadata
from minillm_l4.engine.kv_cache.paged import PagedKvCache


@dataclass(frozen=True)
class PackedPrefillOutput:
    """The logits needed to select one first token for every request."""

    logits: torch.Tensor
    hidden_states: torch.Tensor


def qwen3_packed_prefill(
    model: Any,
    input_ids: torch.Tensor,
    metadata: PackedSequenceMetadata,
    *,
    paged_kv_cache: PagedKvCache,
    block_tables: torch.Tensor | None = None,
    attention_backend: str = "auto",
    row_logits: bool = False,
) -> PackedPrefillOutput:
    """Run one packed Qwen3 prefill and write K/V into physical pages.

    Args:
        model: A loaded ``Qwen3ForCausalLM``-compatible model.
        input_ids: Flat token IDs with shape
            ``[sum(metadata.sequence_lengths)]``.
        metadata: Request boundaries and local token positions.
        paged_kv_cache: Cache already allocated for each request's prompt.
        block_tables: Optional ``[batch, max_blocks]`` physical page table.
        attention_backend: ``auto``, ``torch``, ``triton``, ``sdpa``, or
            ``sdpa_math`` for packed prefill attention.
        row_logits: Project each request's final hidden state separately.
            A multi-row ``lm_head`` GEMM rounds differently from the one-row
            projection a single-request prefill uses; per-row projection keeps
            first-token logits bitwise identical to that reference.

    The output contains only the last hidden state of each prompt's final
    token projected to vocabulary logits.  Intermediate hidden states remain
    flat and are never padded.
    """

    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1:
        raise ValueError("input_ids must be a flat tensor with shape [total_tokens]")
    if int(input_ids.shape[0]) != metadata.total_tokens:
        raise ValueError("input_ids length must match packed metadata")
    if input_ids.device != paged_kv_cache.device:
        raise ValueError("input_ids and paged KV cache must use the same device")

    base_model = getattr(model, "model", None)
    layers = getattr(base_model, "layers", None)
    norm = getattr(base_model, "norm", None)
    embed_tokens = getattr(base_model, "embed_tokens", None)
    lm_head = getattr(model, "lm_head", None)
    if layers is None or norm is None or embed_tokens is None or lm_head is None:
        raise TypeError("qwen3_packed_prefill requires a Qwen3ForCausalLM model")

    hidden_states = embed_tokens(input_ids)
    position_ids = metadata.token_positions.to(dtype=torch.long).unsqueeze(0)
    position_embeddings = base_model.rotary_emb(hidden_states, position_ids)
    config = getattr(model, "config", None)
    layer_count = (
        len(layers)
        if config is None
        else min(len(layers), int(getattr(config, "num_hidden_layers", len(layers))))
    )

    for decoder_layer in layers[:layer_count]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=position_embeddings,
            paged_kv_cache=paged_kv_cache,
            paged_sequence_ids=metadata.sequence_ids,
            paged_packed_metadata=metadata,
            paged_block_tables=block_tables,
            paged_attention_backend=attention_backend,
        )

    hidden_states = norm(hidden_states)
    last_indices = metadata.last_token_indices.to(dtype=torch.long)
    last_hidden_states = hidden_states.index_select(0, last_indices)
    if row_logits:
        logits = torch.cat([lm_head(last_hidden_states[row:row + 1]) for row in range(metadata.batch_size)])
    else:
        logits = lm_head(last_hidden_states)
    if logits.ndim != 2 or int(logits.shape[0]) != metadata.batch_size:
        raise RuntimeError("packed Qwen3 prefill returned invalid logits")
    return PackedPrefillOutput(
        logits=logits,
        hidden_states=hidden_states,
    )


__all__ = ["PackedPrefillOutput", "qwen3_packed_prefill"]
