"""Paged attention with fused Triton decode and a readable PyTorch fallback.

Keys and values stay in physical pages, and attention consumes each request's
logical block table without materializing a left-padded dense KV batch. CUDA
one-token decode uses a fused Triton kernel; CPU, unsupported shapes, and
multi-token reference prefill use the block-wise PyTorch implementation.

The reduction is performed block by block with an online softmax.  Therefore
we never build a ``[batch, heads, max_context]`` score tensor and never
materialize a left-padded dense KV batch.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from .paged import PagedKvCache, PagedKvShapeError
from .triton_paged_attention import (
    can_use_triton_paged_decode,
    triton_paged_decode_attention,
)


def _normalize_positions(
    positions: Sequence[int] | torch.Tensor | None,
    *,
    sequence_ids: Sequence[str],
    cache: PagedKvCache,
    query_tokens: int,
) -> tuple[int, ...]:
    if positions is None:
        return tuple(
            cache.allocator.get_block_table(sequence_id).token_count - query_tokens
            for sequence_id in sequence_ids
        )
    if isinstance(positions, torch.Tensor):
        if positions.ndim == 2 and positions.shape[-1] == 1:
            positions = positions[:, 0]
        if positions.ndim != 1:
            raise ValueError("query_start_positions tensor must be one-dimensional")
        normalized = tuple(int(value) for value in positions.detach().cpu().tolist())
    else:
        normalized = tuple(int(value) for value in positions)
    if len(normalized) != len(sequence_ids):
        raise ValueError("query_start_positions must contain one value per sequence")
    return normalized


def _block_attention_for_sequence(
    query: torch.Tensor,
    cache: PagedKvCache,
    sequence_id: str,
    *,
    layer_index: int,
    query_start_position: int,
    scale: float,
    num_key_value_groups: int,
    causal: bool,
) -> torch.Tensor:
    """Compute one request's attention by reading its physical pages."""

    table = cache.allocator.get_block_table(sequence_id)
    num_heads, query_tokens, head_dim = (
        int(query.shape[0]),
        int(query.shape[1]),
        int(query.shape[2]),
    )
    if query_start_position < 0:
        raise ValueError("query_start_position must be non-negative")
    if query_start_position + query_tokens > table.token_count:
        raise PagedKvShapeError(
            f"query range [{query_start_position}, "
            f"{query_start_position + query_tokens}) exceeds cached token count "
            f"{table.token_count} for sequence {sequence_id!r}"
        )

    kv_heads = cache.num_kv_heads
    kv_head_for_query = torch.arange(
        num_heads,
        device=query.device,
        dtype=torch.long,
    ) // num_key_value_groups
    outputs: list[torch.Tensor] = []

    for query_index in range(query_tokens):
        logical_query_position = query_start_position + query_index
        valid_tokens = (
            logical_query_position + 1 if causal else table.token_count
        )
        if valid_tokens < 1 or valid_tokens > table.token_count:
            raise PagedKvShapeError(
                f"invalid attention read length {valid_tokens} for sequence "
                f"{sequence_id!r} with {table.token_count} cached tokens"
            )

        # Accumulate in fp32 for stable softmax behavior, then cast back to the
        # query dtype.  The running max/sum lets us consume one page at a time.
        running_max = torch.full(
            (num_heads,),
            -torch.inf,
            dtype=torch.float32,
            device=query.device,
        )
        running_sum = torch.zeros(
            (num_heads,),
            dtype=torch.float32,
            device=query.device,
        )
        running_output = torch.zeros(
            (num_heads, head_dim),
            dtype=torch.float32,
            device=query.device,
        )
        # Keep the QK dot product in the model activation dtype.  The trusted
        # Qwen path uses BF16 SDPA on the L4; promoting Q/K to FP32 changes CUDA
        # matmul dispatch and can flip a close greedy-logit decision even
        # though the attention values are numerically close.  Softmax state
        # and the weighted-value reduction remain FP32 below.
        query_token = query[:, query_index, :]

        for logical_block_index, physical_block_id in enumerate(table.block_ids):
            block_start = logical_block_index * table.block_size
            if block_start >= valid_tokens:
                break
            block_end = min(block_start + table.block_size, valid_tokens)
            block_tokens = block_end - block_start

            # Physical page layout is [KV heads, block tokens, head dim].
            # Selecting the KV head for each query head implements GQA without
            # materializing repeated K/V tensors for the whole context.
            key_page = cache.key_blocks[
                layer_index,
                physical_block_id,
                :,
                :block_tokens,
                :,
            ]
            value_page = cache.value_blocks[
                layer_index,
                physical_block_id,
                :,
                :block_tokens,
                :,
            ].float()
            key_page = key_page.index_select(0, kv_head_for_query)
            value_page = value_page.index_select(0, kv_head_for_query)

            scores = torch.bmm(
                query_token.unsqueeze(1),
                key_page.transpose(1, 2),
            ).squeeze(1)
            scores = scores.float() * scale
            page_max = scores.max(dim=-1).values
            new_max = torch.maximum(running_max, page_max)
            old_weight = torch.exp(running_max - new_max)
            page_weight = torch.exp(scores - new_max.unsqueeze(-1))
            running_sum = running_sum * old_weight + page_weight.sum(dim=-1)
            running_output = (
                running_output * old_weight.unsqueeze(-1)
                + torch.bmm(page_weight.unsqueeze(1), value_page).squeeze(1)
            )
            running_max = new_max

        outputs.append(running_output / running_sum.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1))

    return torch.stack(outputs, dim=0).to(dtype=query.dtype)


def paged_attention(
    query: torch.Tensor,
    cache: PagedKvCache,
    sequence_ids: Sequence[str],
    *,
    layer_index: int,
    query_start_positions: Sequence[int] | torch.Tensor | None = None,
    scale: float | None = None,
    num_key_value_groups: int = 1,
    causal: bool = True,
    block_tables: torch.Tensor | None = None,
    sequence_lengths: torch.Tensor | None = None,
    backend: str = "auto",
) -> torch.Tensor:
    """Compute attention directly from physical KV blocks.

    Args:
        query: Query states with shape ``[batch, attention_heads, query_tokens,
            head_dim]``.
        cache: The page-backed KV store containing this layer's keys/values.
        sequence_ids: Row-to-block-table mapping for the batch.
        layer_index: Model layer whose physical pages should be read.
        query_start_positions: Logical position of each row's first query
            token.  Prefill uses zero; one-token decode uses the old context
            length.  If omitted, the query is assumed to be the suffix of the
            current cache.
        scale: Query/key scale.  Defaults to ``1 / sqrt(head_dim)``.
        num_key_value_groups: Number of query heads sharing each KV head.
        causal: Apply the causal upper bound for each query token.

    Returns:
        Tensor with shape ``[batch, query_tokens, attention_heads, head_dim]``,
        matching the output layout expected by the Qwen attention module after
        its attention interface returns.
    """

    if backend not in {"auto", "torch", "triton"}:
        raise ValueError("backend must be 'auto', 'torch', or 'triton'")
    if not isinstance(query, torch.Tensor) or query.ndim != 4:
        raise ValueError("query must have shape [batch, heads, tokens, head_dim]")
    normalized_ids = tuple(str(sequence_id) for sequence_id in sequence_ids)
    if not normalized_ids:
        raise ValueError("sequence_ids must not be empty")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("sequence_ids must be unique")
    batch_size, num_heads, query_tokens, head_dim = (
        int(query.shape[0]),
        int(query.shape[1]),
        int(query.shape[2]),
        int(query.shape[3]),
    )
    if batch_size != len(normalized_ids):
        raise ValueError("query batch size must match sequence_ids")
    if head_dim != cache.head_dim:
        raise PagedKvShapeError(
            f"query head_dim {head_dim} does not match cache head_dim {cache.head_dim}"
        )
    groups = int(num_key_value_groups)
    if groups < 1 or num_heads != cache.num_kv_heads * groups:
        raise ValueError(
            "num_key_value_groups must make attention heads divisible across "
            f"{cache.num_kv_heads} KV heads; got heads={num_heads}, groups={groups}"
        )
    normalized_layer = cache._validate_layer_index(layer_index)
    positions = _normalize_positions(
        query_start_positions,
        sequence_ids=normalized_ids,
        cache=cache,
        query_tokens=query_tokens,
    )
    attention_scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)

    use_triton = can_use_triton_paged_decode(
        query,
        cache.key_blocks[normalized_layer],
        cache.value_blocks[normalized_layer],
        block_tables,
        sequence_lengths,
    )
    if backend == "triton" and not use_triton:
        raise RuntimeError("Triton paged decode was requested but is unsupported")
    if backend != "torch" and use_triton:
        assert block_tables is not None
        assert sequence_lengths is not None
        return triton_paged_decode_attention(
            query,
            cache.key_blocks[normalized_layer],
            cache.value_blocks[normalized_layer],
            block_tables,
            sequence_lengths,
            scale=attention_scale,
        )

    rows = [
        _block_attention_for_sequence(
            query[row],
            cache,
            sequence_id,
            layer_index=normalized_layer,
            query_start_position=start,
            scale=attention_scale,
            num_key_value_groups=groups,
            causal=bool(causal),
        )
        for row, (sequence_id, start) in enumerate(
            zip(normalized_ids, positions, strict=True)
        )
    ]
    return torch.stack(rows, dim=0)


__all__ = ["paged_attention"]
