"""Paged attention with fused Triton decode and a readable PyTorch fallback.

Keys and values stay in physical pages, and attention consumes each request's
logical block table without materializing a left-padded dense KV batch. CUDA
one-token decode and packed varlen prefill use fused Triton kernels; CPU,
unsupported shapes, and correctness-focused reference paths use the readable
block-wise PyTorch implementation.

The reduction is performed block by block with an online softmax.  Therefore
we never build a ``[batch, heads, max_context]`` score tensor and never
materialize a left-padded dense KV batch.
"""

from __future__ import annotations

import math
import operator
from contextlib import nullcontext
from collections.abc import Sequence
from types import SimpleNamespace

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from .packed import PackedSequenceMetadata
from .paged import PagedKvCache, PagedKvShapeError
from .triton_packed_attention import (
    can_use_triton_packed_prefill,
    triton_packed_prefill_attention,
)
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
        if positions.dtype not in {torch.int32, torch.int64}:
            raise ValueError("query_start_positions must contain integer positions")
        if positions.ndim == 2 and positions.shape[-1] == 1:
            positions = positions[:, 0]
        if positions.ndim != 1:
            raise ValueError("query_start_positions tensor must be one-dimensional")
        normalized = tuple(positions.detach().cpu().tolist())
    else:
        try:
            normalized = tuple(operator.index(value) for value in positions)
        except TypeError as error:
            raise ValueError("query_start_positions must contain integer positions") from error
    if len(normalized) != len(sequence_ids):
        raise ValueError("query_start_positions must contain one value per sequence")
    return normalized


def _triton_query_positions(
    positions: Sequence[int] | torch.Tensor | None,
    *,
    sequence_ids: Sequence[str],
    cache: PagedKvCache,
    query_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Keep device positions live for graph replay; validate host positions."""

    if isinstance(positions, torch.Tensor) and positions.is_cuda:
        if positions.ndim == 2 and positions.shape[-1] == 1:
            positions = positions[:, 0]
        if positions.ndim != 1:
            raise ValueError("query_start_positions tensor must be one-dimensional")
        if positions.shape[0] != len(sequence_ids):
            raise ValueError("query_start_positions must contain one value per sequence")
        return positions

    normalized = _normalize_positions(
        positions, sequence_ids=sequence_ids, cache=cache, query_tokens=query_tokens,
    )
    for sequence_id, start in zip(sequence_ids, normalized, strict=True):
        if start < 0:
            raise ValueError("query_start_position must be non-negative")
        length = cache.allocator.get_block_table(sequence_id).token_count
        if start + query_tokens > length:
            raise PagedKvShapeError(
                f"query range [{start}, {start + query_tokens}) exceeds "
                f"cached token count {length} for sequence {sequence_id!r}"
            )
    return torch.tensor(normalized, dtype=torch.long, device=device)


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
    decode_split_count: int | None = None,
    decode_max_sequence_length: int | None = None,
    decode_use_gqa_reuse: bool | None = None,
    decode_block_tokens: int = 16,
    decode_sdpa_compat: bool = False,
    backend: str = "auto",
    _validate_metadata: bool = True,
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
        decode_split_count: Optional fixed split count for long-context Triton
            decode.  CUDA Graph callers provide this from their known graph
            shape so graph capture never reads a device length on the host.
        decode_max_sequence_length: Maximum logical KV length covered by the
            selected graph shape.  Required when split-KV is enabled.
        decode_use_gqa_reuse: Select the grouped-query-aware Triton kernel.
            ``None`` enables it automatically for supported GQA group sizes;
            ``False`` retains the per-query-head reference kernel.
        decode_block_tokens: Number of logical tokens reduced per Triton tile.

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
        # Storage length may extend beyond this query. The kernels must bound
        # both their page loads and softmax by the query's causal position.
        # Device positions are read during execution, including graph replay.
        causal_positions = (
            _triton_query_positions(
                query_start_positions, sequence_ids=normalized_ids,
                cache=cache, query_tokens=query_tokens, device=query.device,
            )
            if causal else None
        )
        return triton_paged_decode_attention(
            query,
            cache.key_blocks[normalized_layer],
            cache.value_blocks[normalized_layer],
            block_tables,
            sequence_lengths,
            scale=attention_scale,
            split_count=(
                1 if decode_split_count is None else int(decode_split_count)
            ),
            max_sequence_length=decode_max_sequence_length,
            use_gqa_reuse=decode_use_gqa_reuse,
            block_tokens=decode_block_tokens,
            query_start_positions=causal_positions,
            sdpa_compat=decode_sdpa_compat,
            _validate_metadata=_validate_metadata,
            _reserved_lengths=(
                tuple(cache.allocator.get_block_table(sid).token_count for sid in normalized_ids)
                if _validate_metadata else None
            ),
        )

    positions = _normalize_positions(
        query_start_positions, sequence_ids=normalized_ids,
        cache=cache, query_tokens=query_tokens,
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


def packed_paged_attention(
    query: torch.Tensor,
    cache: PagedKvCache,
    metadata: PackedSequenceMetadata,
    *,
    layer_index: int,
    scale: float | None = None,
    num_key_value_groups: int = 1,
    block_tables: torch.Tensor | None = None,
    backend: str = "auto",
    prefill_key_states: torch.Tensor | None = None,
    prefill_value_states: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute causal prefill attention for a flat ragged token batch.

    ``query`` has shape ``[sum(prompt_lengths), query_heads, head_dim]``.  The
    metadata maps every flat row to a request and a local position.  The
    PyTorch fallback loops over requests only at the attention operation; the
    surrounding decoder layer, projections, and MLP still process one flat
    batch.  Supported CUDA inputs use one Triton program per flat token/head.
    """

    if backend not in {
        "auto",
        "torch",
        "triton",
        "sdpa",
        "sdpa_math",
    }:
        raise ValueError(
            "backend must be auto, torch, triton, sdpa, or sdpa_math"
        )
    if not isinstance(query, torch.Tensor) or query.ndim != 3:
        raise ValueError("query must have shape [tokens, heads, head_dim]")
    if int(query.shape[0]) != metadata.total_tokens:
        raise ValueError("query token count must match packed metadata")
    if query.device != cache.device:
        raise ValueError("query and paged cache must be on the same device")
    num_tokens, num_heads, head_dim = (int(value) for value in query.shape)
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
    attention_scale = float(scale) if scale is not None else 1.0 / (head_dim**0.5)

    if block_tables is None:
        block_tables = cache.block_table_tensor(
            metadata.sequence_ids,
            device=query.device,
        ).to(dtype=torch.int32)
    if block_tables.ndim != 2 or int(block_tables.shape[0]) != metadata.batch_size:
        raise ValueError("block_tables must have one row per packed request")

    key_blocks = cache.key_blocks[normalized_layer]
    value_blocks = cache.value_blocks[normalized_layer]

    if (prefill_key_states is None) != (prefill_value_states is None):
        raise ValueError(
            "prefill_key_states and prefill_value_states must be supplied together"
        )
    if backend in {"sdpa", "sdpa_math"}:
        if prefill_key_states is None or prefill_value_states is None:
            raise ValueError(
                "the SDPA packed-prefill backend requires current-layer K/V states"
            )
        return _sdpa_packed_prefill_attention(
            query,
            prefill_key_states,
            prefill_value_states,
            metadata,
            scale=attention_scale,
            num_key_value_groups=groups,
            force_math=backend == "sdpa_math",
        )
    use_triton = can_use_triton_packed_prefill(
        query,
        key_blocks,
        value_blocks,
        metadata.token_to_sequence,
        metadata.token_positions,
        block_tables,
    )
    if backend == "triton" and not use_triton:
        raise RuntimeError("Triton packed prefill was requested but is unsupported")
    if backend != "torch" and use_triton:
        return triton_packed_prefill_attention(
            query.contiguous(),
            key_blocks,
            value_blocks,
            metadata.token_to_sequence,
            metadata.token_positions,
            block_tables,
            scale=attention_scale,
        )

    rows: list[torch.Tensor] = []
    for request_index, sequence_id in enumerate(metadata.sequence_ids):
        token_slice = metadata.request_slice(request_index)
        query_row = query[token_slice].transpose(0, 1).contiguous()
        row_output = _block_attention_for_sequence(
            query_row,
            cache,
            sequence_id,
            layer_index=normalized_layer,
            query_start_position=0,
            scale=attention_scale,
            num_key_value_groups=groups,
            causal=True,
        )
        rows.append(row_output)
    output = torch.cat(rows, dim=0)
    if int(output.shape[0]) != num_tokens:
        raise RuntimeError("packed paged attention returned an invalid token count")
    return output


def _sdpa_packed_prefill_attention(
    query: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    metadata: PackedSequenceMetadata,
    *,
    scale: float,
    num_key_value_groups: int,
    force_math: bool,
) -> torch.Tensor:
    """Run fused SDPA independently for each ragged request.

    This is an optimized intermediate backend, not the final vLLM-style
    ragged kernel: each request gets one fused SDPA call, so there is no
    padding but there are multiple attention launches per decoder layer.  It
    is useful on installations where a FlashAttention/FlashInfer Python
    extension is unavailable, and it provides a fast correctness-preserving
    fallback for the educational packed path.
    """

    if key_states.ndim != 4 or value_states.ndim != 4:
        raise ValueError("prefill K/V states must have shape [1, kv_heads, tokens, dim]")
    if key_states.shape != value_states.shape:
        raise ValueError("prefill key and value states must have identical shapes")
    if int(key_states.shape[0]) != 1:
        raise ValueError("packed SDPA currently expects one flat prefill batch")
    if int(key_states.shape[2]) != metadata.total_tokens:
        raise ValueError("prefill K/V token count must match packed metadata")
    if key_states.device != query.device or value_states.device != query.device:
        raise ValueError("prefill K/V states must be on the query device")

    outputs: list[torch.Tensor] = []
    kernel_context = (
        torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH])
        if force_math
        else nullcontext()
    )
    # Delegate to the exact Transformers SDPA adapter a whole unpadded prompt
    # uses (in Transformers 5.x: no mask, is_causal, enable_gqa), so each
    # request's attention takes the same kernel and rounding as HF prefill.
    module = SimpleNamespace(num_key_value_groups=num_key_value_groups, is_causal=True)
    with kernel_context:
        for request_index in range(metadata.batch_size):
            token_slice = metadata.request_slice(request_index)
            # Query enters as [tokens, query_heads, dim]; SDPA expects
            # [batch, heads, tokens, dim] and returns [batch, tokens, heads, dim].
            query_row = query[token_slice].permute(1, 0, 2).unsqueeze(0)
            row_output, _ = sdpa_attention_forward(
                module,
                query_row,
                key_states[:, :, token_slice, :],
                value_states[:, :, token_slice, :],
                None,
                scaling=scale,
            )
            outputs.append(row_output.squeeze(0))
    return torch.cat(outputs, dim=0)


__all__ = ["paged_attention", "packed_paged_attention"]
