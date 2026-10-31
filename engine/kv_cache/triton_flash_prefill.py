"""FlashAttention-style varlen prefill attention over physical KV pages.

The packed-prefill kernel in ``triton_packed_attention`` runs one program per
(query token, head), so it re-reads K/V for every token and cannot use tensor
cores. This kernel is the tiled version:

* each program owns a tile of query tokens of one request and one KV head;
* GQA packing: the tile can hold several query heads that share one KV head
  (up to 4 for Qwen3-4B), so every K/V tile loaded from the pages serves
  ``HEADS * BLOCK_M`` query rows; short prompts may pick fewer heads to get
  more programs;
* Q.K^T and P.V run on tensor cores (``tl.dot``) with an online softmax in
  base 2 (the scale is folded into ``log2(e)``); K is loaded in its page
  layout ``[keys, dim]`` and transposed inside the dot;
* only key tiles that cross the causal diagonal are masked, and the grid
  launches the heaviest (latest) query tiles first.

Causality is lower-right aligned: query row ``i`` of a request with ``q_len``
new tokens and ``kv_len`` cached tokens (new ones included) sits at position
``kv_len - q_len + i``, so the same kernel serves fresh prompts and later
chunks over a cached prefix. The rounding differs from PyTorch's
FlashAttention (different tile order), so it is not bitwise equal to the SDPA
path the exactness rules require; it is a speed backend.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on the runtime environment
    triton = None
    tl = None

LOG2E = 1.4426950408889634


# Fixed launch configs per query-length range, picked by sweeping the config
# space on the L4 with every candidate timed right after PyTorch's
# FlashAttention under CUDA graphs (Triton's autotuner picked differently run to
# run, because the power-capped clock drifts while it tries configs).
# (max query length, heads per program, BLOCK_M, BLOCK_N, warps, stages)
CONFIGS = ((8192, 4, 32, 32, 4, 2), (1 << 62, 2, 64, 32, 4, 2))


def _config(max_query_len: int, group: int) -> tuple[int, int, int, int, int]:
    limit, heads, block_m, block_n, warps, stages = next(c for c in CONFIGS if max_query_len <= c[0])
    return math.gcd(heads, group), block_m, block_n, warps, stages  # heads per program must divide the group


if triton is not None:

    @triton.jit
    def _attend_tile(
        q_ptr, k_ptr, v_ptr, out_ptr, table, q_start, q_len, kv_len, m_start, first_head, kv_head, scale_log2,
        stride_qt, stride_qh, stride_kb, stride_kh, stride_kt, stride_vb, stride_vh, stride_vt,
        stride_ot, stride_oh,
        PAGE: tl.constexpr, HEAD_DIM: tl.constexpr, HEADS: tl.constexpr, BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Attention for BLOCK_M query tokens x HEADS query heads of one request."""

        offset = kv_len - q_len  # position of this request's first new token
        ROWS: tl.constexpr = HEADS * BLOCK_M
        rows = tl.arange(0, ROWS)
        token = m_start + rows % BLOCK_M
        head = first_head + rows // BLOCK_M
        dims = tl.arange(0, HEAD_DIM)
        row_ok = token < q_len
        query = tl.load(q_ptr + (q_start + token)[:, None] * stride_qt + head[:, None] * stride_qh + dims[None, :],
                        mask=row_ok[:, None], other=0.0)
        query_pos = offset + token

        row_max = tl.full((ROWS,), float("-inf"), tl.float32)
        row_sum = tl.zeros((ROWS,), tl.float32)
        acc = tl.zeros((ROWS, HEAD_DIM), tl.float32)
        keys = tl.arange(0, BLOCK_N)

        # Key tiles every row of this query tile can fully see: no mask.
        full_end = (offset + m_start + 1) // BLOCK_N * BLOCK_N
        for start in range(0, full_end, BLOCK_N):
            key = start + keys
            page = tl.load(table + key // PAGE)
            slot = key % PAGE
            k = tl.load(k_ptr + page[:, None] * stride_kb + kv_head * stride_kh + slot[:, None] * stride_kt
                        + dims[None, :])
            scores = tl.dot(query, tl.trans(k)) * scale_log2
            new_max = tl.maximum(row_max, tl.max(scores, 1))
            p = tl.math.exp2(scores - new_max[:, None])
            alpha = tl.math.exp2(row_max - new_max)
            row_sum = row_sum * alpha + tl.sum(p, 1)
            v = tl.load(v_ptr + page[:, None] * stride_vb + kv_head * stride_vh + slot[:, None] * stride_vt
                        + dims[None, :])
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            row_max = new_max

        # Tiles crossing the diagonal (and the ragged end): causal + length mask.
        hi = tl.minimum(kv_len, offset + m_start + BLOCK_M)
        for start in range(full_end, hi, BLOCK_N):
            key = start + keys
            key_ok = key < kv_len
            page = tl.load(table + key // PAGE, mask=key_ok, other=0)
            slot = key % PAGE
            k = tl.load(k_ptr + page[:, None] * stride_kb + kv_head * stride_kh + slot[:, None] * stride_kt
                        + dims[None, :], mask=key_ok[:, None], other=0.0)
            scores = tl.dot(query, tl.trans(k)) * scale_log2
            visible = (key[None, :] <= query_pos[:, None]) & key_ok[None, :]
            scores = tl.where(visible, scores, float("-inf"))
            new_max = tl.maximum(row_max, tl.max(scores, 1))
            p = tl.math.exp2(scores - new_max[:, None])
            alpha = tl.math.exp2(row_max - new_max)
            row_sum = row_sum * alpha + tl.sum(p, 1)
            v = tl.load(v_ptr + page[:, None] * stride_vb + kv_head * stride_vh + slot[:, None] * stride_vt
                        + dims[None, :], mask=key_ok[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            row_max = new_max

        out = acc / row_sum[:, None]
        tl.store(out_ptr + (q_start + token)[:, None] * stride_ot + head[:, None] * stride_oh + dims[None, :],
                 out.to(out_ptr.dtype.element_ty), mask=row_ok[:, None])

    @triton.jit
    def _flash_prefill_kernel(
        q_ptr, k_ptr, v_ptr, out_ptr, block_tables_ptr, cu_seqlens_ptr, kv_lens_ptr,
        scale_log2,
        stride_qt, stride_qh,
        stride_kb, stride_kh, stride_kt,
        stride_vb, stride_vh, stride_vt,
        stride_ot, stride_oh,
        stride_bt,
        GROUP: tl.constexpr, PAGE: tl.constexpr, HEAD_DIM: tl.constexpr,
        HEADS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        seq = tl.program_id(1)
        first_head = tl.program_id(2) * HEADS
        kv_head = first_head // GROUP
        q_start = tl.load(cu_seqlens_ptr + seq)
        q_len = tl.load(cu_seqlens_ptr + seq + 1) - q_start
        kv_len = tl.load(kv_lens_ptr + seq)
        table = block_tables_ptr + seq * stride_bt
        tiles = tl.cdiv(q_len, BLOCK_M)
        tile = tl.num_programs(0) - 1 - tl.program_id(0)  # latest (heaviest) tiles first
        if tile < tiles:
            _attend_tile(q_ptr, k_ptr, v_ptr, out_ptr, table, q_start, q_len, kv_len, tile * BLOCK_M,
                         first_head, kv_head, scale_log2, stride_qt, stride_qh, stride_kb, stride_kh,
                         stride_kt, stride_vb, stride_vh, stride_vt, stride_ot, stride_oh,
                         PAGE, HEAD_DIM, HEADS, BLOCK_M, BLOCK_N)


def triton_flash_prefill_attention(
    query: torch.Tensor,
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens: torch.Tensor,
    kv_lens: torch.Tensor,
    *,
    max_query_len: int,
    scale: float,
) -> torch.Tensor:
    """Causal varlen attention for flat ``[tokens, heads, dim]`` Q over KV pages.

    ``key_blocks``/``value_blocks`` are ``[physical_blocks, kv_heads, page, dim]``
    and already hold every request's K/V (new tokens included).
    ``cu_seqlens`` (int32, ``[requests + 1]``) splits the flat query rows per
    request, ``kv_lens`` (int32) is each request's total K/V length, and
    ``max_query_len`` is a host int (sizes the grid, so no device sync).
    """

    if triton is None or not query.is_cuda:
        raise RuntimeError("Triton flash prefill needs CUDA and Triton")
    if query.ndim != 3 or key_blocks.ndim != 4 or key_blocks.shape != value_blocks.shape:
        raise ValueError("expected query [tokens, heads, dim] and pages [blocks, kv_heads, page, dim]")
    tokens, heads, head_dim = (int(x) for x in query.shape)
    kv_heads, page = int(key_blocks.shape[1]), int(key_blocks.shape[2])
    if heads % kv_heads or head_dim & (head_dim - 1) or page & (page - 1):
        raise ValueError("needs heads divisible by KV heads and power-of-two head_dim and page size")
    if query.stride(2) != 1 or key_blocks.stride(3) != 1 or value_blocks.stride(3) != 1:
        raise ValueError("the last (head_dim) dimension must be contiguous")
    requests = int(cu_seqlens.shape[0]) - 1
    out = torch.empty_like(query)
    if tokens == 0 or requests == 0:
        return out
    group = heads // kv_heads
    heads_per_program, block_m, block_n, warps, stages = _config(max_query_len, group)
    grid = (triton.cdiv(max_query_len, block_m), requests, heads // heads_per_program)
    _flash_prefill_kernel[grid](
        query, key_blocks, value_blocks, out, block_tables, cu_seqlens, kv_lens,
        float(scale) * LOG2E,
        query.stride(0), query.stride(1),
        key_blocks.stride(0), key_blocks.stride(1), key_blocks.stride(2),
        value_blocks.stride(0), value_blocks.stride(1), value_blocks.stride(2),
        out.stride(0), out.stride(1),
        block_tables.stride(0),
        GROUP=group, PAGE=page, HEAD_DIM=head_dim,
        HEADS=heads_per_program, BLOCK_M=block_m, BLOCK_N=block_n, num_warps=warps, num_stages=stages,
    )
    return out
