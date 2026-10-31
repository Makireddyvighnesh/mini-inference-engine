"""Tiled flash prefill over KV pages matches an FP32 reference (CUDA only)."""

from __future__ import annotations

import math

import pytest
import torch

from minillm_l4.engine.kv_cache.triton_packed_kv import triton_is_available

pytestmark = pytest.mark.skipif(not triton_is_available(), reason="needs CUDA and Triton")

HEADS, KV_HEADS, DIM, PAGE = 32, 8, 128, 16


def reference(query, keys, values, q_len, kv_len):
    """FP32 lower-right causal attention for one request: [q_len, heads, dim]."""

    group = HEADS // KV_HEADS
    q = query.float().permute(1, 0, 2)
    k = keys.float().repeat_interleave(group, 0)
    v = values.float().repeat_interleave(group, 0)
    scores = q @ k.transpose(1, 2) / math.sqrt(DIM)
    rows = torch.arange(q_len, device=q.device)[:, None] + (kv_len - q_len)
    scores.masked_fill_(torch.arange(kv_len, device=q.device)[None, :] > rows, float("-inf"))
    return (torch.softmax(scores, -1) @ v).permute(1, 0, 2)


@pytest.mark.parametrize("requests", [
    [(1, 1)],                      # one token
    [(700, 700)],                  # fresh prompt, ragged last tile and page
    [(5, 5), (130, 130), (64, 64), (1000, 1000)],  # packed fresh prompts
    [(37, 300), (256, 1290), (1, 77)],             # later chunks over a cached prefix
])
def test_matches_fp32_reference(requests):
    from minillm_l4.engine.kv_cache.triton_flash_prefill import triton_flash_prefill_attention

    torch.manual_seed(0)
    pages_per = [math.ceil(kv / PAGE) for _, kv in requests]
    total_pages = sum(pages_per) + 3
    # Shuffle physical pages so each request's pages are scattered.
    order = torch.randperm(total_pages, device="cuda").to(torch.int32)
    key_blocks = torch.randn(total_pages, KV_HEADS, PAGE, DIM, device="cuda", dtype=torch.bfloat16)
    value_blocks = torch.randn_like(key_blocks)
    tables = torch.zeros(len(requests), max(pages_per), device="cuda", dtype=torch.int32)
    cursor = 0
    for i, count in enumerate(pages_per):
        tables[i, :count] = order[cursor:cursor + count]
        cursor += count
    q_lens = [q for q, _ in requests]
    query = torch.randn(sum(q_lens), HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    kv_lens = torch.tensor([kv for _, kv in requests], device="cuda", dtype=torch.int32)

    out = triton_flash_prefill_attention(query, key_blocks, value_blocks, tables, cu, kv_lens,
                                         max_query_len=max(q_lens), scale=1 / math.sqrt(DIM))

    for i, (q_len, kv_len) in enumerate(requests):
        pages = tables[i, :pages_per[i]].long()
        keys = key_blocks[pages].permute(1, 0, 2, 3).reshape(KV_HEADS, -1, DIM)[:, :kv_len]
        values = value_blocks[pages].permute(1, 0, 2, 3).reshape(KV_HEADS, -1, DIM)[:, :kv_len]
        start = int(cu[i])
        expected = reference(query[start:start + q_len], keys, values, q_len, kv_len)
        error = (out[start:start + q_len].float() - expected).abs().max().item()
        assert error < 2e-2, (requests[i], error)
