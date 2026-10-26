"""Packed K/V page writes must honor V's own strides (CUDA only)."""

from __future__ import annotations

import pytest
import torch

from minillm_l4.engine.kv_cache.triton_packed_kv import triton_is_available, triton_write_packed_kv

pytestmark = pytest.mark.skipif(not triton_is_available(), reason="needs CUDA and Triton")


def test_strided_value_view_writes_the_same_pages():
    # Two sequences (5 and 11 tokens) over 16-token pages; V is a column view of a
    # merged q/k/v output, as merged projections produce, K is contiguous.
    torch.manual_seed(0)
    tokens, heads, dim, block = 16, 8, 128, 16
    qkv = torch.randn(tokens, (32 + 2 * heads) * dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, heads, tokens, dim, device="cuda", dtype=torch.bfloat16)
    value_view = qkv[:, (32 + heads) * dim:].view(tokens, heads, dim).permute(1, 0, 2).unsqueeze(0)
    assert not value_view.is_contiguous()
    token_to_sequence = torch.tensor([0] * 5 + [1] * 11, device="cuda", dtype=torch.int32)
    positions = torch.tensor(list(range(5)) + list(range(11)), device="cuda", dtype=torch.int32)
    tables = torch.tensor([[2, 0], [1, 3]], device="cuda", dtype=torch.int32)
    pages = {}
    for name, value in (("view", value_view), ("contiguous", value_view.contiguous())):
        key_blocks = torch.zeros(4, heads, block, dim, device="cuda", dtype=torch.bfloat16)
        value_blocks = torch.zeros_like(key_blocks)
        triton_write_packed_kv(key, value, token_to_sequence, positions, tables, key_blocks, value_blocks)
        pages[name] = (key_blocks, value_blocks)
    assert torch.equal(pages["view"][0], pages["contiguous"][0])
    assert torch.equal(pages["view"][1], pages["contiguous"][1])
    assert torch.equal(pages["view"][1][2, :, :5], value_view[0, :, :5])
