"""Fused kernels must match the unfused PyTorch ops bit for bit (CUDA only)."""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, apply_rotary_pos_emb as hf_rope

from minillm_l4.engine.kernels.fused import (
    apply_rotary_pos_emb, fused_kernels_available, install_fused_kernels, rms_norm, silu_mul,
    uninstall_fused_kernels,
)

pytestmark = pytest.mark.skipif(not fused_kernels_available(), reason="fused kernels need CUDA and Triton")


def _activations(rows, cols, seed):
    torch.manual_seed(seed)
    # Heavy-tailed rows resemble real hidden states (a few large channels).
    x = torch.randn(rows, cols, device="cuda") * torch.rand(rows, 1, device="cuda") * 4
    x[:, :8] *= 40
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("cols", [128, 2560])
@pytest.mark.parametrize("rows", [1, 2, 3, 8, 32, 257, 4096])
def test_rms_norm_bitwise(rows, cols):
    norm = Qwen3RMSNorm(cols, eps=1e-6).cuda().to(torch.bfloat16)
    norm.weight.data = (torch.rand(cols, device="cuda") * 2).to(torch.bfloat16)
    x = _activations(rows, cols, rows * cols)
    assert torch.equal(rms_norm(x, norm.weight, norm.variance_epsilon), norm(x))


@pytest.mark.parametrize("rows", [1, 5, 32, 2048])
def test_add_rms_norm_bitwise(rows):
    norm = Qwen3RMSNorm(2560, eps=1e-6).cuda().to(torch.bfloat16)
    x, residual = _activations(rows, 2560, 1), _activations(rows, 2560, 2)
    normed, summed = rms_norm(x, norm.weight, norm.variance_epsilon, residual=residual)
    assert torch.equal(summed, residual + x)
    assert torch.equal(normed, norm(residual + x))


@pytest.mark.parametrize("shape", [(1, 9728), (7, 9728), (1, 2048, 9728)])
def test_silu_mul_bitwise(shape):
    torch.manual_seed(3)
    gate = (torch.randn(shape, device="cuda") * 6).to(torch.bfloat16)
    up = (torch.randn(shape, device="cuda") * 3).to(torch.bfloat16)
    assert torch.equal(silu_mul(gate, up), torch.nn.functional.silu(gate) * up)


def _rotary(seq, batch):
    config = Qwen3Config(hidden_size=4096, num_attention_heads=32, head_dim=128, rope_theta=5_000_000.0)
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    rotary = Qwen3RotaryEmbedding(config).cuda()
    positions = torch.randint(0, 8192, (batch, seq), device="cuda")
    return rotary(torch.empty(1, device="cuda", dtype=torch.bfloat16), positions)


@pytest.mark.parametrize("layout", ["hf_prefill", "packed", "decode"])
def test_rope_bitwise(layout):
    torch.manual_seed(4)
    if layout == "hf_prefill":    # [B, T, H, D] projection transposed to [B, H, T, D]
        q = torch.randn(2, 300, 32, 128, device="cuda").to(torch.bfloat16).transpose(1, 2)
        k = torch.randn(2, 300, 8, 128, device="cuda").to(torch.bfloat16).transpose(1, 2)
        cos, sin = _rotary(300, 2)
    elif layout == "packed":      # flat tokens permuted to [1, H, T, D]
        q = torch.randn(700, 32, 128, device="cuda").to(torch.bfloat16).permute(1, 0, 2).unsqueeze(0)
        k = torch.randn(700, 8, 128, device="cuda").to(torch.bfloat16).permute(1, 0, 2).unsqueeze(0)
        cos, sin = _rotary(700, 1)
    else:                         # decode rows [B, H, 1, D]
        q = torch.randn(9, 1, 32, 128, device="cuda").to(torch.bfloat16).transpose(1, 2)
        k = torch.randn(9, 1, 8, 128, device="cuda").to(torch.bfloat16).transpose(1, 2)
        cos, sin = _rotary(1, 9)
    ref_q, ref_k = hf_rope(q, k, cos, sin)
    out_q, out_k = apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.equal(out_q, ref_q) and torch.equal(out_k, ref_k)
    # Same memory layout (strides of size-1 dimensions never address memory).
    for out, ref in ((out_q, ref_q), (out_k, ref_k)):
        assert [st for st, n in zip(out.stride(), out.shape) if n > 1] == [st for st, n in zip(ref.stride(), ref.shape) if n > 1]


@pytest.mark.parametrize("head_dim", [64, 128])  # 64 exercises the PyTorch fallback for narrow rows
def test_install_is_reversible_and_bitwise_on_a_model(head_dim):
    torch.manual_seed(5)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=128, hidden_size=512, intermediate_size=1024, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=head_dim,
    )).cuda().to(torch.bfloat16).eval()
    ids = torch.randint(0, 128, (2, 40), device="cuda")
    with torch.inference_mode():
        before = model(input_ids=ids).logits
        assert install_fused_kernels(model)
        fused = model(input_ids=ids).logits
        uninstall_fused_kernels(model)
        after = model(input_ids=ids).logits
    assert torch.equal(fused, before) and torch.equal(after, before)
