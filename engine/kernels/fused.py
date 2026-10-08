"""Fused, bit-exact Triton kernels for Qwen3's memory-bound elementwise work.

Between the FP8 projections, a Qwen3 layer runs dozens of small PyTorch ops
that each read and write a full activation tensor: RMSNorm (fp32 cast, square,
mean, rsqrt, two multiplies, bf16 cast), the residual add, SiLU and the gate
multiply, and RoPE (two multiplies, slice, negate, concatenate, add).  In a
2048-token prefill they cost ~95 ms of ~410 ms; in decode they are ~1,000 tiny
kernel launches per step.  Each kernel here does one pass over memory.

Every kernel reproduces the unfused PyTorch result bit for bit, so greedy
tokens do not change:
- intermediate bf16 roundings happen where PyTorch rounds, using integer
  round-to-nearest-even (``_round_bf16``); a ``.to(bf16).to(fp32)`` round trip
  can be folded by the compiler into an FMA and skip the rounding;
- RMSNorm's row sum reproduces PyTorch's ``mean`` reduction order exactly
  (``ATen/native/cuda/Reduce.cuh``): rows are read as 4-wide vectors spread
  over ``bw`` threads, each thread keeps four sequential accumulators combined
  as ``((a0 + a1) + a2) + a3``, then threads fold pairwise ``t += t + half``.
  PyTorch picks ``bw`` from the row count (512 threads for one row, 32 for 16
  or more), so ``rms_norm`` does the same;
- ``exp``/``rsqrt`` use the same libdevice functions as PyTorch's kernels.
These properties are verified on the L4 by ``tests/test_fused_kernels.py``.
"""

from __future__ import annotations

import types
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:  # pragma: no cover - depends on the runtime environment
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


def fused_kernels_available(device: torch.device | str | None = None) -> bool:
    return bool(triton is not None and torch.cuda.is_available()
                and (device is None or torch.device(device).type == "cuda"))


if triton is not None:

    @triton.jit
    def _round_bf16(x):
        """Round fp32 to the nearest bf16 (ties to even), returned as fp32."""
        bits = x.to(tl.uint32, bitcast=True)
        bits = (bits + (((bits >> 16) & 1) + 0x7FFF)) & 0xFFFF0000
        return bits.to(tl.float32, bitcast=True)

    @triton.jit
    def _fold_half(x, HALF: tl.constexpr):
        # x[t] + x[t + HALF]: one level of PyTorch's block/warp tree reduction.
        return tl.sum(tl.reshape(x, (2, HALF)), axis=0)

    @triton.jit
    def _tree_sum(x, BW: tl.constexpr):
        if BW >= 512:
            x = _fold_half(x, 256)
        if BW >= 256:
            x = _fold_half(x, 128)
        if BW >= 128:
            x = _fold_half(x, 64)
        if BW >= 64:
            x = _fold_half(x, 32)
        x = _fold_half(x, 16)
        x = _fold_half(x, 8)
        x = _fold_half(x, 4)
        x = _fold_half(x, 2)
        x = _fold_half(x, 1)
        return tl.sum(x, axis=0)

    @triton.jit
    def _load_row_value(x_ptr, residual_ptr, residual_out_ptr, x_offset, res_offset, mask,
                        HAS_RESIDUAL: tl.constexpr, STORE_RESIDUAL: tl.constexpr):
        x = tl.load(x_ptr + x_offset, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            # bf16 + bf16 in PyTorch: fp32 add, one rounding.
            residual = tl.load(residual_ptr + res_offset, mask=mask, other=0.0).to(tl.float32)
            x = _round_bf16(x + residual)
            if STORE_RESIDUAL:
                tl.store(residual_out_ptr + res_offset, x.to(tl.bfloat16), mask=mask)
        return x

    @triton.jit
    def _rms_norm_kernel(x_ptr, residual_ptr, residual_out_ptr, weight_ptr, out_ptr,
                         x_row_stride, out_row_stride, n_cols, eps,
                         HAS_RESIDUAL: tl.constexpr, BW: tl.constexpr, K: tl.constexpr):
        row = tl.program_id(0)
        lane = tl.arange(0, BW)
        x_row = row * x_row_stride
        res_row = row * n_cols
        n_vec = n_cols // 4
        quad = tl.arange(0, 4)
        acc = tl.zeros((BW, 4), tl.float32)  # column i = PyTorch's accumulator i
        for k in tl.static_range(K):
            vec = k * BW + lane
            col = vec[:, None] * 4 + quad[None, :]  # contiguous [BW, 4] tile: coalesced loads
            mask = (vec < n_vec)[:, None] & (quad < 4)[None, :]
            x = _load_row_value(x_ptr, residual_ptr, residual_out_ptr, x_row + col, res_row + col, mask, HAS_RESIDUAL, True)
            # Rounded multiplies keep the squares from fusing into the adds (PyTorch
            # squares in one kernel and sums in another).
            acc += libdevice.mul_rn(x, x)
        # Split [BW, 4] into its four columns: (BW, 2, 2) -> even/odd columns, then again.
        even, odd = tl.split(tl.reshape(acc, (BW, 2, 2)))  # cols (0, 2) and (1, 3)
        acc0, acc2 = tl.split(even)
        acc1, acc3 = tl.split(odd)
        # mean() and "+ eps" are separate PyTorch kernels: keep the multiply rounded
        # so it cannot fuse with the add into one FMA.
        variance = libdevice.mul_rn(_tree_sum(((acc0 + acc1) + acc2) + acc3, BW), 1.0 / n_cols)
        inv = libdevice.rsqrt(variance + eps)
        cols = tl.arange(0, BW * 4)
        for k in tl.static_range(K):
            col = k * BW * 4 + cols
            mask = col < n_cols
            x = _load_row_value(x_ptr, residual_ptr, residual_out_ptr, x_row + col, res_row + col, mask, HAS_RESIDUAL, False)
            normed = _round_bf16(x * inv)
            weight = tl.load(weight_ptr + col, mask=mask, other=0.0).to(tl.float32)
            tl.store(out_ptr + row * out_row_stride + col, (weight * normed).to(tl.bfloat16), mask=mask)

    @triton.jit
    def _silu_mul_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        # PyTorch: silu in fp32 (x / (1 + exp(-x))), rounded to bf16, then a bf16 multiply.
        silu = _round_bf16(libdevice.div_rn(gate, 1.0 + libdevice.exp(-gate)))
        tl.store(out_ptr + offsets, (silu * up).to(tl.bfloat16), mask=mask)

    @triton.jit
    def _rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, heads, seq,
                     x_sb, x_sh, x_st, cos_sb, cos_st, out_sb, out_sh, out_st,
                     HALF: tl.constexpr):
        pid = tl.program_id(0)
        t = pid % seq
        h = (pid // seq) % heads
        b = pid // (seq * heads)
        offs = tl.arange(0, HALF)
        x = x_ptr + b * x_sb + h * x_sh + t * x_st
        x1 = tl.load(x + offs).to(tl.float32)
        x2 = tl.load(x + HALF + offs).to(tl.float32)
        cos = cos_ptr + b * cos_sb + t * cos_st
        sin = sin_ptr + b * cos_sb + t * cos_st
        c1 = tl.load(cos + offs).to(tl.float32)
        c2 = tl.load(cos + HALF + offs).to(tl.float32)
        s1 = tl.load(sin + offs).to(tl.float32)
        s2 = tl.load(sin + HALF + offs).to(tl.float32)
        # PyTorch: (x * cos) + (rotate_half(x) * sin), each product rounded to bf16.
        out1 = _round_bf16(x1 * c1) + _round_bf16(-x2 * s1)
        out2 = _round_bf16(x2 * c2) + _round_bf16(x1 * s2)
        out = out_ptr + b * out_sb + h * out_sh + t * out_st
        tl.store(out + offs, out1.to(tl.bfloat16))
        tl.store(out + HALF + offs, out2.to(tl.bfloat16))


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float,
             residual: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """``Qwen3RMSNorm(x)``, or ``(norm(x + residual), x + residual)`` when fused with the add."""

    n_cols = int(x.shape[-1])
    if n_cols % 4 or n_cols < 128:
        raise ValueError("rms_norm reproduces PyTorch's vectorized reduction: width must be a multiple of 4, >= 128")
    rows = x.reshape(-1, n_cols)
    if rows.stride(-1) != 1:
        rows = rows.contiguous()
    num_rows = rows.shape[0]
    bw = _torch_reduce_block_width(num_rows, n_cols)
    k = triton.cdiv(n_cols // 4, bw)
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    out_rows = out.view(-1, n_cols)
    warps = max(1, min(8, bw // 32))
    if residual is None:
        _rms_norm_kernel[(num_rows,)](rows, rows, rows, weight, out_rows, rows.stride(0), n_cols,
                                      n_cols, eps, HAS_RESIDUAL=False, BW=bw, K=k, num_warps=warps)
        return out
    residual_rows = residual.reshape(-1, n_cols).contiguous()
    summed = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _rms_norm_kernel[(num_rows,)](rows, residual_rows, summed.view(-1, n_cols), weight, out_rows,
                                  rows.stride(0), n_cols, n_cols, eps,
                                  HAS_RESIDUAL=True, BW=bw, K=k, num_warps=warps)
    return out, summed


def _last_pow2(n: int) -> int:
    return 1 << (int(n).bit_length() - 1)


def _torch_reduce_block_width(num_rows: int, n_cols: int) -> int:
    """Threads PyTorch assigns to one row of an fp32 last-dim ``mean`` (setReduceConfig)."""

    max_threads, warp = 512, 32
    dim0 = n_cols // 4  # vectorized input
    dim0_pow2 = _last_pow2(dim0) if dim0 < max_threads else max_threads
    dim1_pow2 = _last_pow2(num_rows) if num_rows < max_threads else max_threads
    height = min(dim1_pow2, max_threads // min(dim0_pow2, warp))
    return min(dim0_pow2, max_threads // height)


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``F.silu(gate) * up`` in one pass."""

    gate, up = gate.contiguous(), up.contiguous()
    out = torch.empty_like(gate)
    n = gate.numel()
    block = 2048 if n >= 1 << 20 else 512  # small decode tensors need more programs
    _silu_mul_kernel[(triton.cdiv(n, block),)](gate, up, out, n, BLOCK=block, num_warps=8 if block == 2048 else 4)
    return out


def _rope_one(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    batch, heads, seq, dim = x.shape
    out = torch.empty_like(x)  # keeps the strides PyTorch's elementwise output would take
    _rope_kernel[(batch * heads * seq,)](
        x, cos, sin, out, heads, seq,
        x.stride(0), x.stride(1), x.stride(2),
        cos.stride(0) if cos.shape[0] > 1 else 0, cos.stride(1),
        out.stride(0), out.stride(1), out.stride(2), HALF=dim // 2,
    )
    return out


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Drop-in for ``transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb``."""

    supported = (
        unsqueeze_dim == 1 and q.is_cuda and q.dim() == 4 and k.dim() == 4 and cos.dim() == 3
        and q.dtype == k.dtype == cos.dtype == sin.dtype == torch.bfloat16
        and q.stride(-1) == 1 and k.stride(-1) == 1 and cos.stride(-1) == 1 and sin.stride() == cos.stride()
        and q.shape[-1] % 2 == 0 and cos.shape[0] in (1, q.shape[0]) and cos.shape[1] == q.shape[2]
    )
    if not supported:
        from transformers.models.qwen3 import modeling_qwen3
        original = _ORIGINAL_ROPE or modeling_qwen3.apply_rotary_pos_emb
        return original(q, k, cos, sin, unsqueeze_dim)
    return _rope_one(q, cos, sin), _rope_one(k, cos, sin)


# --- Installation ------------------------------------------------------------

_ORIGINAL_ROPE: Any = None
_ROPE_INSTALLS = 0


def _fusable_norm(hidden_states: torch.Tensor) -> bool:
    width = int(hidden_states.shape[-1])
    return hidden_states.dtype == torch.bfloat16 and hidden_states.is_cuda and width % 4 == 0 and width >= 128


def _norm_forward(self, hidden_states):
    if not _fusable_norm(hidden_states):
        return type(self).forward(self, hidden_states)
    return rms_norm(hidden_states, self.weight, self.variance_epsilon)


def _mlp_forward(self, x):
    return self.down_proj(silu_mul(self.gate_proj(x), self.up_proj(x)))


def _decoder_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
                     use_cache=False, position_embeddings=None, **kwargs):
    # Same as Qwen3DecoderLayer.forward, with the post-attention residual add
    # fused into the following RMSNorm.
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids,
        past_key_values=past_key_values, use_cache=use_cache, position_embeddings=position_embeddings, **kwargs,
    )
    norm = self.post_attention_layernorm
    if _fusable_norm(hidden_states):
        hidden_states, residual = rms_norm(hidden_states, norm.weight, norm.variance_epsilon, residual=residual)
    else:
        residual = residual + hidden_states
        hidden_states = norm(residual)
    hidden_states = self.mlp(hidden_states)
    return residual + hidden_states


def warm_fused_kernels(model: Any) -> None:
    """Compile every kernel specialization the model can hit, outside any timed run.

    RMSNorm specializes on PyTorch's per-row thread count, which depends on
    the row count (1, 2-3, 4-7, 8-15, 16+), so each width gets one call per class.
    """

    config = model.config
    device = next(model.parameters()).device
    widths = {int(config.hidden_size), int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))}
    with torch.inference_mode():
        for width in widths:
            if width % 4 or width < 128:
                continue
            weight = torch.ones(width, dtype=torch.bfloat16, device=device)
            for rows in (1, 2, 4, 8, 16):
                x = torch.zeros(rows, width, dtype=torch.bfloat16, device=device)
                rms_norm(x, weight, 1e-6)
                rms_norm(x, weight, 1e-6, residual=x)
        # silu_mul switches block size at 2**20 elements (about 108 prompt rows).
        intermediate = int(config.intermediate_size)
        for rows in (1, -(-(1 << 20) // intermediate)):
            gate = torch.zeros(rows, intermediate, dtype=torch.bfloat16, device=device)
            silu_mul(gate, gate)
        # Triton specializes integer arguments that equal 1 or are multiples of 16,
        # so RoPE compiles per (q heads, kv heads) x (decode, seq % 16 == 0, other):
        # warm one [B, T, H, D] -> [B, H, T, D] view of each, as projections produce.
        head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        heads = (int(config.num_attention_heads), int(getattr(config, "num_key_value_heads", config.num_attention_heads)))
        for seq in (1, 16, 17):
            q, k = (torch.zeros(1, seq, h, head_dim, dtype=torch.bfloat16, device=device).transpose(1, 2) for h in heads)
            cos = torch.zeros(1, seq, head_dim, dtype=torch.bfloat16, device=device)
            apply_rotary_pos_emb(q, k, cos, cos)
    torch.cuda.synchronize(device)


def install_fused_kernels(model: Any) -> bool:
    """Route this model's norms, MLPs, layer adds, and RoPE through the fused kernels.

    Per-module ``forward`` overrides; RoPE is a module-level function, so it is
    patched in Transformers' Qwen3 module and in this project's paged
    attention adapter (process-wide, reference counted). Returns False and
    changes nothing when Triton/CUDA is unavailable.
    """

    global _ORIGINAL_ROPE, _ROPE_INSTALLS
    if getattr(model, "_minillm_fused_kernels", False):
        return True
    device = next(model.parameters()).device
    if not fused_kernels_available(device):
        return False
    from transformers.models.qwen3 import modeling_qwen3
    from minillm_l4.engine.kv_cache import qwen3_paged

    for module in model.modules():
        if isinstance(module, modeling_qwen3.Qwen3RMSNorm):
            module.forward = types.MethodType(_norm_forward, module)
        elif isinstance(module, modeling_qwen3.Qwen3MLP):
            module.forward = types.MethodType(_mlp_forward, module)
        elif isinstance(module, modeling_qwen3.Qwen3DecoderLayer):
            module.forward = types.MethodType(_decoder_forward, module)
    if _ROPE_INSTALLS == 0:
        _ORIGINAL_ROPE = modeling_qwen3.apply_rotary_pos_emb
        modeling_qwen3.apply_rotary_pos_emb = apply_rotary_pos_emb
        qwen3_paged.apply_rotary_pos_emb = apply_rotary_pos_emb
    _ROPE_INSTALLS += 1
    model._minillm_fused_kernels = True
    warm_fused_kernels(model)
    return True


def uninstall_fused_kernels(model: Any) -> None:
    """Restore the unfused PyTorch path for ``model``."""

    global _ROPE_INSTALLS
    if not getattr(model, "_minillm_fused_kernels", False):
        return
    from transformers.models.qwen3 import modeling_qwen3
    from minillm_l4.engine.kv_cache import qwen3_paged

    for module in model.modules():
        if isinstance(module, (modeling_qwen3.Qwen3RMSNorm, modeling_qwen3.Qwen3MLP, modeling_qwen3.Qwen3DecoderLayer)):
            module.__dict__.pop("forward", None)
    _ROPE_INSTALLS -= 1
    if _ROPE_INSTALLS == 0:
        modeling_qwen3.apply_rotary_pos_emb = _ORIGINAL_ROPE
        qwen3_paged.apply_rotary_pos_emb = _ORIGINAL_ROPE
    model._minillm_fused_kernels = False


__all__ = [
    "apply_rotary_pos_emb", "fused_kernels_available", "install_fused_kernels",
    "rms_norm", "silu_mul", "uninstall_fused_kernels", "warm_fused_kernels",
]
