"""Merged FP8 projections: q/k/v as one GEMM, gate/up as one GEMM.

Qwen3's k and v projections are narrow (N=1024): one 128-wide tile column per
eight heads leaves the L4's 58 SMs underfilled, and each projection quantizes
the same activations again. Concatenating the weights along N runs q/k/v as
one N=6144 GEMM and gate/up as one N=19456 GEMM.

Exactness: an output column depends only on its own weight rows, their
128x128 block scales, and the activation quantization (per row and 128-column
group of K, identical for every projection of the same input). Every SM89
config shares BLOCK_K=128, so tile shape does not change a column's bits.
All split points are multiples of 128, so block scales concatenate without
straddling. The merged result is bitwise equal to the separate calls.

The original modules keep working: their weights become views of the merged
storage (no extra memory), so dense fallbacks and unmerged paths still run.
The returned q/k/v and gate/up are column views of one output; the fused
RMSNorm and SiLU-mul read them through strides instead of copying.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

QKV = ("q_proj", "k_proj", "v_proj")
GATE_UP = ("gate_proj", "up_proj")
_ATTRS = {QKV: "_minillm_qkv", GATE_UP: "_minillm_gate_up"}


@dataclass(frozen=True)
class MergedLinear:
    weight: torch.Tensor
    weight_scale_inv: torch.Tensor
    splits: tuple[int, ...]
    block_size: tuple[int, int]
    allow_deepgemm: bool


def _mergeable(linears: list[Any]) -> bool:
    first = linears[0]
    block = tuple(getattr(first, "block_size", None) or ())
    if len(block) != 2:
        return False
    for linear in linears:
        weight = getattr(linear, "weight", None)
        scales = getattr(linear, "weight_scale_inv", None)
        if (
            weight is None or weight.dtype != torch.float8_e4m3fn or not isinstance(scales, torch.Tensor)
            or getattr(linear, "bias", None) is not None or getattr(linear, "activation_scale", None) is not None
            or tuple(linear.block_size) != block or weight.shape[1] != first.weight.shape[1]
            or weight.shape[0] % block[0] or weight.device != first.weight.device
        ):
            return False
    return True


def _replace(linear: Any, name: str, value: torch.Tensor) -> None:
    if name in linear._parameters:
        linear._parameters[name] = torch.nn.Parameter(value, requires_grad=False)
    else:
        linear._buffers[name] = value


def _merge(module: Any, names: tuple[str, ...]) -> MergedLinear | None:
    linears = [getattr(module, name, None) for name in names]
    if any(linear is None for linear in linears) or not _mergeable(linears):
        return None
    block = tuple(int(b) for b in linears[0].block_size)
    weight = torch.cat([linear.weight.data for linear in linears])
    scales = torch.cat([linear.weight_scale_inv.data for linear in linears])
    splits, start = [], 0
    for linear in linears:
        rows = int(linear.weight.shape[0])
        _replace(linear, "weight", weight[start:start + rows])
        _replace(linear, "weight_scale_inv", scales[start // block[0]:(start + rows) // block[0]])
        splits.append(rows)
        start += rows
    return MergedLinear(weight, scales, tuple(splits), block,
                        allow_deepgemm=not getattr(linears[0], "_deepgemm_disabled", False))


def _linear(merged: MergedLinear, x: torch.Tensor) -> torch.Tensor:
    # The same call FP8Linear.forward makes, through whichever fp8_linear is installed.
    from transformers.integrations import finegrained_fp8

    return finegrained_fp8.fp8_linear(
        x, merged.weight, merged.weight_scale_inv, block_size=list(merged.block_size),
        activation_scale=None, bias=None, allow_deepgemm=merged.allow_deepgemm,
    )


def project_qkv(attention: Any, hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """``(q_proj(x), k_proj(x), v_proj(x))``; column views of one GEMM when merged."""

    merged = getattr(attention, "_minillm_qkv", None)
    if merged is None:
        return attention.q_proj(hidden_states), attention.k_proj(hidden_states), attention.v_proj(hidden_states)
    return _linear(merged, hidden_states).split(merged.splits, dim=-1)


def project_gate_up(mlp: Any, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """``(gate_proj(x), up_proj(x))``; column views of one GEMM when merged."""

    merged = getattr(mlp, "_minillm_gate_up", None)
    if merged is None:
        return mlp.gate_proj(x), mlp.up_proj(x)
    return _linear(merged, x).split(merged.splits, dim=-1)


def install_merged_projections(model: Any, pretune_tokens: int | None = None) -> int:
    """Merge every Qwen3 attention's q/k/v and MLP's gate/up; returns the merge count.

    Attention merges take effect in the paged attention paths
    (``engine/kv_cache/qwen3_paged.py``); MLP merges take effect in the fused
    MLP forward (``install_fused_kernels``). With the SM89 dispatcher
    installed, the merged shapes are pre-tuned here, so no timed request
    autotunes them.
    """

    from transformers.models.qwen3 import modeling_qwen3

    count = 0
    for module in model.modules():
        if isinstance(module, modeling_qwen3.Qwen3Attention):
            names = QKV
        elif isinstance(module, modeling_qwen3.Qwen3MLP):
            names = GATE_UP
        else:
            continue
        if getattr(module, _ATTRS[names], None) is not None:
            continue
        merged = _merge(module, names)
        if merged is not None:
            # Plain attribute, not a submodule: keeps state_dict and modules() unchanged.
            object.__setattr__(module, _ATTRS[names], merged)
            count += 1
    if count:
        _pretune(model, pretune_tokens)
    return count


def uninstall_merged_projections(model: Any) -> None:
    """Go back to separate projection calls (weights stay views; results are identical)."""

    for module in model.modules():
        for attr in _ATTRS.values():
            if attr in module.__dict__:
                object.__delattr__(module, attr)


def merged_linears(model: Any) -> list[MergedLinear]:
    return [module.__dict__[attr] for module in model.modules() for attr in _ATTRS.values()
            if attr in module.__dict__]


def _pretune(model: Any, max_tokens: int | None) -> None:
    from transformers.integrations import finegrained_fp8

    from minillm_l4.engine.kernels import sm89_fp8

    if finegrained_fp8.fp8_linear is not sm89_fp8.sm89_fp8_linear:
        return
    dtype = next(p.dtype for p in model.parameters() if p.dtype in {torch.bfloat16, torch.float16})
    sm89_fp8.pretune_sm89_fp8_shapes(
        [(m.weight, m.weight_scale_inv, m.block_size) for m in merged_linears(model)],
        dtype, sm89_fp8.MAX_M_BUCKET if max_tokens is None else max_tokens,
    )
