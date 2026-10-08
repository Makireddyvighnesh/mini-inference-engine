"""SM89-tuned dynamic block-scaled FP8 linear for the NVIDIA L4.

The checkpoint stores FP8 E4M3 weights and one inverse scale per 128x128
weight block.  This kernel quantizes each 128-wide activation block at run
time, executes the FP8 matrix product, applies both scales, and accumulates in
FP32.  It deliberately implements only the checkpoint contract used by this
project and delegates unsupported calls to Transformers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - covered by the runtime availability check
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


SUPPORTED_BLOCK_SIZE = (128, 128)
_original_fp8_linear: Callable[..., torch.Tensor] | None = None


def sm89_available(device: torch.device | str | None = None) -> bool:
    """Return whether Triton and an NVIDIA compute-capability 8.9 GPU exist."""

    return bool(
        triton is not None
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability(device) == (8, 9)
    )


def _supports_call(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block_size: list[int] | tuple[int, int] | None,
    activation_scale: torch.Tensor | None,
) -> bool:
    return bool(
        input.is_cuda
        and weight.is_cuda
        and weight_scale_inv.is_cuda
        and sm89_available(weight.device)
        and input.dtype in {torch.float16, torch.bfloat16}
        and weight.dtype == torch.float8_e4m3fn
        and weight_scale_inv.dtype == torch.float32
        and input.ndim >= 2
        and weight.ndim == 2
        and weight_scale_inv.ndim == 2
        and block_size is not None
        and tuple(block_size) == SUPPORTED_BLOCK_SIZE
        and activation_scale is None
        and input.shape[-1] == weight.shape[1]
        and tuple(weight_scale_inv.shape)
        == ((weight.shape[0] + 127) // 128, (weight.shape[1] + 127) // 128)
    )


if triton is not None:

    @triton.jit
    def _grouped_tile(pid, tiles_m, tiles_n, group_m: tl.constexpr):
        tiles_per_group = group_m * tiles_n
        group_id = pid // tiles_per_group
        first_m = group_id * group_m
        actual_group_m = tl.minimum(tiles_m - first_m, group_m)
        within_group = pid % tiles_per_group
        return first_m + within_group % actual_group_m, within_group // actual_group_m


    @triton.autotune(
        configs=[
            # Autoregressive decode has M=1.  A 16-row tile is appropriate for
            # prefill but wastes work and registers for one live row, even
            # though the masked rows are never stored.  Keep a dedicated
            # vector-matrix family for the batch-1 and small-batch regime.
            triton.Config(
                {"BLOCK_M": 1, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 1},
                num_warps=4,
                num_stages=2,
            ),
            triton.Config(
                {"BLOCK_M": 1, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 1},
                num_warps=4,
                num_stages=2,
            ),
            triton.Config(
                {"BLOCK_M": 1, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 1},
                num_warps=8,
                num_stages=2,
            ),
            triton.Config(
                {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8},
                num_warps=4,
                num_stages=2,
            ),
            triton.Config(
                {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
                num_warps=8,
                num_stages=3,
            ),
            triton.Config(
                {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
                num_warps=16,
                num_stages=3,
            ),
            triton.Config(
                {"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8},
                num_warps=16,
                num_stages=3,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
                num_warps=8,
                num_stages=3,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
                num_warps=8,
                num_stages=3,
            ),
        ],
        # Tune per power-of-two row bucket, not per exact M: continuous
        # batching and flattened prefill produce ever-new token counts, and a
        # fresh per-M search costs seconds inside a live request.
        key=["M_BUCKET", "N", "K"],
        cache_results=True,
    )
    @triton.jit
    def _fp8_linear_kernel(
        activation,
        weight,
        output,
        weight_scales,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_wn,
        stride_wk,
        stride_om,
        stride_on,
        stride_sn,
        stride_sk,
        M_BUCKET,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        tiles_m = tl.cdiv(M, BLOCK_M)
        tiles_n = tl.cdiv(N, BLOCK_N)
        pid_m, pid_n = _grouped_tile(pid, tiles_m, tiles_n, GROUP_M)

        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)
        mask_m = offsets_m < M
        mask_n = offsets_n < N
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        activation_ptrs = (
            activation + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
        )
        weight_ptrs = weight + offsets_k[:, None] * stride_wk + offsets_n[None, :] * stride_wn
        scale_ptrs = weight_scales + (offsets_n // 128) * stride_sn

        for k_block in range(0, tl.cdiv(K, BLOCK_K)):
            remaining_k = K - k_block * BLOCK_K
            mask_k = offsets_k < remaining_k
            activation_tile = tl.load(
                activation_ptrs,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            activation_scale = tl.max(tl.abs(activation_tile), axis=1) / 448.0
            activation_fp8 = (
                activation_tile / tl.maximum(activation_scale[:, None], 1e-12)
            ).to(tl.float8e4nv)
            weight_tile = tl.load(
                weight_ptrs,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            )
            weight_scale = tl.load(
                scale_ptrs + k_block * stride_sk,
                mask=mask_n,
                other=1.0,
            )
            accumulator += (
                tl.dot(activation_fp8, weight_tile)
                * activation_scale[:, None]
                * weight_scale[None, :]
            )
            activation_ptrs += BLOCK_K * stride_ak
            weight_ptrs += BLOCK_K * stride_wk

        output_ptrs = output + offsets_m[:, None] * stride_om + offsets_n[None, :] * stride_on
        tl.store(
            output_ptrs,
            accumulator.to(output.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


def sm89_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block_size: list[int] | tuple[int, int] | None = None,
    bias: torch.Tensor | None = None,
    activation_scale: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    allow_deepgemm: bool = True,
) -> torch.Tensor:
    """Transformers-compatible FP8 linear with a guarded fallback."""

    if not _supports_call(
        input, weight, weight_scale_inv, block_size, activation_scale
    ):
        if _original_fp8_linear is None:
            raise RuntimeError("Unsupported SM89 FP8 call and no fallback is installed")
        return _original_fp8_linear(
            input,
            weight,
            weight_scale_inv,
            block_size=block_size,
            bias=bias,
            activation_scale=activation_scale,
            output_dtype=output_dtype,
            allow_deepgemm=allow_deepgemm,
        )
    if output_dtype is not None and output_dtype not in {
        torch.float16,
        torch.bfloat16,
    }:
        raise TypeError("SM89 FP8 output must be float16 or bfloat16")

    k = int(input.shape[-1])
    m = int(input.numel() // k)
    n = int(weight.shape[0])
    input_2d = input.reshape(m, k).contiguous()
    weight_2d = weight.contiguous()
    scales_2d = weight_scale_inv.contiguous()
    output = torch.empty(
        (m, n), device=input.device, dtype=output_dtype or input.dtype
    )
    grid = lambda meta: (  # noqa: E731
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    _fp8_linear_kernel[grid](
        input_2d,
        weight_2d,
        output,
        scales_2d,
        m,
        n,
        k,
        input_2d.stride(0),
        input_2d.stride(1),
        weight_2d.stride(0),
        weight_2d.stride(1),
        output.stride(0),
        output.stride(1),
        scales_2d.stride(0),
        scales_2d.stride(1),
        1 << (m - 1).bit_length(),
    )
    result = output.reshape(*input.shape[:-1], n)
    return result if bias is None else result + bias


def pretune_sm89_fp8(model: Any, max_tokens: int = 16384) -> int:
    """Autotune every power-of-two row bucket up to ``max_tokens`` for each
    FP8 projection shape in ``model``, so no search happens inside a timed
    request.  Results persist in the Triton cache; returns the bucket count.
    """

    shapes = {
        (module.weight, module.weight_scale_inv, tuple(getattr(module, "block_size", SUPPORTED_BLOCK_SIZE)))
        for module in model.modules()
        if getattr(getattr(module, "weight", None), "dtype", None) == torch.float8_e4m3fn
        and isinstance(getattr(module, "weight_scale_inv", None), torch.Tensor)
    }
    unique = {}
    for weight, scales, block in shapes:
        unique.setdefault(tuple(weight.shape), (weight, scales, block))
    dtype = next(p.dtype for p in model.parameters() if p.dtype in {torch.bfloat16, torch.float16})
    buckets = [1 << i for i in range(max(1, int(max_tokens)).bit_length())]
    with torch.inference_mode():
        for weight, scales, block in unique.values():
            for rows in buckets:
                sm89_fp8_linear(
                    torch.zeros((rows, weight.shape[1]), dtype=dtype, device=weight.device),
                    weight, scales, block_size=list(block),
                )
    torch.cuda.synchronize()
    return len(unique) * len(buckets)


def install_sm89_fp8_dispatch() -> dict[str, Any]:
    """Install the custom dispatcher after validating the target GPU."""

    if not sm89_available():
        capability = (
            list(torch.cuda.get_device_capability())
            if torch.cuda.is_available()
            else None
        )
        raise RuntimeError(
            "The custom FP8 kernel requires an NVIDIA compute-capability 8.9 GPU; "
            f"detected capability={capability}"
        )

    from transformers.integrations import finegrained_fp8

    global _original_fp8_linear
    if finegrained_fp8.fp8_linear is not sm89_fp8_linear:
        _original_fp8_linear = finegrained_fp8.fp8_linear
        finegrained_fp8.fp8_linear = sm89_fp8_linear
    return {
        "installed": True,
        "backend": "minillm_sm89_triton_fp8",
        "compute_capability_required": [8, 9],
        "block_size": [128, 128],
        "fused_operations": [
            "dynamic activation scaling",
            "FP8 E4M3 activation quantization",
            "block-scaled FP8 matrix multiplication",
        ],
        "fallback": "transformers_fine_grained_fp8",
    }
