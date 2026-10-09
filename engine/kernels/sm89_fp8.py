"""SM89-tuned dynamic block-scaled FP8 linear for the NVIDIA L4.

The checkpoint stores FP8 E4M3 weights and one inverse scale per 128x128
weight block. Small calls quantize inside the GEMM; large calls quantize once
and share FP8 activations across N tiles. Both paths apply scales and
accumulate in FP32 in the same order. This implements the contract used by this
project and delegates unsupported calls to Transformers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - covered by the runtime availability check
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


SUPPORTED_BLOCK_SIZE = (128, 128)
# Row counts above this share its autotune bucket.  Every config uses
# BLOCK_K=128, so the config only changes speed, never a row's bits; a forward
# larger than any pre-tuned bucket would otherwise pay a multi-second search.
MAX_M_BUCKET = 16384
# Provisional crossover, to be revised from same-session L4 measurements.
# Explicit path="fused" / "split" is available regardless of this threshold.
SPLIT_M_THRESHOLD = 256
QUANT_BLOCK_M = 16
_original_fp8_linear: Callable[..., torch.Tensor] | None = None


@dataclass(frozen=True)
class FP8GemmConfig:
    """CPU-readable config; forcing one bypasses only GEMM autotuning."""

    block_m: int
    block_n: int
    num_warps: int
    num_stages: int
    group_m: int = 8
    block_k: int = 128

    @property
    def kwargs(self) -> dict[str, int]:
        return {
            "BLOCK_M": self.block_m, "BLOCK_N": self.block_n,
            "BLOCK_K": self.block_k, "GROUP_M": self.group_m,
            "PIPELINE_STAGES": self.num_stages,
        }

    @property
    def label(self) -> str:
        return (
            f"{self.block_m}x{self.block_n}x{self.block_k}"
            f"/w{self.num_warps}/s{self.num_stages}/g{self.group_m}"
        )

    def launch_kwargs(self) -> dict[str, int]:
        return {**self.kwargs, "num_warps": self.num_warps, "num_stages": self.num_stages}


# Ada has a 99 KiB shared-memory CTA limit. Avoid a full Cartesian product
# of tiles/stages. CPU-only Triton 3.7.1 compilation of aligned model shapes
# uses 33--99 KiB for these configs (including pipelined scale loads).
SPLIT_GEMM_CONFIGS = (
    FP8GemmConfig(64, 64, 4, 3),
    FP8GemmConfig(64, 64, 4, 4),
    FP8GemmConfig(64, 64, 4, 5),
    FP8GemmConfig(64, 128, 4, 3),
    FP8GemmConfig(64, 128, 4, 4),
    FP8GemmConfig(128, 64, 4, 3),
    FP8GemmConfig(128, 64, 4, 4),
    FP8GemmConfig(128, 128, 4, 3),
    FP8GemmConfig(128, 128, 8, 3),
    FP8GemmConfig(128, 128, 8, 4),
    FP8GemmConfig(64, 256, 4, 3),
    FP8GemmConfig(64, 256, 8, 3),
    FP8GemmConfig(128, 256, 8, 3),
    FP8GemmConfig(128, 128, 8, 3, group_m=16),
)


def m_bucket(rows: int) -> int:
    """Ceiling power of two, capped at the last pre-tuned bucket."""

    if rows < 0:
        raise ValueError("rows must be nonnegative")
    return min(1 << (max(1, rows) - 1).bit_length(), MAX_M_BUCKET)


def select_fp8_path(rows: int, path: str = "auto") -> str:
    """Resolve dispatch without touching CUDA; the threshold is inclusive."""

    if path not in {"auto", "fused", "split"}:
        raise ValueError(f"Unknown SM89 FP8 path: {path!r}")
    if rows < 0:
        raise ValueError("rows must be nonnegative")
    return ("split" if rows >= SPLIT_M_THRESHOLD else "fused") if path == "auto" else path


def pretune_row_buckets(max_tokens: int) -> tuple[int, ...]:
    """Cover the ceiling bucket too, including a non-power-of-two limit."""

    return tuple(1 << i for i in range(m_bucket(max(1, max_tokens)).bit_length()))


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

        # Widen only addresses: row * stride can exceed int32 on long prefills.
        address_m = offsets_m.to(tl.int64)
        address_n = offsets_n.to(tl.int64)
        address_k = offsets_k.to(tl.int64)
        activation_ptrs = (
            activation + address_m[:, None] * stride_am + address_k[None, :] * stride_ak
        )
        weight_ptrs = weight + address_k[:, None] * stride_wk + address_n[None, :] * stride_wn
        scale_ptrs = weight_scales + (address_n // 128) * stride_sn

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
                scale_ptrs + tl.cast(k_block, tl.int64) * stride_sk,
                mask=mask_n,
                other=1.0,
            )
            accumulator += (
                tl.dot(activation_fp8, weight_tile)
                * activation_scale[:, None]
                * weight_scale[None, :]
            )
            activation_ptrs += tl.cast(BLOCK_K, tl.int64) * stride_ak
            weight_ptrs += tl.cast(BLOCK_K, tl.int64) * stride_wk

        output_ptrs = output + address_m[:, None] * stride_om + address_n[None, :] * stride_on
        tl.store(
            output_ptrs,
            accumulator.to(output.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


    @triton.jit
    def _quantize_activation_kernel(
        activation, activation_fp8, activation_scales,
        M, K, stride_am, stride_ak, groups_k,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        offsets_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        k_block = tl.program_id(1)
        offsets_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = (offsets_m[:, None] < M) & (offsets_k[None, :] < K)
        address_m = offsets_m.to(tl.int64)
        address_k = offsets_k.to(tl.int64)
        activation_tile = tl.load(
            activation + address_m[:, None] * stride_am + address_k[None, :] * stride_ak,
            mask=mask, other=0.0,
        ).to(tl.float32)
        # Copy the fused arithmetic literally. In particular, store the raw
        # scale (zero for a zero block), not the clamped division denominator.
        activation_scale = tl.max(tl.abs(activation_tile), axis=1) / 448.0
        quantized = (
            activation_tile / tl.maximum(activation_scale[:, None], 1e-12)
        ).to(tl.float8e4nv)
        tl.store(
            activation_fp8 + address_m[:, None] * K + address_k[None, :],
            quantized, mask=mask,
        )
        tl.store(
            activation_scales + address_m * groups_k + tl.cast(k_block, tl.int64),
            activation_scale, mask=offsets_m < M,
        )


    @triton.autotune(
        configs=[
            triton.Config(c.kwargs, num_warps=c.num_warps, num_stages=c.num_stages)
            for c in SPLIT_GEMM_CONFIGS
        ],
        key=["M_BUCKET", "N", "K"],
        cache_results=True,
    )
    @triton.jit
    def _fp8_split_gemm_kernel(
        activation, weight, output, activation_scales, weight_scales,
        M, N, K,
        stride_am, stride_ak, stride_wn, stride_wk, stride_om, stride_on,
        stride_asm, stride_ask, stride_sn, stride_sk, M_BUCKET,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
        PIPELINE_STAGES: tl.constexpr,
    ):
        pid_m, pid_n = _grouped_tile(
            tl.program_id(0), tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N), GROUP_M,
        )
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)
        mask_m = offsets_m < M
        mask_n = offsets_n < N
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        address_m = offsets_m.to(tl.int64)
        address_n = offsets_n.to(tl.int64)
        address_k = offsets_k.to(tl.int64)
        activation_ptrs = (
            activation + address_m[:, None] * stride_am + address_k[None, :] * stride_ak
        )
        weight_ptrs = weight + address_k[:, None] * stride_wk + address_n[None, :] * stride_wn
        activation_scale_ptrs = activation_scales + address_m * stride_asm
        scale_ptrs = weight_scales + (address_n // 128) * stride_sn
        for k_block in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=PIPELINE_STAGES):
            mask_k = offsets_k < K - k_block * BLOCK_K
            activation_fp8 = tl.load(
                activation_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0,
            )
            weight_tile = tl.load(
                weight_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0,
            )
            activation_scale = tl.load(
                activation_scale_ptrs + tl.cast(k_block, tl.int64) * stride_ask, mask=mask_m, other=0.0,
            )
            weight_scale = tl.load(
                scale_ptrs + tl.cast(k_block, tl.int64) * stride_sk, mask=mask_n, other=1.0,
            )
            # Do not combine the two scales, accumulate a multi-block dot,
            # use split-K, change BLOCK_K, or change floating-point fusion.
            accumulator += (
                tl.dot(activation_fp8, weight_tile)
                * activation_scale[:, None]
                * weight_scale[None, :]
            )
            activation_ptrs += tl.cast(BLOCK_K, tl.int64) * stride_ak
            weight_ptrs += tl.cast(BLOCK_K, tl.int64) * stride_wk
        tl.store(
            output + address_m[:, None] * stride_om + address_n[None, :] * stride_on,
            accumulator.to(output.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


def quantize_sm89_fp8_activation(input_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D activation once, with raw fp32 scales per row/K block.

    This is separate from linear for timing the quantizer and forcing GEMM
    configs without requantizing the same input in the exactness sweep.
    """

    if input_2d.ndim != 2 or input_2d.dtype not in {torch.bfloat16, torch.float16}:
        raise ValueError("Expected a 2D bf16/fp16 activation")
    if not input_2d.is_cuda or not sm89_available(input_2d.device):
        raise RuntimeError("Activation quantization requires SM89 CUDA and Triton")
    m, k = input_2d.shape
    if k == 0:
        raise ValueError("Activation width must be positive")
    groups_k = triton.cdiv(k, 128)
    quantized = torch.empty((m, k), device=input_2d.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((m, groups_k), device=input_2d.device, dtype=torch.float32)
    if m:
        _quantize_activation_kernel[(triton.cdiv(m, QUANT_BLOCK_M), groups_k)](
            input_2d, quantized, scales, m, k,
            input_2d.stride(0), input_2d.stride(1), groups_k,
            BLOCK_M=QUANT_BLOCK_M, BLOCK_K=128, num_warps=4, num_stages=1,
        )
    return quantized, scales


def _launch_split_gemm(activation, weight, output, activation_scales, weight_scales, config=None):
    m, k = activation.shape
    n = weight.shape[0]
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)
    args = (
        activation, weight, output, activation_scales, weight_scales, m, n, k,
        *activation.stride(), *weight.stride(), *output.stride(),
        *activation_scales.stride(), *weight_scales.stride(), m_bucket(m),
    )
    if config is None:
        _fp8_split_gemm_kernel[grid](*args)
    else:
        # .fn is the undecorated JITFunction, so this runs exactly the given
        # config without reading or populating the autotuner cache.
        _fp8_split_gemm_kernel.fn[grid](*args, **config.launch_kwargs())


def sm89_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block_size: list[int] | tuple[int, int] | None = None,
    bias: torch.Tensor | None = None,
    activation_scale: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    allow_deepgemm: bool = True,
    *,
    path: str = "auto",
    gemm_config: FP8GemmConfig | None = None,
) -> torch.Tensor:
    """Transformers-compatible linear; explicit paths support bitwise A/B.

    ``gemm_config`` requires ``path="split"`` and bypasses its autotuner.
    Both paths preserve the fused floating-point arithmetic and BLOCK_K=128.
    """

    select_fp8_path(0, path)  # Reject typos even on the fallback path.
    if gemm_config is not None and (path != "split" or gemm_config not in SPLIT_GEMM_CONFIGS):
        raise ValueError("gemm_config must be a listed split config with path='split'")

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
    chosen_path = select_fp8_path(m, path)
    if m == 0 or n == 0:
        result = output.reshape(*input.shape[:-1], n)
        return result if bias is None else result + bias
    if chosen_path == "split":
        quantized, activation_scales = quantize_sm89_fp8_activation(input_2d)
        _launch_split_gemm(quantized, weight_2d, output, activation_scales, scales_2d, gemm_config)
        result = output.reshape(*input.shape[:-1], n)
        return result if bias is None else result + bias
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
        m_bucket(m),
    )
    result = output.reshape(*input.shape[:-1], n)
    return result if bias is None else result + bias


def pretune_sm89_fp8(model: Any, max_tokens: int = MAX_M_BUCKET) -> int:
    """Warm both GEMM paths and the quantizer for every row bucket/shape.

    Tune at power-of-two row counts through the capped ceiling bucket of
    ``max_tokens``. At buckets >=16, also launch ``bucket - 1`` to compile
    Triton's non-divisible-by-16 M specialization. Smaller buckets already
    have only that specialization (M==1 stays separate). Each extra call
    reuses the same M_BUCKET/N/K/dtype autotune key and selected config: one
    compilation/launch per GEMM, never another search. Split calls also warm
    the fixed-config quantizer; its M specializations are shared across
    buckets. Reuse the bucket's activation buffer to limit warmup overhead.

    In serving, N/K are fixed per model projection, and linear makes inputs,
    weights and scales contiguous and allocates contiguous outputs. Thus all
    strides (including ceil(K/128) scale strides) and quantizer groups_k are
    fixed per shape; only M varies. No do_not_specialize is needed for these
    integer args. Results persist in Triton's cache; returns shape/bucket
    pairs, regardless of the number of specialization warmup calls.
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
    buckets = pretune_row_buckets(int(max_tokens))
    with torch.inference_mode():
        for weight, scales, block in unique.values():
            for bucket in buckets:
                activation = torch.zeros((bucket, weight.shape[1]), dtype=dtype, device=weight.device)
                for rows in (bucket, bucket - 1) if bucket >= 16 else (bucket,):
                    for path in ("fused", "split"):
                        sm89_fp8_linear(activation[:rows], weight, scales, block_size=list(block), path=path)
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
