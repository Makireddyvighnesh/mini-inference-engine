"""Deterministic data and raw-bit checks shared by the FP8 GPU scripts/tests.

Importing this module does not initialize CUDA. It is not part of inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelShape:
    name: str
    n: int
    k: int
    count: int = 1


MODEL_SHAPES = (
    ModelShape("q", 4096, 2560),
    ModelShape("kv", 1024, 2560, 2),
    ModelShape("o", 2560, 4096),
    ModelShape("gate_up", 9728, 2560, 2),
    ModelShape("down", 2560, 9728),
)
# Exercise both N and K masks, including a tail crossing a scale boundary.
TAIL_SHAPES = (ModelShape("tail_1", 129, 1), ModelShape("tail_127", 257, 127),
               ModelShape("tail_129", 193, 129), ModelShape("tail_259", 321, 259))
EXACT_MS = (1, 2, 3, 7, 16, 17, 64, 100, 128, 255, 256, 1000, 2048,
            4097, 8192, 16384, 20000)
BENCH_MS = (1, 16, 64, 256, 1024, 2048, 4096, 8192, 16384)


def make_weights(shape: ModelShape, seed: int, device: str = "cuda"):
    generator = torch.Generator(device=device).manual_seed(seed + shape.n * 101 + shape.k * 17)
    # Non-power-of-two scales and signed, nonuniform weights expose changes
    # in scale multiplication and accumulation order.
    weights = (torch.randn((shape.n, shape.k), generator=generator, device=device) * 32).clamp(-448, 448)
    weights = weights.to(torch.float8_e4m3fn)
    scales = torch.rand(
        ((shape.n + 127) // 128, (shape.k + 127) // 128),
        generator=generator, device=device, dtype=torch.float32,
    ) * 0.09 + 0.001
    return weights, scales


def make_activations(rows: int, k: int, seed: int, dtype=torch.bfloat16, device: str = "cuda"):
    generator = torch.Generator(device=device).manual_seed(seed + k * 31 + rows * 7)
    x = torch.randn((rows, k), generator=generator, device=device)
    x[:, ::31] *= 80
    # Row zero remains random, including at M=1. The others exercise zero
    # raw scales, large finite values, and scales below the denominator clamp.
    if rows > 1:
        x[1] = 0
    if rows > 2:
        x[2] *= 1e2 if dtype == torch.float16 else 1e12
        if dtype == torch.float16:
            x[2].clamp_(-60000, 60000)
    if rows > 3:
        x[3] *= 1e-7 if dtype == torch.float16 else 1e-15
    if rows > 4:
        x[4, :min(128, k)] = 0
        x[4, 128:min(256, k)] *= 1e-7 if dtype == torch.float16 else 1e-15
    if rows > 5:
        x[5] = -0.0
    if rows > 6:
        x[6, ::2] = 448
        x[6, 1::2] = -448
    return x.to(dtype)


def bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare representations, including zero signs and NaN payloads."""

    return bool(
        left.shape == right.shape and left.dtype == right.dtype
        and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))
    )


def bit_mismatch_details(left: torch.Tensor, right: torch.Tensor) -> str:
    if left.shape != right.shape or left.dtype != right.dtype:
        return f"shape/dtype {left.shape}/{left.dtype} != {right.shape}/{right.dtype}"
    # Outputs are bf16/fp16: count elements, rather than counting bytes.
    bits_left = left.contiguous().view(torch.int16)
    bits_right = right.contiguous().view(torch.int16)
    bad = bits_left != bits_right
    count = int(bad.sum().item())
    if not count:
        return "identical"
    first = tuple(int(i) for i in bad.nonzero()[0].tolist())
    return (
        f"{count}/{left.numel()} elements; first={first}; "
        f"bits=0x{int(bits_left[first].item()) & 0xffff:04x}/"
        f"0x{int(bits_right[first].item()) & 0xffff:04x}; "
        f"values={left[first].item()}/{right[first].item()}"
    )
