"""Same-session fused/split FP8 linear timings on the L4 (no model download).

  .conda-env/bin/python -m minillm_l4.scripts.bench_fp8_gemm
  /home/ubuntu/miniconda3/envs/nemo35_vllm/bin/python -m minillm_l4.scripts.bench_fp8_gemm --vllm

Time the complete linear call, including allocations and activation
quantization, with triton.testing.do_bench. Also report quantization and GEMM
alone on preallocated buffers. Warmup/autotuning are outside timing. Per-layer
totals are q + 2*kv + o + 2*gate_up + down, i.e. seven separate projections.
No claim of vLLM bitwise equivalence: its quantizer has different epsilon and
rounding conventions. --vllm times its quantizer + GEMM as requested.
"""

from __future__ import annotations

import argparse
import sys

import torch

from minillm_l4.engine.kernels import sm89_fp8 as fp8
from minillm_l4.engine.kernels.sm89_fp8_validation import (
    BENCH_MS, MODEL_SHAPES, bit_mismatch_details, bitwise_equal, make_activations, make_weights,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ms", nargs="+", type=int, default=BENCH_MS)
    parser.add_argument("--shapes", nargs="+", choices=[s.name for s in MODEL_SHAPES],
                        default=[s.name for s in MODEL_SHAPES])
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--warmup-ms", type=float, default=100)
    parser.add_argument("--rep-ms", type=float, default=300)
    parser.add_argument("--vllm", action="store_true")
    args = parser.parse_args(argv)
    if min(args.ms) < 1 or args.warmup_ms <= 0 or args.rep_ms <= 0:
        parser.error("M and timing durations must be positive")
    return args


def config_label(kernel) -> str:
    config = kernel.best_config
    return fp8.FP8GemmConfig(
        config.kwargs["BLOCK_M"], config.kwargs["BLOCK_N"], config.num_warps,
        config.num_stages, config.kwargs["GROUP_M"], config.kwargs["BLOCK_K"],
    ).label


def tflops(rows: int, n: int, k: int, milliseconds: float) -> float:
    return 2 * rows * n * k / (milliseconds * 1e9)


@torch.inference_mode()
def main(argv=None) -> int:
    args = parse_args(argv)
    if not fp8.sm89_available():
        print("ERROR: this script requires an NVIDIA L4/SM89 and Triton", file=sys.stderr)
        return 2
    from triton.testing import do_bench

    vllm_linear = None
    if args.vllm:
        try:
            from vllm.model_executor.layers.quantization.utils.fp8_utils import (
                per_token_group_quant_fp8, w8a8_triton_block_scaled_mm,
            )
        except ImportError as error:
            print(f"ERROR: --vllm requires the vLLM environment: {error}", file=sys.stderr)
            return 2

        def vllm_linear(x, weight, scales):
            quantized, a_scales = per_token_group_quant_fp8(
                x, 128, dtype=torch.float8_e4m3fn, use_ue8m0=False,
            )
            return w8a8_triton_block_scaled_mm(
                quantized, weight, a_scales, scales, [128, 128], output_dtype=x.dtype,
            )

    def bench(fn):
        # Use one scalar median consistently, with cache flushing by do_bench.
        return float(do_bench(fn, warmup=args.warmup_ms, rep=args.rep_ms, return_mode="median"))

    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    retries_before = torch.cuda.memory_stats().get("num_alloc_retries", 0)
    print(f"device={torch.cuda.get_device_name()} torch={torch.__version__} "
          f"triton={fp8.triton.__version__} seed={args.seed} threshold={fp8.SPLIT_M_THRESHOLD} "
          f"max_bucket={fp8.MAX_M_BUCKET} warmup_ms={args.warmup_ms} rep_ms={args.rep_ms}", flush=True)
    print("shape      M     fused_us  split_us  speedup  fused_TF split_TF quant_us gemm_us  split config", flush=True)
    if args.vllm:
        print("vLLM timing: quantizer + w8a8_triton_block_scaled_mm, separate projections", flush=True)
    totals = {m: {"fused": 0.0, "split": 0.0, "auto": 0.0, "vllm": 0.0} for m in args.ms}
    failures = 0
    for shape in MODEL_SHAPES:
        if shape.name not in args.shapes:
            continue
        weight, scales = make_weights(shape, args.seed)
        for rows in args.ms:
            x = make_activations(rows, shape.k, args.seed)
            fused_call = lambda: fp8.sm89_fp8_linear(x, weight, scales, [128, 128], path="fused")
            split_call = lambda: fp8.sm89_fp8_linear(x, weight, scales, [128, 128], path="split")
            fused = fused_call()
            fused_config = config_label(fp8._fp8_linear_kernel)
            split = split_call()
            split_config = config_label(fp8._fp8_split_gemm_kernel)
            if not bitwise_equal(fused, split):
                failures += 1
                print(f"FAIL exactness {shape.name} M={rows}: {bit_mismatch_details(fused, split)}", flush=True)
            del fused, split
            quantized, a_scales = fp8.quantize_sm89_fp8_activation(x)
            output = torch.empty((rows, shape.n), dtype=x.dtype, device=x.device)
            gemm_call = lambda: fp8._launch_split_gemm(quantized, weight, output, a_scales, scales)
            fp8._quantize_activation_kernel[(fp8.triton.cdiv(rows, fp8.QUANT_BLOCK_M), fp8.triton.cdiv(shape.k, 128))](
                x, quantized, a_scales, rows, shape.k, *x.stride(), a_scales.shape[1],
                BLOCK_M=fp8.QUANT_BLOCK_M, BLOCK_K=128, num_warps=4, num_stages=1,
            )
            quant_call = lambda: fp8._quantize_activation_kernel[
                (fp8.triton.cdiv(rows, fp8.QUANT_BLOCK_M), fp8.triton.cdiv(shape.k, 128))
            ](
                x, quantized, a_scales, rows, shape.k, *x.stride(), a_scales.shape[1],
                BLOCK_M=fp8.QUANT_BLOCK_M, BLOCK_K=128, num_warps=4, num_stages=1,
            )
            gemm_call()
            if vllm_linear is not None:
                vllm_linear(x, weight, scales)
            torch.cuda.synchronize()
            # Alternate the order to reduce systematic thermal/order bias.
            if rows % 2:
                fused_ms, split_ms = bench(fused_call), bench(split_call)
            else:
                split_ms, fused_ms = bench(split_call), bench(fused_call)
            quant_ms, gemm_ms = bench(quant_call), bench(gemm_call)
            print(f"{shape.name:<10} {rows:>6} {fused_ms * 1000:>9.1f} {split_ms * 1000:>9.1f} "
                  f"{fused_ms / split_ms:>7.3f} {tflops(rows, shape.n, shape.k, fused_ms):>9.2f} "
                  f"{tflops(rows, shape.n, shape.k, split_ms):>8.2f} {quant_ms * 1000:>8.1f} "
                  f"{gemm_ms * 1000:>7.1f} {split_config}", flush=True)
            print(f"  fused config={fused_config}; auto={fp8.select_fp8_path(rows)}", flush=True)
            for path, value in (("fused", fused_ms), ("split", split_ms)):
                totals[rows][path] += shape.count * value
            totals[rows]["auto"] += shape.count * (split_ms if fp8.select_fp8_path(rows) == "split" else fused_ms)
            if vllm_linear is not None:
                vllm_ms = bench(lambda: vllm_linear(x, weight, scales))
                totals[rows]["vllm"] += shape.count * vllm_ms
                print(f"  vllm_us={vllm_ms * 1000:.1f} vllm_TF={tflops(rows, shape.n, shape.k, vllm_ms):.2f} "
                      f"vllm/split={vllm_ms / split_ms:.3f}", flush=True)
            del x, quantized, a_scales, output
        del weight, scales
    all_shapes = set(args.shapes) == {s.name for s in MODEL_SHAPES}
    print("per-layer projection totals" if all_shapes else "selected-shape subtotals (with projection multiplicities)", flush=True)
    print("M       fused_us   split_us    auto_us   speedup" + ("   vllm_us" if args.vllm else ""), flush=True)
    for rows, values in totals.items():
        line = (f"{rows:>6} {values['fused'] * 1000:>10.1f} {values['split'] * 1000:>10.1f} "
                f"{values['auto'] * 1000:>10.1f} {values['fused'] / values['split']:>9.3f}")
        print(line + (f" {values['vllm'] * 1000:>9.1f}" if args.vllm else ""), flush=True)
    stats = torch.cuda.memory_stats()
    print(f"peak_allocated_MiB={torch.cuda.max_memory_allocated() / 2**20:.1f} "
          f"num_alloc_retries={stats.get('num_alloc_retries', 0) - retries_before} "
          f"exactness_failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
