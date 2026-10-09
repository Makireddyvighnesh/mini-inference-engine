"""Exhaustive L4 bitwise gate for fused vs prequantized FP8 linear.

Run from the repository parent:
  .conda-env/bin/python -m minillm_l4.scripts.check_fp8_exact

Defaults cover every requested M, every model shape plus N/K tails, every
split GEMM config, and EVERY row's independence against 1-row fused and
forced split GEMMs. The row sweep is lengthy; --row-samples 16 provides an explicitly sampled
development check. A launch/compile failure counts as a failure, not a skip.
Use --dtypes bf16 fp16 --output-dtypes bf16 fp16 for the full dtype contract.
Use --check-pretune-only --ms 1000 1001 2047 to report cold pretune time and
first/repeat call latency, failing on any new autotune/JIT cache entry after
pretune. Set TRITON_CACHE_DIR to a fresh directory before that invocation.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from minillm_l4.engine.kernels import sm89_fp8 as fp8
from minillm_l4.engine.kernels.sm89_fp8_validation import (
    EXACT_MS, MODEL_SHAPES, TAIL_SHAPES, bit_mismatch_details,
    bitwise_equal, make_activations, make_weights,
)


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ms", type=int, nargs="+", default=EXACT_MS)
    parser.add_argument("--shapes", nargs="+", choices=[s.name for s in MODEL_SHAPES],
                        default=[s.name for s in MODEL_SHAPES])
    parser.add_argument("--no-tails", action="store_true")
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=["bf16"])
    parser.add_argument("--output-dtypes", nargs="+", choices=DTYPES,
                        help="default: same dtype as the input")
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--row-samples", type=int, default=0,
                        help="0 checks ALL rows; positive values sample evenly, always including special rows")
    parser.add_argument("--check-pretune-only", action="store_true",
                        help="warm serving shapes, then check first/repeat calls add no autotune/JIT entries; no tails")
    args = parser.parse_args(argv)
    if min(args.ms) < 1 or args.row_samples < 0:
        parser.error("M must be positive and --row-samples nonnegative")
    return args


def row_indices(rows: int, samples: int) -> tuple[int, ...] | range:
    if not samples or samples >= rows:
        return range(rows)
    indices = set(range(min(rows, 7)))
    indices.update((i * (rows - 1)) // max(1, samples - 1) for i in range(samples))
    indices.add(rows - 1)
    return tuple(sorted(indices))


def _one_row_reference(x, weight, scales, output_dtype, indices):
    # Real 1-row calls, not another multi-row kernel posing as a reference.
    # Concatenation in bounded batches keeps allocator pressure manageable.
    chunks, pending = [], []
    for row in indices:
        pending.append(fp8.sm89_fp8_linear(
            x[row:row + 1], weight, scales, [128, 128],
            path="fused", output_dtype=output_dtype,
        ))
        if len(pending) == 256:
            chunks.append(torch.cat(pending))
            pending.clear()
    if pending:
        chunks.append(torch.cat(pending))
    return torch.cat(chunks)


def _pretune_cache_sizes() -> tuple[int, ...]:
    # Validation-only inspection of the installed Triton runtime's caches.
    gemms = (fp8._fp8_linear_kernel, fp8._fp8_split_gemm_kernel)
    jits = tuple(kernel.fn for kernel in gemms) + (fp8._quantize_activation_kernel,)
    return tuple(len(kernel.cache) for kernel in gemms) + tuple(
        sum(len(cache[0]) for cache in jit.device_caches.values()) for jit in jits
    )


def _check_pretune_only(args) -> int:
    model = torch.nn.Module()
    shapes = [shape for shape in MODEL_SHAPES if shape.name in args.shapes]
    for shape in shapes:
        projection = torch.nn.Module()
        weight, scales = make_weights(shape, args.seed)
        projection.weight = torch.nn.Parameter(weight, requires_grad=False)
        projection.register_buffer("weight_scale_inv", scales)
        model.add_module(shape.name, projection)
    failures = 0
    print("cache counts: fused/split autotune, fused/split/quantizer JIT", flush=True)

    def timed(call):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = call()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        del result
        return elapsed_ms

    for dtype_name in args.dtypes:
        dtype = DTYPES[dtype_name]
        model.dtype_anchor = torch.nn.Parameter(torch.zeros(1, device="cuda", dtype=dtype), requires_grad=False)
        start = time.perf_counter()
        pairs = fp8.pretune_sm89_fp8(model, max_tokens=max(args.ms))
        pretune_ms = (time.perf_counter() - start) * 1000
        warmed = _pretune_cache_sizes()
        print(f"pretune dtype={dtype_name} pairs={pairs} ms={pretune_ms:.1f} caches={warmed}", flush=True)
        for shape in shapes:
            projection = getattr(model, shape.name)
            for rows in args.ms:
                x = make_activations(rows, shape.k, args.seed, dtype)
                for path in ("fused", "split", "quantizer"):
                    def call():
                        if path == "quantizer":
                            return fp8.quantize_sm89_fp8_activation(x)
                        return fp8.sm89_fp8_linear(
                            x, projection.weight, projection.weight_scale_inv, [128, 128], path=path,
                        )

                    first_ms, repeat_ms = timed(call), timed(call)
                    after = _pretune_cache_sizes()
                    ok = after == warmed
                    failures += not ok
                    print(f"{shape.name} M={rows} {dtype_name} {path} "
                          f"first_ms={first_ms:.3f} repeat_ms={repeat_ms:.3f} "
                          f"caches={after} {'PASS' if ok else 'FAIL new cache entry'}", flush=True)
                del x
    print(f"{'FAIL' if failures else 'PASS'}: pretune specialization coverage; {failures} failures", flush=True)
    return 1 if failures else 0


@torch.inference_mode()
def main(argv=None) -> int:
    args = parse_args(argv)
    if not fp8.sm89_available():
        print("ERROR: this script requires an NVIDIA L4/SM89 and Triton", file=sys.stderr)
        return 2
    torch.manual_seed(args.seed)
    print(f"device={torch.cuda.get_device_name()} torch={torch.__version__} "
          f"triton={fp8.triton.__version__} seed={args.seed} "
          f"threshold={fp8.SPLIT_M_THRESHOLD} max_bucket={fp8.MAX_M_BUCKET}", flush=True)
    if args.check_pretune_only:
        return _check_pretune_only(args)
    print(f"row independence: {'ALL rows' if not args.row_samples else 'SAMPLED rows'}; "
          f"split configs={len(fp8.SPLIT_GEMM_CONFIGS)}", flush=True)
    for i, config in enumerate(fp8.SPLIT_GEMM_CONFIGS):
        print(f"c{i:02d} {config.label}", flush=True)
    print("shape      M      in/out     auto  split configs  rows(old/all-new)  status", flush=True)
    shapes = [s for s in MODEL_SHAPES if s.name in args.shapes]
    if not args.no_tails:
        shapes += list(TAIL_SHAPES)
    failures, cases = [], 0

    def check(label, reference, actual):
        if not bitwise_equal(reference, actual):
            detail = f"{label}: {bit_mismatch_details(reference, actual)}"
            failures.append(detail)
            print(f"  FAIL {detail}", flush=True)
            return False
        return True

    for shape in shapes:
        weight, scales = make_weights(shape, args.seed)
        for rows in args.ms:
            for dtype_name in args.dtypes:
                x = make_activations(rows, shape.k, args.seed, DTYPES[dtype_name])
                for out_name in args.output_dtypes or [dtype_name]:
                    cases += 1
                    before = len(failures)
                    prefix = f"{shape.name} M={rows} {dtype_name}/{out_name}"
                    try:
                        reference = fp8.sm89_fp8_linear(
                            x, weight, scales, [128, 128], path="fused", output_dtype=DTYPES[out_name],
                        )
                        indices = row_indices(rows, args.row_samples)
                        row_reference = _one_row_reference(x, weight, scales, DTYPES[out_name], indices)
                        index = torch.tensor(list(indices), device=x.device, dtype=torch.long)
                        row_ok = check(f"{prefix} fused row independence", row_reference, reference[index])
                        auto = fp8.sm89_fp8_linear(x, weight, scales, [128, 128], output_dtype=DTYPES[out_name])
                        auto_ok = check(f"{prefix} auto", reference, auto)
                        del auto
                        split = fp8.sm89_fp8_linear(
                            x, weight, scales, [128, 128], path="split", output_dtype=DTYPES[out_name],
                        )
                        split_ok = check(f"{prefix} split autotuned", reference, split)
                        row_ok &= check(f"{prefix} split row independence", row_reference, split[index])
                        del split
                        quantized, activation_scales = fp8.quantize_sm89_fp8_activation(x)
                        output = torch.empty_like(reference)
                        single_outputs = torch.empty_like(row_reference)
                        config_passes = 0
                        for i, config in enumerate(fp8.SPLIT_GEMM_CONFIGS):
                            try:
                                fp8._launch_split_gemm(quantized, weight, output, activation_scales, scales, config)
                                # Comparing EVERY element also covers each config's row independence.
                                config_passes += check(f"{prefix} c{i:02d}", reference, output)
                                row_ok &= check(f"{prefix} c{i:02d} row independence", row_reference, output[index])
                                # Real M=1 GEMMs under EACH forced config. Use
                                # slices of the quantized input; below we also
                                # prove those bytes/scales equal 1-row quantization.
                                # Store all selected rows before comparing, to
                                # avoid a host/device sync after every row.
                                for position, row in enumerate(indices):
                                    fp8._launch_split_gemm(
                                        quantized[row:row + 1], weight,
                                        single_outputs[position:position + 1],
                                        activation_scales[row:row + 1], scales, config,
                                    )
                                row_ok &= check(f"{prefix} c{i:02d} split 1-row calls", row_reference, single_outputs)
                            except Exception as error:
                                failures.append(f"{prefix} c{i:02d}: {type(error).__name__}: {error}")
                                print(f"  FAIL {failures[-1]}", flush=True)
                        # Quantization itself must be row-independent, too.
                        quant_row_ok = True
                        for row in indices:
                            q1, s1 = fp8.quantize_sm89_fp8_activation(x[row:row + 1])
                            if not bitwise_equal(q1, quantized[row:row + 1]) or not bitwise_equal(s1, activation_scales[row:row + 1]):
                                failures.append(f"{prefix} quantizer row={row}")
                                print(f"  FAIL {failures[-1]}", flush=True)
                                quant_row_ok = False
                                break
                        row_ok &= quant_row_ok
                        status = "PASS" if len(failures) == before else "FAIL"
                        print(f"{shape.name:<10} {rows:>6} {dtype_name + '/' + out_name:>10} "
                              f"{'PASS' if auto_ok else 'FAIL':>5} {'PASS' if split_ok else 'FAIL':>5} "
                              f"{config_passes:>2}/{len(fp8.SPLIT_GEMM_CONFIGS):<2} "
                              f"{len(indices):>6}/{'PASS' if row_ok else 'FAIL':<4} {status}", flush=True)
                        del reference, row_reference, output, single_outputs, quantized, activation_scales
                    except Exception as error:
                        failures.append(f"{prefix}: {type(error).__name__}: {error}")
                        print(f"{shape.name:<10} {rows:>6} {dtype_name + '/' + out_name:>10} FAIL {failures[-1]}", flush=True)
                del x
        del weight, scales
    torch.cuda.synchronize()
    print(f"{'FAIL' if failures else 'PASS'}: {cases} cases; {len(failures)} failures", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
