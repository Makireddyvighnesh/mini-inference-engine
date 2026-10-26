"""Measured GEMM throughput per precision vs NVIDIA's L4 datasheet.

Sections (one session, no model download):

* square  - M=N=K from 512 to 16384 for FP32, TF32, FP16, BF16 (cuBLAS
            matmul), FP8 e4m3 (cuBLASLt ``torch._scaled_mm``, per-tensor
            scales, bf16 output), INT8 (``torch._int_mm``, int32 output), and
            our block-scaled Triton FP8 GEMM (``sm89_fp8`` split kernel on
            pre-quantized activations);
* model   - Qwen3-4B projection shapes (q, kv, o, gate_up, down) at
            M=16..8192 rows: BF16 matmul, cuBLAS FP8, and our full FP8 linear
            (activation quantization included, as the engine calls it);
* memory  - device-to-device copy bandwidth vs the 300 GB/s datasheet value.

Datasheet peaks are NVIDIA's dense numbers (the sparse figures halved) at the
2040 MHz boost clock. The L4 is capped at 72 W and lowers its SM clock under
sustained tensor-core load, so every case also records the median SM clock,
power, temperature and clock-limit reasons (NVML, sampled every 20 ms while
the case runs), and reports a clock-adjusted peak = datasheet * clock / 2040.
Timing: triton.testing.do_bench median (L2 flushed between runs).

  .conda-env/bin/python -m minillm_l4.benchmarks.commands.run_gemm_peak \\
      --output-dir minillm_l4/results/gemm_peak_<date>
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import torch

from minillm_l4.benchmarks.core.hardware import collect_environment_metadata

SECTIONS = ("square", "model", "memory")
BOOST_CLOCK_MHZ = 2040
# NVIDIA L4 datasheet (TF32/FP16/BF16/FP8/INT8 are published "with sparsity";
# dense is half). Units: TFLOPS (TOPS for INT8), GB/s for memory.
DATASHEET = {
    "fp32": 30.3, "tf32": 60.0, "fp16": 121.0, "bf16": 121.0,
    "fp8": 242.5, "int8": 242.5, "memory_gbps": 300.0,
}
SQUARE_SIZES = (512, 1024, 2048, 4096, 8192, 16384)
MODEL_ROWS = (16, 64, 256, 1024, 2048, 4096, 8192)
# Throttle-reason bits from nvml.h.
REASON_BITS = {0x1: "gpu_idle", 0x2: "app_clock", 0x4: "sw_power_cap", 0x8: "hw_slowdown",
               0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake"}


class Nvml:
    """Minimal ctypes NVML reader (clock, power, temperature, limit reasons)."""

    def __init__(self, device: int = 0):
        self.lib = None
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            if lib.nvmlInit_v2() != 0:
                return
            p = torch.cuda.get_device_properties(device)
            bus = f"{p.pci_domain_id:08X}:{p.pci_bus_id:02X}:{p.pci_device_id:02X}.0"
            self.handle = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByPciBusId_v2(bus.encode(), ctypes.byref(self.handle)) == 0:
                self.lib = lib
        except (OSError, AttributeError):
            pass

    def sample(self) -> dict | None:
        if self.lib is None:
            return None
        clock, power, temp, reasons = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint(), ctypes.c_ulonglong()
        self.lib.nvmlDeviceGetClockInfo(self.handle, 1, ctypes.byref(clock))  # NVML_CLOCK_SM
        self.lib.nvmlDeviceGetPowerUsage(self.handle, ctypes.byref(power))
        self.lib.nvmlDeviceGetTemperature(self.handle, 0, ctypes.byref(temp))
        self.lib.nvmlDeviceGetCurrentClocksThrottleReasons(self.handle, ctypes.byref(reasons))
        return {"sm_mhz": clock.value, "watts": power.value / 1000, "temp_c": temp.value,
                "reasons": reasons.value}


class Telemetry:
    """Background NVML sampling for the duration of a `with` block."""

    def __init__(self, nvml: Nvml, interval_s: float = 0.02):
        self.nvml, self.interval_s, self.samples = nvml, interval_s, []

    def __enter__(self):
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _run(self):
        while not self.stop.is_set():
            sample = self.nvml.sample()
            if sample is not None:
                self.samples.append(sample)
            time.sleep(self.interval_s)

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()

    def summary(self) -> dict:
        if not self.samples:
            return {}
        count = len(self.samples)
        # Fraction of samples in which each clock-limit reason was active.
        reasons = {name: sum(bool(s["reasons"] & bit) for s in self.samples) / count
                   for bit, name in REASON_BITS.items() if bit != 0x1}
        return {
            "sm_mhz_median": statistics.median(s["sm_mhz"] for s in self.samples),
            "watts_median": statistics.median(s["watts"] for s in self.samples),
            "temp_c_max": max(s["temp_c"] for s in self.samples),
            "limit_reasons": {name: round(f, 2) for name, f in reasons.items() if f >= 0.05},
        }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sections", nargs="+", choices=SECTIONS, default=list(SECTIONS))
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SQUARE_SIZES))
    parser.add_argument("--rows", nargs="+", type=int, default=list(MODEL_ROWS))
    parser.add_argument("--warmup-ms", type=float, default=300)
    parser.add_argument("--rep-ms", type=float, default=1500,
                        help="timed window per case; long enough to reach the sustained (power-capped) clock")
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory is not empty: {args.output_dir}")
    return args


def make_square_cases(n: int):
    """Return {precision: callable} for an n x n x n GEMM."""

    from minillm_l4.engine.kernels import sm89_fp8 as fp8

    gen = torch.Generator(device="cuda").manual_seed(n)
    a32 = torch.randn((n, n), device="cuda", generator=gen)
    b32 = torch.randn((n, n), device="cuda", generator=gen)
    a16, b16 = a32.half(), b32.half()
    abf, bbf = a32.bfloat16(), b32.bfloat16()
    a8 = a32.to(torch.float8_e4m3fn)
    w8 = b32.to(torch.float8_e4m3fn)  # [N, K] row-major; .t() is the column-major B cuBLASLt needs
    one = torch.ones((), device="cuda")
    ai8 = (a32 * 20).clamp(-127, 127).to(torch.int8)
    wi8 = (b32 * 20).clamp(-127, 127).to(torch.int8)
    blocks = (n + 127) // 128
    a_scales = torch.full((n, blocks), 0.01, device="cuda")
    w_scales = torch.full((blocks, blocks), 0.01, device="cuda")
    out_ours = torch.empty((n, n), device="cuda", dtype=torch.bfloat16)

    def fp32():
        torch.backends.cuda.matmul.allow_tf32 = False
        return a32 @ b32

    def tf32():
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            return a32 @ b32
        finally:
            torch.backends.cuda.matmul.allow_tf32 = False

    return {
        "fp32": fp32,
        "tf32": tf32,
        "fp16": lambda: a16 @ b16,
        "bf16": lambda: abf @ bbf,
        "fp8": lambda: torch._scaled_mm(a8, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16),
        "int8": lambda: torch._int_mm(ai8, wi8.t()),
        "fp8_ours": lambda: fp8._launch_split_gemm(a8, w8, out_ours, a_scales, w_scales),
    }


def make_model_cases(rows: int, n: int, k: int, seed: int):
    """BF16 matmul, cuBLAS FP8, and our engine FP8 linear for one projection."""

    from minillm_l4.engine.kernels import sm89_fp8 as fp8
    from minillm_l4.engine.kernels.sm89_fp8_validation import ModelShape, make_activations, make_weights

    weight, scales = make_weights(ModelShape("p", n, k), seed)
    x = make_activations(rows, k, seed)
    w_bf16 = weight.to(torch.bfloat16)
    x8 = x.clamp(-448, 448).to(torch.float8_e4m3fn)
    one = torch.ones((), device="cuda")
    return {
        "bf16": lambda: x @ w_bf16.t(),
        "fp8": lambda: torch._scaled_mm(x8, weight.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16),
        "fp8_ours": lambda: fp8.sm89_fp8_linear(x, weight, scales, [128, 128]),
    }


def datasheet_key(precision: str) -> str:
    return "fp8" if precision == "fp8_ours" else precision


def measure(fn, flops: float, precision: str, nvml: Nvml, args) -> dict:
    from triton.testing import do_bench

    fn()  # compile / autotune outside the timed window
    torch.cuda.synchronize()
    with Telemetry(nvml) as tele:
        ms = float(do_bench(fn, warmup=args.warmup_ms, rep=args.rep_ms, return_mode="median"))
    tel = tele.summary()
    tflops = flops / (ms * 1e9)
    peak = DATASHEET[datasheet_key(precision)]
    row = {"ms": ms, "tflops": tflops, "datasheet_tflops": peak, "pct_of_datasheet": 100 * tflops / peak, **tel}
    if tel.get("sm_mhz_median"):
        row["clock_adjusted_peak"] = peak * tel["sm_mhz_median"] / BOOST_CLOCK_MHZ
        row["pct_of_clock_adjusted"] = 100 * tflops / row["clock_adjusted_peak"]
    return row


def fmt(row: dict) -> str:
    return (f"{row['tflops']:7.1f} TF  {row['pct_of_datasheet']:5.1f}% of {row['datasheet_tflops']:5.1f}  "
            f"clk {row.get('sm_mhz_median', 0):5.0f} MHz  {row.get('watts_median', 0):4.1f} W  "
            f"{row.get('temp_c_max', 0):3d} C  {' '.join(f'{k}={v:.0%}' for k, v in row.get('limit_reasons', {}).items())}")


@torch.inference_mode()
def main(argv=None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available", file=sys.stderr)
        return 2
    from minillm_l4.engine.kernels import sm89_fp8 as fp8
    from minillm_l4.engine.kernels.sm89_fp8_validation import MODEL_SHAPES

    nvml = Nvml()
    result = {"environment": collect_environment_metadata(), "datasheet": DATASHEET,
              "boost_clock_mhz": BOOST_CLOCK_MHZ, "args": {k: str(v) for k, v in vars(args).items()},
              "idle": nvml.sample(), "square": [], "model": [], "memory": []}
    print(f"device={torch.cuda.get_device_name()} torch={torch.__version__} triton={fp8.triton.__version__} "
          f"idle={result['idle']}", flush=True)

    if "square" in args.sections:
        for n in args.sizes:
            for precision, fn in make_square_cases(n).items():
                row = {"precision": precision, "m": n, "n": n, "k": n,
                       **measure(fn, 2 * n**3, precision, nvml, args)}
                result["square"].append(row)
                print(f"[square] {precision:8s} {n:6d}  {fmt(row)}", flush=True)
            torch.cuda.empty_cache()

    if "model" in args.sections:
        for shape in MODEL_SHAPES:
            for rows in args.rows:
                for precision, fn in make_model_cases(rows, shape.n, shape.k, 20261026).items():
                    row = {"precision": precision, "shape": shape.name, "m": rows, "n": shape.n, "k": shape.k,
                           **measure(fn, 2 * rows * shape.n * shape.k, precision, nvml, args)}
                    result["model"].append(row)
                    print(f"[model] {shape.name:7s} M={rows:5d} {precision:8s} {fmt(row)}", flush=True)
            torch.cuda.empty_cache()

    if "memory" in args.sections:
        from triton.testing import do_bench
        for mib in (64, 256, 1024):
            src = torch.empty(mib * 2**20, dtype=torch.uint8, device="cuda")
            dst = torch.empty_like(src)
            with Telemetry(nvml) as tele:
                ms = float(do_bench(lambda: dst.copy_(src), warmup=args.warmup_ms, rep=args.rep_ms,
                                    return_mode="median"))
            gbps = 2 * src.numel() / (ms * 1e6)  # read + write
            row = {"bytes": src.numel(), "ms": ms, "gbps": gbps, "datasheet_gbps": DATASHEET["memory_gbps"],
                   "pct_of_datasheet": 100 * gbps / DATASHEET["memory_gbps"], **tele.summary()}
            result["memory"].append(row)
            print(f"[memory] copy {mib:5d} MiB  {gbps:6.1f} GB/s  {row['pct_of_datasheet']:5.1f}% of 300", flush=True)
            del src, dst

    result["num_alloc_retries"] = torch.cuda.memory_stats().get("num_alloc_retries", 0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "gemm_peak.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
