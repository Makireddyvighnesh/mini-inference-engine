"""Where one decode step's time goes: weights, KV cache, other kernels, launch overhead.

For batch sizes 1/2/4/8 and context lengths 128/256/512/1024 (pinned Qwen3-4B
FP8 with fused kernels and merged projections, continuous batching, no HF
reference check):

* timing    - decode step time (TPOT p50) eager and with CUDA Graphs
              (1 warm-up + 3 measured runs per case, unprofiled);
* profile   - GPU kernel time per decode step by category, from two eager
              runs under torch.profiler: (prompt + 1 + K tokens) minus
              (prompt + 1 token), divided by K, so prefill cancels out:
              - weights: FP8 projections (q/k/v, o, gate/up, down + quantizer)
              - weights: lm_head (BF16, tied to the embedding; cuBLAS)
              - KV cache: paged decode attention (reads every cached token)
              - KV cache: write of the new token's K/V
              - norm / RoPE / SiLU, and everything else (embedding,
                residual adds, sampling, copies);
* derived   - CPU/launch gap = eager step - GPU kernel time; per-launch
              cost = (eager - graph) / kernels per step; bytes each step must
              read (weights, KV cache) and their floor at the measured copy
              bandwidth.

All timing runs before the first profiler session: a finished session leaves
CUPTI attached and slows later steps.

  .conda-env/bin/python -m minillm_l4.benchmarks.commands.run_decode_breakdown \\
      --output-dir minillm_l4/results/decode_breakdown_<date>
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from minillm_l4.benchmarks.commands.run_showcase import MODEL_ID, MODEL_REVISION, load_qwen_fp8, metrics, paged, prompt
from minillm_l4.benchmarks.core.harness import BenchmarkHarness, RequestEventRecorder
from minillm_l4.benchmarks.core.schemas import HarnessConfig, RequestSpec, WorkloadSpec

BATCHES = (1, 2, 4, 8)
CONTEXTS = (128, 256, 512, 1024)
COPY_BANDWIDTH_GBPS = 231.0  # measured device copy bandwidth (docs/phase_notes/gemm_peak.md)
CATEGORIES = (
    ("weights_fp8_projections", ("_fp8_linear_kernel", "_fp8_split_gemm_kernel", "_quantize_activation_kernel")),
    ("kv_attention", ("_paged_decode",)),
    # Prefill attention: PyTorch SDPA FlashAttention / memory-efficient kernels, or our packed kernel.
    ("prefill_attention", ("flash", "fmha", "attention", "_packed_prefill_kernel")),
    ("kv_write", ("_write_decode_kv_kernel", "_write_packed_kv_kernel")),
    ("norm_rope_silu", ("_rms_norm_kernel", "_rope_kernel", "_silu_mul_kernel")),
    # cuBLAS kernels in a decode step are the BF16 lm_head GEMM/GEMV.
    ("weights_lm_head", ("gemm", "gemv", "cutlass", "xmma", "cublas")),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=list(BATCHES))
    parser.add_argument("--contexts", nargs="+", type=int, default=list(CONTEXTS))
    parser.add_argument("--decode-tokens", type=int, default=64, help="output tokens per request in timing runs")
    parser.add_argument("--profile-steps", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory is not empty: {args.output_dir}")
    return args


def category(name: str) -> str:
    for label, patterns in CATEGORIES:
        if any(p in name for p in patterns):
            return label
    return "other"


def model_bytes(model) -> dict:
    """Bytes one decode step must read for weights, and KV bytes per cached token."""

    fp8 = sum(m.weight.numel() + m.weight_scale_inv.numel() * 4 for m in model.modules()
              if getattr(getattr(m, "weight", None), "dtype", None) == torch.float8_e4m3fn)
    head = model.lm_head.weight
    config = model.config
    kv_per_token = (config.num_hidden_layers * 2 * config.num_key_value_heads
                    * getattr(config, "head_dim", config.hidden_size // config.num_attention_heads) * 2)
    return {"fp8_weight_bytes": fp8, "lm_head_bytes": head.numel() * head.element_size(),
            "kv_bytes_per_token": kv_per_token}


def kernel_profile(model, requests, steps: int) -> dict:
    """Per-category GPU kernel microseconds and kernel count for ``requests`` (eager)."""

    from torch.profiler import ProfilerActivity, profile

    runner = paged(model, requests, prefill_chunk_size=None)
    try:
        start = time.perf_counter_ns()
        recorders = tuple(RequestEventRecorder(r.request_id, run_started_ns=start) for r in requests)
        runner(requests, recorders)  # warm this exact shape outside the profiler
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        recorders = tuple(RequestEventRecorder(r.request_id, run_started_ns=start) for r in requests)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            runner(requests, recorders)
            torch.cuda.synchronize()
    finally:
        runner.close()
        del runner
        gc.collect()
        torch.cuda.empty_cache()
    totals, names, count = {}, {}, 0
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CUDA or event.device_time <= 0:
            continue
        key = category(event.name)
        totals[key] = totals.get(key, 0.0) + event.device_time  # microseconds
        names[event.name] = names.get(event.name, 0.0) + event.device_time
        count += 1
    top = dict(sorted(names.items(), key=lambda item: -item[1])[:25])
    return {"us": totals, "kernels": count, "top_kernels_us": top}


@torch.inference_mode()
def main(argv=None) -> int:
    args = parse_args(argv)
    from minillm_l4.engine.kernels.fused import install_fused_kernels
    from minillm_l4.engine.kernels.merged_projections import install_merged_projections

    bundle = load_qwen_fp8(fp8_kernel_path="sm89", local_files_only=True)
    model, tok = bundle.model, bundle.tokenizer
    if not install_fused_kernels(model):
        raise RuntimeError("fused kernels unavailable")
    install_merged_projections(model)
    harness = BenchmarkHarness(HarnessConfig(
        repetitions=args.repetitions, warmup_repetitions=1, respect_arrival_schedule=True,
        collect_gpu=False, collect_system_telemetry=False))
    sizes = model_bytes(model)
    result = {"created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": MODEL_ID,
              "revision": MODEL_REVISION, "gpu": torch.cuda.get_device_name(0), "hf_reference_check": False,
              "copy_bandwidth_gbps": COPY_BANDWIDTH_GBPS, **sizes, "cases": []}
    prompts = {c: [prompt(tok, c, f"decode-breakdown-r{i}") for i in range(max(args.batches))] for c in args.contexts}

    def requests(batch, context, new_tokens):
        return [RequestSpec(f"b{batch}-c{context}-r{i}", prompts[context][i], new_tokens) for i in range(batch)]

    # 1. Unprofiled timing, every case, before any profiler session.
    for batch in args.batches:
        for context in args.contexts:
            case = {"batch": batch, "context": context}
            for graphs in (False, True):
                reqs = requests(batch, context, args.decode_tokens)
                workload = WorkloadSpec(name=f"decode_b{batch}_c{context}", seed=17, requests=tuple(reqs),
                                        model_id=MODEL_ID, model_revision=MODEL_REVISION, dtype="fp8",
                                        device="cuda:0", arrival_pattern="closed_loop")
                runner = paged(model, reqs, prefill_chunk_size=None, cuda_graphs=graphs)
                try:
                    run = harness.run_batched(workload, batch, runner)
                finally:
                    runner.close()
                    del runner
                    gc.collect()
                    torch.cuda.empty_cache()
                case["graph_step_ms" if graphs else "eager_step_ms"] = metrics(run, reqs)["tpot_p50_ms"]
            result["cases"].append(case)
            print(f"[timing] b={batch} ctx={context:5d}  eager {case['eager_step_ms']:6.2f} ms  "
                  f"graph {case['graph_step_ms']:6.2f} ms", flush=True)

    # 2. Profiled GPU kernel time per decode step (difference of two runs).
    for case in result["cases"]:
        batch, context = case["batch"], case["context"]
        short = kernel_profile(model, requests(batch, context, 1), 0)
        long = kernel_profile(model, requests(batch, context, 1 + args.profile_steps), args.profile_steps)
        per_step = {k: (long["us"].get(k, 0.0) - short["us"].get(k, 0.0)) / args.profile_steps / 1000
                    for k in set(long["us"]) | set(short["us"])}
        gpu_ms = sum(per_step.values())
        kv_bytes = batch * (context + args.profile_steps / 2) * sizes["kv_bytes_per_token"]
        weight_bytes = sizes["fp8_weight_bytes"] + sizes["lm_head_bytes"]
        kernels = (long["kernels"] - short["kernels"]) / args.profile_steps
        case.update({
            "gpu_ms_by_category": per_step, "gpu_kernel_ms": gpu_ms, "kernels_per_step": kernels,
            "eager_launch_gap_ms": case["eager_step_ms"] - gpu_ms,
            "graph_gap_ms": case["graph_step_ms"] - gpu_ms,
            "per_launch_us": 1000 * (case["eager_step_ms"] - case["graph_step_ms"]) / kernels if kernels else None,
            "weight_bytes": weight_bytes, "kv_bytes": kv_bytes,
            "weight_floor_ms": weight_bytes / (COPY_BANDWIDTH_GBPS * 1e6),
            "kv_floor_ms": kv_bytes / (COPY_BANDWIDTH_GBPS * 1e6),
        })
        parts = "  ".join(f"{k} {v:5.2f}" for k, v in sorted(per_step.items()))
        print(f"[profile] b={batch} ctx={context:5d}  GPU {gpu_ms:6.2f} ms ({kernels:.0f} kernels)  "
              f"launch gap {case['eager_launch_gap_ms']:5.2f} ms  |  {parts}", flush=True)

    result["num_alloc_retries"] = torch.cuda.memory_stats().get("num_alloc_retries", 0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "decode_breakdown.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
