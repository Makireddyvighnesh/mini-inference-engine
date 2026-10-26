"""Prefill and decode steps on the L4 roofline: compute-bound or memory-bound?

For whole-prompt prefill (one request, 128..8192 tokens) and continuous decode
(batch 1/2/4/8 at context 512), with fused kernels and merged projections
(pinned Qwen3-4B FP8, no HF reference check):

* timing  - prefill TTFT p50 and decode step (TPOT p50, CUDA Graphs),
            1 warm-up + 3 measured runs, all before any profiler session;
* profile - GPU kernel time per category (``run_decode_breakdown``
            categories): a prompt + 1 token run is one prefill; decode is the
            difference of two runs divided by the extra steps;
* model   - the dimensions needed to count FLOPs and bytes per category
            exactly (written to the JSON; the chart script does the math).

  .conda-env/bin/python -m minillm_l4.benchmarks.commands.run_roofline \\
      --output-dir minillm_l4/results/roofline_<date>
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from minillm_l4.benchmarks.commands.run_decode_breakdown import kernel_profile
from minillm_l4.benchmarks.commands.run_showcase import MODEL_ID, MODEL_REVISION, load_qwen_fp8, metrics, paged, prompt
from minillm_l4.benchmarks.core.harness import BenchmarkHarness
from minillm_l4.benchmarks.core.schemas import HarnessConfig, RequestSpec, WorkloadSpec

PREFILL_TOKENS = (128, 256, 512, 1024, 2048, 4096, 8192)
DECODE_BATCHES = (1, 2, 4, 8)
DECODE_CONTEXT = 512


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefill-tokens", nargs="+", type=int, default=list(PREFILL_TOKENS))
    parser.add_argument("--decode-batches", nargs="+", type=int, default=list(DECODE_BATCHES))
    parser.add_argument("--decode-context", type=int, default=DECODE_CONTEXT)
    parser.add_argument("--profile-steps", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory is not empty: {args.output_dir}")
    return args


def model_dims(model) -> dict:
    config = model.config
    linears = [m for m in model.modules()
               if getattr(getattr(m, "weight", None), "dtype", None) == torch.float8_e4m3fn]
    return {
        "layers": config.num_hidden_layers, "hidden": config.hidden_size, "vocab": config.vocab_size,
        "q_heads": config.num_attention_heads, "kv_heads": config.num_key_value_heads,
        "head_dim": getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
        "intermediate": config.intermediate_size,
        # FP8 projection weights: parameter count, scale bytes, and sum over linears of (K + N)
        # (activation bytes in and out per row are 2 * (K + N) in BF16).
        "fp8_params": sum(m.weight.numel() for m in linears),
        "fp8_scale_bytes": sum(m.weight_scale_inv.numel() * m.weight_scale_inv.element_size() for m in linears),
        "fp8_in_plus_out": sum(m.weight.shape[0] + m.weight.shape[1] for m in linears),
        "lm_head_bytes": model.lm_head.weight.numel() * model.lm_head.weight.element_size(),
    }


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
    result = {"created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": MODEL_ID,
              "revision": MODEL_REVISION, "gpu": torch.cuda.get_device_name(0), "hf_reference_check": False,
              "decode_context": args.decode_context, "profile_steps": args.profile_steps,
              "model_dims": model_dims(model), "prefill": [], "decode": []}
    decode_prompts = [prompt(tok, args.decode_context, f"roofline-r{i}") for i in range(max(args.decode_batches))]

    def timed(name, reqs, batch, **options):
        workload = WorkloadSpec(name=name, seed=17, requests=tuple(reqs), model_id=MODEL_ID,
                                model_revision=MODEL_REVISION, dtype="fp8", device="cuda:0",
                                arrival_pattern="closed_loop")
        runner = paged(model, reqs, prefill_chunk_size=None, **options)
        try:
            return metrics(harness.run_batched(workload, batch, runner), reqs)
        finally:
            runner.close()
            del runner
            gc.collect()
            torch.cuda.empty_cache()

    # 1. Unprofiled timing (a finished profiler session slows later steps).
    for tokens in args.prefill_tokens:
        reqs = [RequestSpec(f"prefill-{tokens}", prompt(tok, tokens, "roofline-prefill"), 1)]
        result["prefill"].append({"tokens": tokens, "ttft_ms": timed(f"prefill_{tokens}", reqs, 1)["ttft_p50_ms"]})
        print(f"[timing] prefill {tokens:5d} tokens: TTFT {result['prefill'][-1]['ttft_ms']:8.1f} ms", flush=True)
    for batch in args.decode_batches:
        reqs = [RequestSpec(f"decode-b{batch}-r{i}", decode_prompts[i], 64) for i in range(batch)]
        step = timed(f"decode_b{batch}", reqs, batch, cuda_graphs=True)["tpot_p50_ms"]
        result["decode"].append({"batch": batch, "context": args.decode_context, "graph_step_ms": step})
        print(f"[timing] decode batch {batch}: graph step {step:6.2f} ms", flush=True)

    # 2. GPU kernel time per category.
    for case in result["prefill"]:
        reqs = [RequestSpec(f"prefill-{case['tokens']}", prompt(tok, case["tokens"], "roofline-prefill"), 1)]
        profile = kernel_profile(model, reqs, 0)
        case["gpu_ms_by_category"] = {k: v / 1000 for k, v in profile["us"].items()}
        case["kernels"] = profile["kernels"]
        case["top_kernels_ms"] = {k: v / 1000 for k, v in profile["top_kernels_us"].items()}
        print(f"[profile] prefill {case['tokens']:5d}: GPU {sum(case['gpu_ms_by_category'].values()):8.1f} ms  "
              + "  ".join(f"{k} {v:.1f}" for k, v in sorted(case["gpu_ms_by_category"].items())), flush=True)
    for case in result["decode"]:
        batch = case["batch"]
        short = kernel_profile(model, [RequestSpec(f"d{batch}-{i}", decode_prompts[i], 1) for i in range(batch)], 0)
        long = kernel_profile(model, [RequestSpec(f"d{batch}-{i}", decode_prompts[i], 1 + args.profile_steps)
                                      for i in range(batch)], args.profile_steps)
        case["gpu_ms_by_category"] = {k: (long["us"].get(k, 0.0) - short["us"].get(k, 0.0)) / args.profile_steps / 1000
                                      for k in set(long["us"]) | set(short["us"])}
        print(f"[profile] decode batch {batch}: GPU {sum(case['gpu_ms_by_category'].values()):6.2f} ms  "
              + "  ".join(f"{k} {v:.2f}" for k, v in sorted(case["gpu_ms_by_category"].items())), flush=True)

    result["num_alloc_retries"] = torch.cuda.memory_stats().get("num_alloc_retries", 0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "roofline.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
