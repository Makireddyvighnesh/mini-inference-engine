"""Anatomy of one long request: TTFT, TPOT, ITL, and where every millisecond goes.

One request (default: 4,096-token prompt, 2,048 new tokens) on the pinned
Qwen3-4B FP8 with fused kernels and merged projections, no HF reference check.

1. timing (unprofiled, 1 warm-up + N measured runs per config):
   * recommended - adaptive mixed prefill + CUDA Graph decode (engine default);
   * whole_graph - whole prompt in one forward + CUDA Graph decode;
   * whole_eager - whole prompt in one forward + eager decode (launch cost visible).
   Reports TTFT, its parts from request events (arrival -> admission -> prefill
   chunks -> first token), TPOT, the ITL distribution, and ITL per context bucket
   (decode slows as the KV cache grows from 4,096 to 6,144 tokens).
2. profile (torch.profiler, after all timing; CUPTI slows later steps):
   * prefill: GPU kernel time by category for the prompt (max_new_tokens=1),
     against the measured wall time; the rest is CPU work / GPU idle;
   * decode: per-step GPU kernel time by category at contexts 4,096 / 5,120 /
     6,144 from two runs ((1 + K) tokens minus 1 token, divided by K), eager and
     CUDA Graph, plus GPU idle time per step (step time minus kernel time).
   Categories: FP8 weight GEMMs, lm_head, KV-cache read (decode attention),
   prefill attention, KV write, norm/RoPE/SiLU, other (embedding, sampling,
   copies). Bytes each decode step must read and their floor at 231 GB/s.

  .conda-env/bin/python -m minillm_l4.benchmarks.commands.run_request_anatomy \\
      --output-dir minillm_l4/results/request_anatomy_<date>
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch

from minillm_l4.benchmarks.commands.run_decode_breakdown import COPY_BANDWIDTH_GBPS, category, model_bytes
from minillm_l4.benchmarks.commands.run_showcase import MODEL_ID, MODEL_REVISION, load_qwen_fp8, paged, prompt
from minillm_l4.benchmarks.core.harness import RequestEventRecorder
from minillm_l4.benchmarks.core.schemas import RequestSpec

CONFIGS = {
    "recommended": {"mixed_batch": True, "adaptive_chunking": True, "cuda_graphs": True},
    "whole_graph": {"prefill_chunk_size": None, "cuda_graphs": True},
    "whole_eager": {"prefill_chunk_size": None, "cuda_graphs": False},
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--new-tokens", type=int, default=2048)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--profile-steps", type=int, default=16)
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory is not empty: {args.output_dir}")
    return args


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q / 100 * len(xs)))]


def run_once(runner, request):
    start = time.perf_counter_ns()
    recorder = RequestEventRecorder(request.request_id, run_started_ns=start)
    outcome = runner([request], (recorder,))[0]
    if outcome.status != "completed":
        raise RuntimeError(f"request did not complete: {outcome.status}")
    events = {}
    for e in recorder.events:
        events.setdefault(e.event, []).append(e.timestamp_ns / 1e6)
    return events, runner.last_summary


def timing(model, request, options, repetitions):
    runner = paged(model, [request], max_prefill_tokens=request.prompt_tokens, **options)
    runs = []
    try:
        run_once(runner, request)  # warm-up: graph capture, Triton specializations
        torch.cuda.synchronize()
        for _ in range(repetitions):
            events, summary = run_once(runner, request)
            tokens = events["token_ready"]
            itl = [b - a for a, b in zip(tokens, tokens[1:])]
            chunks = [(s, e) for s, e in zip(events.get("prefill_chunk_start", []), events.get("prefill_chunk_end", []))]
            prefill_start = events.get("prefill_start", [tokens[0]])[0]
            runs.append({
                "ttft_ms": tokens[0],  # arrival is the run start (0)
                "ttft_parts_ms": {
                    "admission_and_kv_page_allocation": events.get("admission", [0.0])[0],
                    "admission_to_prefill_start": prefill_start - events.get("admission", [0.0])[0],
                    "prefill_chunks_gpu_steps": sum(e - s for s, e in chunks),
                    "between_chunks": (chunks[-1][1] - chunks[0][0] - sum(e - s for s, e in chunks)) if chunks else 0.0,
                    "prefill_end_to_first_token": tokens[0] - (chunks[-1][1] if chunks else prefill_start),
                },
                "prefill_chunks": [round(e - s, 2) for s, e in chunks],
                "prefill_chunk_tokens": [r["computed_tokens"] for r in summary.get("prefill_records", [])],
                "tpot_ms": (tokens[-1] - tokens[0]) / (len(tokens) - 1),
                "itl_ms": itl,
                "e2e_ms": events["completion"][0] if "completion" in events else tokens[-1],
                "decode_mode": summary.get("decode_mode_used"),
                "graph_replays": summary.get("graph_replays"),
            })
    finally:
        runner.close()
        del runner
        gc.collect()
        torch.cuda.empty_cache()
    return runs


def summarize_timing(runs, prompt_tokens, bucket=256):
    itl = [x for r in runs for x in r["itl_ms"]]
    best = sorted(runs, key=lambda r: r["ttft_ms"])[len(runs) // 2]
    buckets = {}
    for r in runs:
        for i, x in enumerate(r["itl_ms"]):
            context = prompt_tokens + 1 + i  # tokens in the KV cache when this step ran
            buckets.setdefault(context // bucket * bucket, []).append(x)
    return {
        "ttft_ms": statistics.median(r["ttft_ms"] for r in runs),
        "ttft_parts_ms": best["ttft_parts_ms"], "prefill_chunks_ms": best["prefill_chunks"],
        "prefill_chunk_tokens": best["prefill_chunk_tokens"],
        "tpot_ms": statistics.median(r["tpot_ms"] for r in runs),
        "itl_p50_ms": pct(itl, 50), "itl_p90_ms": pct(itl, 90), "itl_p99_ms": pct(itl, 99),
        "itl_max_ms": max(itl), "itl_min_ms": min(itl),
        "itl_over_2x_median": sum(x > 2 * pct(itl, 50) for x in itl) / len(runs),
        "e2e_ms": statistics.median(r["e2e_ms"] for r in runs),
        "decode_tokens_per_s": 1000 / statistics.median(r["tpot_ms"] for r in runs),
        "itl_p50_by_context": {str(k): statistics.median(v) for k, v in sorted(buckets.items())},
        "itl_last_run_ms": runs[-1]["itl_ms"],
        "decode_mode": best["decode_mode"], "graph_replays": best["graph_replays"],
    }


def profiled(model, request, options):
    """GPU kernel time by category (ms), kernel count, busy span, and top kernels for one run."""

    from torch.profiler import ProfilerActivity, profile

    runner = paged(model, [request], max_prefill_tokens=request.prompt_tokens, **options)
    try:
        run_once(runner, request)  # warm this exact shape (and capture graphs) outside the profiler
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            wall = time.perf_counter()
            run_once(runner, request)
            torch.cuda.synchronize()
            wall = (time.perf_counter() - wall) * 1000
    finally:
        runner.close()
        del runner
        gc.collect()
        torch.cuda.empty_cache()
    totals, names, count, starts, ends = {}, {}, 0, [], []
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CUDA or event.device_time <= 0:
            continue
        key = category(event.name)
        totals[key] = totals.get(key, 0.0) + event.device_time / 1000
        names[event.name] = names.get(event.name, 0.0) + event.device_time / 1000
        starts.append(event.time_range.start)
        ends.append(event.time_range.end)
        count += 1
    return {"ms": totals, "kernels": count, "wall_ms": wall, "gpu_ms": sum(totals.values()),
            "top_kernels_ms": dict(sorted(names.items(), key=lambda kv: -kv[1])[:20])}


def decode_profile(model, tok, context, steps, graphs):
    base = prompt(tok, context, f"anatomy-ctx{context}")
    options = {"prefill_chunk_size": None, "cuda_graphs": graphs}
    short = profiled(model, RequestSpec(f"p{context}a", base, 1), options)
    long = profiled(model, RequestSpec(f"p{context}b", base, 1 + steps), options)
    per_step = {k: (long["ms"].get(k, 0.0) - short["ms"].get(k, 0.0)) / steps for k in set(long["ms"]) | set(short["ms"])}
    gpu = sum(per_step.values())
    step_wall = (long["wall_ms"] - short["wall_ms"]) / steps
    return {"context": context, "graphs": graphs, "gpu_ms_by_category": per_step, "gpu_kernel_ms": gpu,
            "kernels_per_step": (long["kernels"] - short["kernels"]) / steps,
            "profiled_step_ms": step_wall, "gpu_idle_ms": step_wall - gpu,
            "top_kernels_ms": long["top_kernels_ms"]}


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
    sizes = model_bytes(model)
    result = {"created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": MODEL_ID,
              "revision": MODEL_REVISION, "gpu": torch.cuda.get_device_name(0), "hf_reference_check": False,
              "prompt_tokens": args.prompt_tokens, "new_tokens": args.new_tokens,
              "copy_bandwidth_gbps": COPY_BANDWIDTH_GBPS, **sizes, "timing": {}, "profile": {}}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    def save():
        (out / "request_anatomy.json").write_text(json.dumps(result, indent=1) + "\n")

    request = RequestSpec("long", prompt(tok, args.prompt_tokens, "anatomy"), args.new_tokens)

    # 1. Unprofiled timing.
    for name in args.configs:
        s = summarize_timing(timing(model, request, CONFIGS[name], args.repetitions), args.prompt_tokens)
        result["timing"][name] = s
        save()
        print(f"[timing] {name:12s} TTFT {s['ttft_ms']:7.1f} ms  TPOT {s['tpot_ms']:6.2f} ms  "
              f"ITL p50/p99/max {s['itl_p50_ms']:.2f}/{s['itl_p99_ms']:.2f}/{s['itl_max_ms']:.2f} ms  "
              f"chunks {s['prefill_chunk_tokens']} {s['prefill_chunks_ms']}  parts {s['ttft_parts_ms']}", flush=True)

    # 2. Profiles.
    for name in ("recommended", "whole_graph"):
        p = profiled(model, RequestSpec(f"prefill-{name}", request.prompt_token_ids, 1), CONFIGS[name])
        result["profile"][f"prefill_{name}"] = p
        save()
        print(f"[prefill ] {name:12s} wall {p['wall_ms']:7.1f} ms  GPU {p['gpu_ms']:7.1f} ms  "
              + "  ".join(f"{k} {v:.1f}" for k, v in sorted(p["ms"].items(), key=lambda kv: -kv[1])), flush=True)
    contexts = (args.prompt_tokens, args.prompt_tokens + args.new_tokens // 2, args.prompt_tokens + args.new_tokens)
    decode = []
    for graphs in (False, True):
        for context in contexts:
            d = decode_profile(model, tok, context, args.profile_steps, graphs)
            kv_bytes = context * sizes["kv_bytes_per_token"]
            d.update({"kv_bytes": kv_bytes, "kv_floor_ms": kv_bytes / (COPY_BANDWIDTH_GBPS * 1e6),
                      "weight_floor_ms": (sizes["fp8_weight_bytes"] + sizes["lm_head_bytes"]) / (COPY_BANDWIDTH_GBPS * 1e6)})
            decode.append(d)
            result["profile"]["decode"] = decode
            save()
            print(f"[decode  ] {'graph' if graphs else 'eager'} ctx={context:5d} step {d['profiled_step_ms']:6.2f} ms  "
                  f"GPU {d['gpu_kernel_ms']:6.2f} ms ({d['kernels_per_step']:.0f} kernels)  idle {d['gpu_idle_ms']:5.2f}  |  "
                  + "  ".join(f"{k} {v:.2f}" for k, v in sorted(d["gpu_ms_by_category"].items(), key=lambda kv: -kv[1])),
                  flush=True)
    result["num_alloc_retries"] = torch.cuda.memory_stats().get("num_alloc_retries", 0)
    save()
    print(f"wrote {out / 'request_anatomy.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
