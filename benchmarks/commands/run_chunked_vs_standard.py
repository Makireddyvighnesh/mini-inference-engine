"""Chunked prefill vs standard prefill: what happens to users already decoding?

Eight chat requests (256-token prompts, 256 new tokens) start at t=0. While
they decode, one long prompt arrives (2k, 8k or 16k tokens, 32 new tokens).
Three policies run the same trace (CUDA Graph decode, fused kernels, merged
projections, no prefix cache, no HF reference check):

* standard       - whole-prompt prefill: the long prompt is one forward, and
                   every chat user waits for it;
* chunked_512    - mixed batching: each step carries every decode token plus
                   at most 512 prompt tokens;
* adaptive       - mixed batching with chunk sizes from the step planner
                   (150 ms step target while others decode); the engine's
                   recommended default.

Recorded per run: every token's time for every request (the chat users'
gaps between tokens show the stall), the long prompt's TTFT, and when the
last request finished. 1 warm-up + 3 measured runs per case on one runner (so
CUDA graphs are captured before timing); the summary is the median over runs
and the timeline is from the last run.

``--workload mixed`` replaces that trace with mixed real-world traffic: 40
requests with Poisson arrivals (1.5 per second), prompts drawn 60% short
(128-1024 tokens), 30% medium (1k-4k), 10% long (8k-16k), and 64-256 new
tokens each; the KV pool is sized to free memory minus a headroom, so requests
queue when it fills. Default: 1 warm-up + 1 measured run per policy.

  .conda-env/bin/python -m minillm_l4.benchmarks.commands.run_chunked_vs_standard \\
      --output-dir minillm_l4/results/chunked_vs_standard_<date>
  .conda-env/bin/python -m minillm_l4.benchmarks.commands.run_chunked_vs_standard --workload mixed \\
      --output-dir minillm_l4/results/chunked_vs_standard_mixed_<date>
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch

from minillm_l4.benchmarks.commands.run_showcase import MODEL_ID, MODEL_REVISION, load_qwen_fp8
from minillm_l4.benchmarks.core.harness import RequestEventRecorder
from minillm_l4.benchmarks.core.schemas import RequestSpec
from minillm_l4.benchmarks.core.synthetic import SyntheticSample, exact_token_ids

CHAT_USERS, CHAT_PROMPT, CHAT_OUTPUT = 8, 256, 256
LONG_OUTPUT, LONG_ARRIVAL_MS = 32, 1500.0
LONG_PROMPTS = (2048, 8192, 16384)
BLOCK = 16
# Mixed traffic: (share, min prompt, max prompt) per prompt class.
MIXED_CLASSES = (("short", 0.6, 128, 1024), ("medium", 0.3, 1024, 4096), ("long", 0.1, 8192, 16384))
POLICIES = {
    "standard": dict(prefill_chunk_size=None),
    "chunked_512": dict(prefill_chunk_size=None, mixed_batch=True, max_prefill_tokens=512),
    "adaptive": dict(prefill_chunk_size=None, mixed_batch=True, adaptive_chunking=True),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--long-prompts", nargs="+", type=int, default=list(LONG_PROMPTS))
    parser.add_argument("--repetitions", type=int, default=None,
                        help="measured runs per case (default 3; 1 for --workload mixed)")
    parser.add_argument("--workload", choices=("long_prompt", "mixed"), default="long_prompt")
    parser.add_argument("--requests", type=int, default=40, help="mixed: number of requests")
    parser.add_argument("--rate", type=float, default=1.5, help="mixed: mean arrivals per second")
    parser.add_argument("--seed", type=int, default=7, help="mixed: trace seed")
    parser.add_argument("--headroom-gib", type=float, default=4.0, help="mixed: GPU memory left out of the KV pool")
    args = parser.parse_args(argv)
    if args.repetitions is None:
        args.repetitions = 1 if args.workload == "mixed" else 3
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory is not empty: {args.output_dir}")
    return args


def trace(tok, long_prompt):
    def ids(sample_id, n, out):
        return tuple(exact_token_ids(tok, SyntheticSample(
            sample_id=sample_id, category="chunked_vs_standard", target_prompt_tokens=n, target_output_tokens=out,
            seed_text="Explain how prefill, cached decoding, and batching affect inference.")))

    chat = [RequestSpec(f"chat-{i}", ids(f"chat-{i}", CHAT_PROMPT, CHAT_OUTPUT), CHAT_OUTPUT, scheduled_arrival_ms=0.0)
            for i in range(CHAT_USERS)]
    long = RequestSpec("long", ids(f"long-{long_prompt}", long_prompt, LONG_OUTPUT), LONG_OUTPUT,
                       scheduled_arrival_ms=LONG_ARRIVAL_MS)
    return (*chat, long)


def mixed_trace(tok, count, rate, seed):
    """Poisson arrivals, prompt lengths from MIXED_CLASSES, 64-256 new tokens."""

    rng = random.Random(seed)
    requests, classes, now = [], {}, 0.0
    for i in range(count):
        now += rng.expovariate(rate) * 1000
        name, _, low, high = rng.choices(MIXED_CLASSES, weights=[c[1] for c in MIXED_CLASSES])[0]
        prompt = rng.randrange(low, high + 1, 16)
        output = rng.randrange(64, 257)
        rid = f"{name}-{i}"
        requests.append(RequestSpec(rid, tuple(exact_token_ids(tok, SyntheticSample(
            sample_id=f"mixed-{seed}-{i}", category="chunked_vs_standard", target_prompt_tokens=prompt,
            target_output_tokens=output,
            seed_text="Explain how prefill, cached decoding, and batching affect inference."))), output,
            scheduled_arrival_ms=round(now, 1)))
        classes[rid] = name
    return tuple(requests), classes


def make_runner(model, requests, options, max_blocks=None):
    from minillm_l4.engine.serving.chunked_prefill import ChunkedPrefillPagedRunner

    blocks = sum(math.ceil((r.prompt_tokens + r.max_new_tokens) / BLOCK) for r in requests) + len(requests)
    if max_blocks is not None:
        blocks = min(blocks, max_blocks)
    options = {"max_prefill_tokens": sum(r.prompt_tokens for r in requests), **options}
    return ChunkedPrefillPagedRunner(model, block_size=BLOCK, num_blocks=blocks, max_batch_size=len(requests),
                                     device="cuda", decode_backend="auto", enable_prefix=False, cuda_graphs=True,
                                     **options)


def run_once(runner, requests):
    start = time.perf_counter_ns()
    recorders = tuple(RequestEventRecorder(r.request_id, run_started_ns=start) for r in requests)
    outcomes = runner(requests, recorders)
    if any(o.status != "completed" for o in outcomes):
        raise RuntimeError(f"incomplete requests: {[o.status for o in outcomes]}")
    chunks = [r["computed_tokens"] for r in runner.last_summary.get("prefill_records", []) if r["request_id"] == "long"]
    # Event timestamps are relative to the run start.
    tokens = {rec.request_id: sorted(e.timestamp_ns / 1e6 for e in rec.events if e.event == "token_ready")
              for rec in recorders}
    return tokens, chunks


def summarize(tokens, long_prompt):
    gaps = [b - a for rid, times in tokens.items() if rid.startswith("chat") for a, b in zip(times, times[1:])]
    gaps.sort()
    long_first = tokens["long"][0]
    # Chat gaps that overlap the long prompt's prefill (arrival .. its first token).
    during = [b - a for rid, times in tokens.items() if rid.startswith("chat")
              for a, b in zip(times, times[1:]) if b > LONG_ARRIVAL_MS and a < long_first]
    return {
        "long_prompt": long_prompt,
        "long_ttft_ms": long_first - LONG_ARRIVAL_MS,
        "chat_gap_p50_ms": gaps[len(gaps) // 2],
        "chat_gap_p99_ms": gaps[min(len(gaps) - 1, int(len(gaps) * 0.99))],
        "chat_gap_max_ms": gaps[-1],
        "chat_gap_mean_during_long_ms": statistics.mean(during) if during else 0.0,
        "chat_tokens_during_long": len(during),
        "chat_finish_ms": max(times[-1] for rid, times in tokens.items() if rid.startswith("chat")),
        "all_finish_ms": max(times[-1] for times in tokens.values()),
    }


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * q))] if values else 0.0


def summarize_mixed(tokens, requests, classes):
    arrival = {r.request_id: r.scheduled_arrival_ms for r in requests}
    ttft = {rid: times[0] - arrival[rid] for rid, times in tokens.items()}
    gaps = [b - a for times in tokens.values() for a, b in zip(times, times[1:])]
    worst = {rid: max((b - a for a, b in zip(times, times[1:])), default=0.0) for rid, times in tokens.items()}
    start = min(arrival.values())
    end = max(times[-1] for times in tokens.values())
    out = {"ttft_p50_ms": pct(ttft.values(), 0.5), "ttft_p90_ms": pct(ttft.values(), 0.9),
           "ttft_max_ms": max(ttft.values()),
           "gap_p50_ms": pct(gaps, 0.5), "gap_p90_ms": pct(gaps, 0.9), "gap_p99_ms": pct(gaps, 0.99),
           "gap_max_ms": max(gaps), "gaps_over_200ms": sum(g > 200 for g in gaps),
           "gaps_over_500ms": sum(g > 500 for g in gaps), "gaps_over_1s": sum(g > 1000 for g in gaps),
           "gap_count": len(gaps), "worst_gap_p50_ms": pct(worst.values(), 0.5),
           "output_tokens": sum(len(t) for t in tokens.values()), "makespan_ms": end - start,
           "throughput_tok_s": sum(len(t) for t in tokens.values()) / ((end - start) / 1000)}
    for name, *_ in MIXED_CLASSES:
        picked = [ttft[rid] for rid in ttft if classes[rid] == name]
        out[f"ttft_{name}_p50_ms"] = pct(picked, 0.5)
        out[f"ttft_{name}_p90_ms"] = pct(picked, 0.9)
    return out


def main_mixed(args, bundle, result):
    config = bundle.model.config
    kv_bytes_per_token = (config.num_hidden_layers * 2 * config.num_key_value_heads
                          * getattr(config, "head_dim", config.hidden_size // config.num_attention_heads) * 2)
    free, _ = torch.cuda.mem_get_info()
    max_blocks = int((free - args.headroom_gib * 2**30) // (BLOCK * kv_bytes_per_token))
    requests, classes = mixed_trace(bundle.tokenizer, args.requests, args.rate, args.seed)
    result.update({"workload": "mixed", "rate_per_s": args.rate, "seed": args.seed, "kv_pool_blocks": max_blocks,
                   "kv_pool_tokens": max_blocks * BLOCK,
                   "requests": [{"id": r.request_id, "class": classes[r.request_id], "prompt_tokens": r.prompt_tokens,
                                 "output_tokens": r.max_new_tokens, "arrival_ms": r.scheduled_arrival_ms}
                                for r in requests]})
    print(f"mixed trace: {len(requests)} requests over {requests[-1].scheduled_arrival_ms / 1000:.1f} s, "
          f"{sum(r.prompt_tokens for r in requests)} prompt tokens, KV pool {max_blocks * BLOCK} tokens", flush=True)
    for policy, options in POLICIES.items():
        runner = make_runner(bundle.model, requests, options, max_blocks=max_blocks)
        try:
            run_once(runner, requests)  # warm-up: graph capture, kernel compile, planner fit
            runs = [run_once(runner, requests) for _ in range(args.repetitions)]
        finally:
            runner.close()
            del runner
            gc.collect()
            torch.cuda.empty_cache()
        summaries = [summarize_mixed(tokens, requests, classes) for tokens, _ in runs]
        case = {"policy": policy, **{key: statistics.median(s[key] for s in summaries) for key in summaries[0]},
                "timeline_last_run": runs[-1][0]}
        result["cases"].append(case)
        print(f"{policy:12s} TTFT p50 {case['ttft_p50_ms']:7.0f} p90 {case['ttft_p90_ms']:7.0f} ms | gap p50 "
              f"{case['gap_p50_ms']:5.1f} p99 {case['gap_p99_ms']:7.1f} max {case['gap_max_ms']:7.1f} ms | "
              f"gaps >500 ms {case['gaps_over_500ms']} | {case['throughput_tok_s']:5.0f} tok/s", flush=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "chunked_vs_standard.json").write_text(json.dumps(result, indent=1) + "\n")


@torch.inference_mode()
def main(argv=None) -> int:
    args = parse_args(argv)
    from minillm_l4.engine.kernels.fused import install_fused_kernels
    from minillm_l4.engine.kernels.merged_projections import install_merged_projections

    bundle = load_qwen_fp8(fp8_kernel_path="sm89", local_files_only=True)
    if not install_fused_kernels(bundle.model):
        raise RuntimeError("fused kernels unavailable")
    install_merged_projections(bundle.model)
    result = {"created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": MODEL_ID,
              "revision": MODEL_REVISION, "gpu": torch.cuda.get_device_name(0), "hf_reference_check": False,
              "chat_users": CHAT_USERS, "chat_prompt_tokens": CHAT_PROMPT, "chat_output_tokens": CHAT_OUTPUT,
              "long_output_tokens": LONG_OUTPUT, "long_arrival_ms": LONG_ARRIVAL_MS, "cases": []}
    if args.workload == "mixed":
        main_mixed(args, bundle, result)
        print(f"wrote {args.output_dir / 'chunked_vs_standard.json'}")
        return 0
    for long_prompt in args.long_prompts:
        requests = trace(bundle.tokenizer, long_prompt)
        for policy, options in POLICIES.items():
            # One runner per case: the warm-up run captures every CUDA graph batch bucket (8, 9 and 1
            # requests occur) and fits the planner, so no capture lands inside a measured run.
            runner = make_runner(bundle.model, requests, options)
            try:
                run_once(runner, requests)
                runs = [run_once(runner, requests) for _ in range(args.repetitions)]
            finally:
                runner.close()
                del runner
                gc.collect()
                torch.cuda.empty_cache()
            summaries = [summarize(tokens, long_prompt) for tokens, _ in runs]
            case = {"policy": policy, "long_prompt": long_prompt,
                    **{key: statistics.median(s[key] for s in summaries) for key in summaries[0]},
                    "long_chunks_last_run": runs[-1][1], "timeline_last_run": runs[-1][0]}
            result["cases"].append(case)
            print(f"[{long_prompt:6d}] {policy:12s} long TTFT {case['long_ttft_ms']:8.1f} ms | chat gap p50 "
                  f"{case['chat_gap_p50_ms']:6.1f} p99 {case['chat_gap_p99_ms']:7.1f} max {case['chat_gap_max_ms']:7.1f} ms"
                  f" | chat done {case['chat_finish_ms'] / 1000:5.2f} s | chunks {len(case['long_chunks_last_run'])}",
                  flush=True)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "chunked_vs_standard.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {args.output_dir / 'chunked_vs_standard.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
