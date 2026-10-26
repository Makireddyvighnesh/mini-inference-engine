"""What happens between a request arriving and its first token (Q1).

A burst of B identical-length requests arrives at t=0 (B = 1..32, prompts
128..8192 tokens) at the recommended engine configuration: mixed batching,
adaptive chunks, CUDA Graph decode, fused kernels, merged projections, prefix
cache off, 16 output tokens. The KV pool is sized to the GPU's free memory
(not to the workload), so large bursts queue for pages as they would in
production. No HF reference check.

Each request's TTFT splits exactly into (from its recorded events):
  queue      arrival -> admission          waiting for KV pages / a slot
  wait       admission -> first chunk      admitted, waiting behind other prompts' chunks
  compute    sum of its chunk steps        forwards that carried its prompt (shared with
                                           other rows in the same step)
  gaps       first chunk -> last chunk end, minus compute: steps that ran other work
             (other prompts, decode) between its chunks
The first token is sampled from the last chunk's logits (zero extra time).

Also timed outside the engine (CPU, median of 20): tokenizing the prompt text
and detokenizing one token, the host work around the engine in a server.

  .conda-env/bin/python -m minillm_l4.benchmarks.commands.run_ttft_anatomy \\
      --output-dir minillm_l4/results/ttft_anatomy_<date>
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path

import torch

from minillm_l4.benchmarks.commands.run_showcase import MODEL_ID, MODEL_REVISION, load_qwen_fp8, prompt
from minillm_l4.benchmarks.core.harness import BenchmarkHarness
from minillm_l4.benchmarks.core.metrics import percentile
from minillm_l4.benchmarks.core.schemas import HarnessConfig, RequestSpec, WorkloadSpec

BATCHES = (1, 2, 4, 8, 16, 32)
PROMPTS = (128, 256, 512, 1024, 2048, 4096, 8192)
BLOCK = 16
OUTPUT_TOKENS = 16


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=list(BATCHES))
    parser.add_argument("--prompts", nargs="+", type=int, default=list(PROMPTS))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--headroom-gib", type=float, default=4.0,
                        help="free memory kept out of the KV pool for activations and graphs")
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory is not empty: {args.output_dir}")
    return args


def anatomy(events: list[dict]) -> dict[str, dict]:
    """Per request: TTFT components (ms) and a compact timeline, from one run's events."""

    by_request: dict[str, list[dict]] = {}
    for event in events:
        by_request.setdefault(event["request_id"], []).append(event)
    out = {}
    for request_id, rows in by_request.items():
        t = {}
        chunks, start = [], None
        for e in sorted(rows, key=lambda e: (e["timestamp_ns"], e["sequence"])):
            name, ms = e["event"], e["timestamp_ns"] / 1e6
            if name in ("arrival", "admission", "prefill_start", "first_token_ready") and name not in t:
                t[name] = ms
            elif name == "prefill_chunk_start":
                start = ms
            elif name == "prefill_chunk_end" and start is not None:
                meta = e.get("metadata") or {}
                chunks.append((start, ms, meta.get("end_token", 0) - meta.get("start_token", 0)))
                start = None
        if not {"arrival", "admission", "first_token_ready"} <= t.keys() or not chunks:
            continue
        compute = sum(end - begin for begin, end, _ in chunks)
        first, last = chunks[0][0], chunks[-1][1]
        out[request_id] = {
            "ttft": t["first_token_ready"] - t["arrival"],
            "queue": t["admission"] - t["arrival"],
            "wait": first - t["admission"],
            "compute": compute,
            "gaps": (last - first) - compute,
            "emit": t["first_token_ready"] - last,
            "chunks": len(chunks),
            "timeline": {"arrival": t["arrival"], "admission": t["admission"], "first_token": t["first_token_ready"],
                         "chunks": [[round(a, 3), round(b, 3), n] for a, b, n in chunks]},
        }
    return out


def host_costs(tokenizer, prompts: dict[int, tuple[int, ...]]) -> list[dict]:
    rows = []
    for length, ids in sorted(prompts.items()):
        text = tokenizer.decode(ids)
        encode, decode = [], []
        for _ in range(20):
            start = time.perf_counter()
            tokenizer.encode(text, add_special_tokens=False)
            encode.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            tokenizer.decode(ids[-1:])
            decode.append((time.perf_counter() - start) * 1000)
        rows.append({"tokens": length, "tokenize_ms": statistics.median(encode),
                     "detokenize_one_ms": statistics.median(decode)})
    return rows


@torch.inference_mode()
def main(argv=None) -> int:
    args = parse_args(argv)
    from minillm_l4.benchmarks.runners.chunked_prefill import ChunkedPrefillPagedRunner
    from minillm_l4.engine.kernels.fused import install_fused_kernels
    from minillm_l4.engine.kernels.merged_projections import install_merged_projections

    bundle = load_qwen_fp8(fp8_kernel_path="sm89", local_files_only=True)
    model, tok = bundle.model, bundle.tokenizer
    if not install_fused_kernels(model):
        raise RuntimeError("fused kernels unavailable")
    install_merged_projections(model)
    config = model.config
    kv_bytes_per_token = (config.num_hidden_layers * 2 * config.num_key_value_heads
                          * getattr(config, "head_dim", config.hidden_size // config.num_attention_heads) * 2)
    free, _ = torch.cuda.mem_get_info()
    budget_blocks = int((free - args.headroom_gib * 2**30) // (BLOCK * kv_bytes_per_token))
    harness = BenchmarkHarness(HarnessConfig(
        repetitions=args.repetitions, warmup_repetitions=1, respect_arrival_schedule=True,
        collect_gpu=False, collect_system_telemetry=False))
    prompts = {n: [prompt(tok, n, f"ttft-r{i}") for i in range(max(args.batches))] for n in args.prompts}
    result = {"created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": MODEL_ID,
              "revision": MODEL_REVISION, "gpu": torch.cuda.get_device_name(0), "hf_reference_check": False,
              "engine": {"mixed_batch": True, "adaptive_chunking": True, "cuda_graphs": True, "fused_kernels": True,
                         "merged_projections": True, "prefix_cache": False, "max_batch_size": max(args.batches)},
              "output_tokens": OUTPUT_TOKENS, "kv_bytes_per_token": kv_bytes_per_token,
              "kv_pool_blocks": budget_blocks, "kv_pool_tokens": budget_blocks * BLOCK,
              "host": host_costs(tok, {n: p[0] for n, p in prompts.items()}), "cases": []}
    print(f"KV pool: {budget_blocks} blocks = {budget_blocks * BLOCK:,} tokens "
          f"({budget_blocks * BLOCK * kv_bytes_per_token / 2**30:.1f} GiB)", flush=True)

    for length in args.prompts:
        for batch in args.batches:
            reqs = tuple(RequestSpec(f"r{i:02d}", prompts[length][i], OUTPUT_TOKENS, scheduled_arrival_ms=0.0)
                         for i in range(batch))
            needed = sum(math.ceil((length + OUTPUT_TOKENS) / BLOCK) for _ in reqs) + batch
            runner = ChunkedPrefillPagedRunner(
                model, block_size=BLOCK, num_blocks=min(needed, budget_blocks), max_batch_size=max(args.batches),
                max_prefill_tokens=256, device="cuda", decode_backend="auto", decode_sdpa_compat=True,
                enable_prefix=False, prefill_chunk_size=None, mixed_batch=True, adaptive_chunking=True,
                cuda_graphs=True)
            retries = torch.cuda.memory_stats().get("num_alloc_retries", 0)
            try:
                run = harness.run_trace(WorkloadSpec(
                    name=f"ttft_b{batch}_p{length}", seed=17, requests=reqs, model_id=MODEL_ID,
                    model_revision=MODEL_REVISION, dtype="fp8", device="cuda:0", arrival_pattern="fixed_rate"), runner)
                summary = runner.last_summary
            finally:
                runner.close()
                del runner
                gc.collect()
                torch.cuda.empty_cache()
            per_run = [anatomy(r["events"]) for r in run.runs]
            rows = [row for requests in per_run for row in requests.values()]
            ttft = sorted(r["ttft"] for r in rows)
            case = {
                "batch": batch, "prompt_tokens": length, "pool_limited": needed > budget_blocks,
                "requests_measured": len(rows), "failed": batch * len(run.runs) - len(rows),
                "ttft_p50_ms": statistics.median(ttft), "ttft_p95_ms": percentile(ttft, 95),
                "ttft_max_ms": max(ttft), "ttft_min_ms": min(ttft), "ttft_mean_ms": statistics.fmean(ttft),
                "mean_ms": {k: statistics.fmean(r[k] for r in rows) for k in ("queue", "wait", "compute", "gaps", "emit")},
                "chunks_per_request": statistics.fmean(r["chunks"] for r in rows),
                "peak_inflight": summary.get("maximum_inflight_requests"),
                "alloc_retries": torch.cuda.memory_stats().get("num_alloc_retries", 0) - retries,
                # Timeline of the last measured run, for the Gantt chart.
                "timeline": {rid: r["timeline"] for rid, r in sorted(per_run[-1].items())},
            }
            result["cases"].append(case)
            m = case["mean_ms"]
            print(f"[ttft] p={length:5d} b={batch:2d}  p50 {case['ttft_p50_ms']:8.1f}  p95 {case['ttft_p95_ms']:8.1f}  "
                  f"max {case['ttft_max_ms']:8.1f} ms | mean queue {m['queue']:7.1f} wait {m['wait']:7.1f} "
                  f"compute {m['compute']:7.1f} gaps {m['gaps']:7.1f} | chunks {case['chunks_per_request']:.1f} "
                  f"inflight {case['peak_inflight']} retries {case['alloc_retries']}", flush=True)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "ttft_anatomy.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {args.output_dir / 'ttft_anatomy.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
