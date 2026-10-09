"""One-session showcase: every engine stage on matched workloads.

Sections (all on the pinned Qwen3-4B FP8 model, one L4, same session):

1. single   - one request (512-token prompt, 128 outputs) through each decode
              engine: HF generate, manual loop, no KV cache, explicit KV cache,
              paged KV + Triton decode, paged KV + CUDA Graph decode.
2. batch    - decode throughput vs batch size (1/4/8/16) for the main engines.
3. serving  - 16 requests (128-1024-token prompts) arriving every 150 ms, all
              policies allowed 16 requests at once:
              static batching vs continuous batching (dense, paged) vs mixed
              batching with adaptive chunks.
4. longmix  - long prompts (6144) arriving among short ones while others
              decode: whole-prompt prefill vs budgeted vs adaptive chunks.
5. prefix   - requests sharing a 2048-token system prompt, prefix cache off/on.
6. prefill  - one-prompt TTFT for 128..8192 tokens.

No Hugging Face reference check is run.  Results: one JSON per case,
``showcase.json`` (headline metrics), and ``report.md``.

  .conda-env/bin/python -m minillm_l4.benchmarks.commands.run_showcase \
      --output-dir minillm_l4/results/showcase_20261008
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

from minillm_l4.benchmarks.core.harness import BenchmarkHarness
from minillm_l4.benchmarks.core.metrics import percentile
from minillm_l4.benchmarks.core.schemas import HarnessConfig, RequestSpec, WorkloadSpec
from minillm_l4.benchmarks.core.synthetic import SyntheticSample, exact_token_ids
from minillm_l4.benchmarks.commands.run_vllm_compare import (
    MODEL_ID, MODEL_REVISION, case_validity, exit_if_invalid, print_invalid_case,
)

SECTIONS = ("single", "batch", "serving", "longmix", "prefix", "prefill")
SEED_TEXT = "Explain how prefill, cached decoding, and batching affect language-model inference."


def load_qwen_fp8(**options):
    # Metrics and validity tooling must remain usable without engine imports.
    from minillm_l4.benchmarks.runners.huggingface_baseline import load_qwen_fp8 as load
    return load(**options)


def prompt(tokenizer, length: int, tag: str) -> tuple[int, ...]:
    return tuple(exact_token_ids(tokenizer, SyntheticSample(
        sample_id=f"showcase-{tag}-{length}", category="showcase", target_prompt_tokens=length,
        target_output_tokens=128, seed_text=SEED_TEXT)))


def pages(requests, block=16) -> int:
    return sum(math.ceil((r.prompt_tokens + r.max_new_tokens) / block) for r in requests) + len(requests)


def paged(model, requests, **options):
    from minillm_l4.benchmarks.runners.chunked_prefill import ChunkedPrefillPagedRunner

    return ChunkedPrefillPagedRunner(
        model, block_size=16, num_blocks=pages(requests), max_batch_size=len(requests), device="cuda",
        decode_backend="auto", decode_sdpa_compat=True,
        **{"enable_prefix": False, "max_prefill_tokens": 256, **options})


def metrics(result, requests, *, expected_runs=None):
    rows = [record["metrics"] for run in result.runs for record in run["requests"]]
    # Worst pause per measured run, then the median across runs (a single max is noisy).
    run_worst = [max((max(r["metrics"]["itl_ms"]) for r in run["requests"] if r["metrics"].get("itl_ms")), default=None)
                 for run in result.runs]
    run_worst = [w for w in run_worst if w is not None]
    ttft = sorted(r["ttft_ms"] for r in rows if r["ttft_ms"] is not None)
    tpot = [r["tpot_ms"] for r in rows if r.get("tpot_ms") is not None]
    gaps = [max(r["itl_ms"]) for r in rows if r.get("itl_ms")]
    runs = [run["summary"] for run in result.runs]
    gpu = [s["gpu_utilization_percent"].get("p50") for s in runs if s["gpu_utilization_percent"].get("count")]
    entry = {
        "ttft_p50_ms": statistics.median(ttft) if ttft else None,
        "ttft_p95_ms": percentile(ttft, 95) if ttft else None,
        "tpot_p50_ms": statistics.median(tpot) if tpot else None,
        "worst_gap_ms": max(gaps) if gaps else None,
        "worst_gap_p50_ms": statistics.median(gaps) if gaps else None,
        "worst_gap_run_median_ms": statistics.median(run_worst) if run_worst else None,
        "tokens_per_s": statistics.median(s["tokens_per_second"] for s in runs),
        "duration_ms": statistics.median(s["duration_ms"] for s in runs),
        "gpu_util_p50": statistics.median(gpu) if gpu else None,
        "completed": sum(1 for r in rows if r["status"] == "completed"),
        "expected": len(requests) * (len(result.runs) if expected_runs is None else expected_runs),
    }
    entry.update(case_validity(entry))
    return entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sections", nargs="+", choices=SECTIONS, default=list(SECTIONS))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"{args.output_dir} is not empty; choose a new dated directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_qwen_fp8(fp8_kernel_path="sm89", local_files_only=True)
    model, tok = bundle.model, bundle.tokenizer
    harness = BenchmarkHarness(HarnessConfig(
        repetitions=args.repetitions, warmup_repetitions=args.warmup_repetitions,
        respect_arrival_schedule=True, collect_gpu=True, collect_system_telemetry=False,
        sample_interval_seconds=0.25))
    summary = {"created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "model": MODEL_ID, "revision": MODEL_REVISION, "gpu": torch.cuda.get_device_name(0),
               "repetitions": args.repetitions, "warmup_repetitions": args.warmup_repetitions,
               "hf_reference_check": False, "sections": {}}

    def run(section, name, requests, make_runner, *, batch_size=None, label=None, extra=None):
        workload = WorkloadSpec(name=f"{section}_{name}", seed=17, requests=tuple(requests), model_id=MODEL_ID,
                                model_revision=MODEL_REVISION, dtype="fp8", device="cuda:0",
                                arrival_pattern="closed_loop" if batch_size else "fixed_rate",
                                metadata={"section": section, "engine": name})
        runner = make_runner()
        retries_before = torch.cuda.memory_stats().get("num_alloc_retries", 0)
        torch.cuda.reset_peak_memory_stats()
        try:
            result = (harness.run_batched(workload, batch_size, runner) if batch_size
                      else harness.run_trace(workload, runner))
        finally:
            close = getattr(runner, "close", None)
            if close:
                close()
            del runner
            gc.collect()
            torch.cuda.empty_cache()
        entry = {"engine": name, "label": label or name, **(extra or {}),
                 **metrics(result, requests, expected_runs=args.repetitions),
                 # Allocator retries mean the run hit GPU memory pressure (cache flush + retry).
                 "alloc_retries": torch.cuda.memory_stats().get("num_alloc_retries", 0) - retries_before,
                 "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
        entry.update(case_validity(entry))
        summary["sections"].setdefault(section, []).append(entry)
        raw = result.to_dict()
        raw["case_metrics"] = entry
        (args.output_dir / f"{section}_{name}.json").write_text(json.dumps(raw, default=str) + "\n")
        (args.output_dir / "showcase.json").write_text(json.dumps(summary, indent=1) + "\n")
        print_invalid_case(f"[{section}] {name}", entry)
        print(f"[{section}] {name}: retries {entry['alloc_retries']}, peak {entry['peak_allocated_gib']:.1f} GiB, "
              f"TTFT p50 {entry['ttft_p50_ms'] or 0:,.0f} ms, TPOT p50 "
              f"{entry['tpot_p50_ms'] or 0:,.1f} ms, {entry['tokens_per_s']:,.1f} tok/s, "
              f"{entry['completed']}/{entry['expected']} completed", flush=True)

    if "single" in args.sections or "batch" in args.sections:
        from minillm_l4.benchmarks.runners.huggingface_baseline import HuggingFaceGreedyBatchRunner
        from minillm_l4.benchmarks.runners.kv_cache import KvCacheBatchRunner
        from minillm_l4.benchmarks.runners.manual_decode import ManualGreedyBatchRunner
        from minillm_l4.benchmarks.runners.paged_cuda_graph import PagedCudaGraphBatchRunner
        from minillm_l4.benchmarks.runners.paged_kv import PagedAttentionBatchRunner

        p512 = prompt(tok, 512, "decode")
        engines = {
            "hf_generate": ("HF generate()", lambda reqs: HuggingFaceGreedyBatchRunner(model, device="cuda")),
            "manual_loop": ("Manual decode loop", lambda reqs: ManualGreedyBatchRunner(model, device="cuda")),
            "no_kv_cache": ("No KV cache (recompute)", lambda reqs: KvCacheBatchRunner(model, mode="recompute", device="cuda")),
            "kv_cache": ("Explicit KV cache", lambda reqs: KvCacheBatchRunner(model, mode="contiguous", device="cuda")),
            "paged_triton": ("Paged KV + Triton decode", lambda reqs: PagedAttentionBatchRunner(
                model, block_size=16, num_blocks=pages(reqs), device="cuda", prefill_backend="sdpa")),
            "cuda_graph": ("Paged KV + CUDA Graph decode", lambda reqs: PagedCudaGraphBatchRunner(
                model, block_size=16, num_blocks=pages(reqs), device="cuda:0", prefill_backend="auto")),
        }
        if "single" in args.sections:
            reqs = [RequestSpec("single-r0", p512, 128)]
            for name, (label, make) in engines.items():
                run("single", name, reqs, lambda make=make: make(reqs), batch_size=1, label=label)
        if "batch" in args.sections:
            for batch in (1, 4, 8, 16):
                reqs = [RequestSpec(f"batch{batch}-r{i}", p512, 128) for i in range(batch)]
                for name in ("hf_generate", "kv_cache", "paged_triton", "cuda_graph"):
                    label, make = engines[name]
                    run("batch", f"{name}_b{batch}", reqs, lambda make=make: make(reqs), batch_size=batch,
                        label=label, extra={"batch": batch, "family": name})

    if "serving" in args.sections:
        from minillm_l4.benchmarks.runners.concurrent_requests import StaticRequestTraceRunner
        from minillm_l4.benchmarks.runners.continuous_requests import ContinuousRequestTraceRunner

        # Prompts sized so dense KV fits without allocator retries; every policy may run
        # all 16 requests at once, so only the scheduling / KV layout differs.
        lengths = [128, 256, 512, 1024] * 4
        random.Random(17).shuffle(lengths)
        prompts = {n: prompt(tok, n, "serve") for n in set(lengths)}
        reqs = [RequestSpec(f"serve-r{i}", prompts[n], 128, scheduled_arrival_ms=150.0 * i) for i, n in enumerate(lengths)]
        policies = {
            "static": ("Static batching", lambda: StaticRequestTraceRunner(model, max_batch_size=16, device="cuda")),
            "continuous_dense": ("Continuous batching (dense KV)", lambda: ContinuousRequestTraceRunner(
                model, max_batch_size=16, max_prefill_tokens=2048, device="cuda")),
            "continuous_paged": ("Continuous + paged KV", lambda: paged(model, reqs, prefill_chunk_size=None)),
            "mixed_adaptive": ("Mixed batching + adaptive chunks", lambda: paged(
                model, reqs, prefill_chunk_size=None, mixed_batch=True, adaptive_chunking=True)),
        }
        for name, (label, make) in policies.items():
            run("serving", name, reqs, make, label=label)

    if "longmix" in args.sections:
        prompts = {n: prompt(tok, n, "long") for n in (128, 6144)}
        reqs = [RequestSpec(f"long-r{i}", prompts[128 if i % 2 == 0 else 6144], 64, scheduled_arrival_ms=400.0 * i)
                for i in range(6)]
        policies = {
            "whole": ("Whole-prompt prefill", lambda: paged(model, reqs, prefill_chunk_size=None)),
            "mixed_512": ("Mixed batching, 512-token budget", lambda: paged(
                model, reqs, prefill_chunk_size=None, mixed_batch=True, max_prefill_tokens=512)),
            "mixed_2048": ("Mixed batching, 2048-token budget", lambda: paged(
                model, reqs, prefill_chunk_size=None, mixed_batch=True, max_prefill_tokens=2048)),
            "adaptive": ("Mixed batching + adaptive chunks", lambda: paged(
                model, reqs, prefill_chunk_size=None, mixed_batch=True, adaptive_chunking=True)),
        }
        for name, (label, make) in policies.items():
            run("longmix", name, reqs, make, label=label)

    if "prefix" in args.sections:
        system = prompt(tok, 2048, "system")
        reqs = [RequestSpec(f"prefix-r{i}", system + prompt(tok, 64, f"user{i}"), 64,
                            scheduled_arrival_ms=300.0 * i) for i in range(8)]
        for name, label, enabled in (("off", "No prefix cache", False), ("on", "Prefix cache", True)):
            run("prefix", name, reqs, lambda enabled=enabled: paged(
                model, reqs, prefill_chunk_size=None, enable_prefix=enabled, max_entries=8), label=label)

    if "prefill" in args.sections:
        for n in (128, 256, 512, 1024, 2048, 4096, 8192):
            reqs = [RequestSpec(f"prefill-{n}", prompt(tok, n, "prefill"), 1)]
            run("prefill", f"p{n}", reqs, lambda reqs=reqs: paged(model, reqs, prefill_chunk_size=None),
                label=f"{n} tokens", extra={"prompt_tokens": n})

    print(f"Summary: {args.output_dir / 'showcase.json'}")
    exit_if_invalid(entry for entries in summary["sections"].values() for entry in entries)


if __name__ == "__main__":
    main()
