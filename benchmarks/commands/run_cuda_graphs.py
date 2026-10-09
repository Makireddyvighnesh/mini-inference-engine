"""One-session eager/graph comparison for continuous and mixed serving.

One warmup and three measured runs per case; no HF reference check. Run the
policy equivalence gate separately. Each runner is freed before the next case.

  .conda-env/bin/python -m minillm_l4.benchmarks.commands.run_cuda_graphs \
      --output-dir minillm_l4/results/cuda_graphs_20261008
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import time
from pathlib import Path

import torch

from minillm_l4.benchmarks.commands.run_showcase import MODEL_ID, MODEL_REVISION, load_qwen_fp8, metrics, paged, prompt
from minillm_l4.benchmarks.commands.run_vllm_compare import case_validity, exit_if_invalid, print_invalid_case
from minillm_l4.benchmarks.core.harness import BenchmarkHarness
from minillm_l4.benchmarks.core.schemas import HarnessConfig, RequestSpec, WorkloadSpec

SECTIONS = ("decode", "prefill", "serving", "overhead")
BATCH_SIZES = (1, 2, 4, 8, 16, 32)
PREFILL_LENGTHS = (128, 256, 512, 1024, 2048, 4096, 8192)


def case_metrics(result, requests, scheduler_runs, *, expected_runs=None):
    """Showcase metrics plus decode-only rate and actual replay share.

    Decode throughput counts tokens after the first, divided by summed decode
    step wall time (packing, replay/eager forward, readback, and bookkeeping).
    It is emitted only when all decode steps are separate from prompt work.
    E2E throughput includes prompt work, arrivals, and completion/cleanup.
    """
    entry = metrics(result, requests, expected_runs=expected_runs)
    rates = []
    for summary in scheduler_runs:
        steps = [r for r in summary["execution_records"] if r["kind"] == "decode"]
        if steps and all(r.get("wall_ms") is not None for r in steps):
            wall_ms = sum(r["wall_ms"] for r in steps)
            if wall_ms:
                rates.append(1000 * sum(r["batch_size"] for r in steps) / wall_ms)
    replays = sum(s["graph_replays"] for s in scheduler_runs)
    eager = sum(s["eager_decode_steps"] for s in scheduler_runs)
    modes = {s["decode_mode_used"] for s in scheduler_runs}
    entry.update(
        decode_tokens_per_s=statistics.median(rates) if rates else None,
        end_to_end_tokens_per_s=entry["tokens_per_s"],
        output_tokens_per_s=entry["tokens_per_s"],
        gpu_busy_percent=entry["gpu_util_p50"],
        worst_pause_ms=entry["worst_gap_run_median_ms"],
        graph_replay_share=replays / (replays + eager) if replays + eager else 0.0,
        graph_replays=replays, eager_decode_steps=eager,
        decode_mode_used=next(iter(modes)) if len(modes) == 1 else "mixed",
        measured_capture_count=sum(len(s["graph_captures_this_run"]) for s in scheduler_runs),
    )
    return entry


def serving_requests(tokenizer, workload):
    if workload == "serving":
        lengths = [128, 256, 512, 1024] * 4
        random.Random(17).shuffle(lengths)
        interval, outputs, tag = 150.0, 128, "serve"
    elif workload == "longmix":
        lengths = [128, 6144] * 3
        interval, outputs, tag = 400.0, 64, "long"
    else:
        raise ValueError("unknown serving workload")
    prompts = {n: prompt(tokenizer, n, tag) for n in set(lengths)}
    return tuple(RequestSpec(f"{tag}-r{i}", prompts[n], outputs, scheduled_arrival_ms=interval * i)
                 for i, n in enumerate(lengths))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sections", nargs="+", choices=SECTIONS, default=list(SECTIONS))
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"{args.output_dir} is not empty; choose a new dated directory")
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA; eager CPU fallback cannot measure graphs")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_qwen_fp8(fp8_kernel_path="sm89", local_files_only=True)
    model, tokenizer = bundle.model, bundle.tokenizer
    harness = BenchmarkHarness(HarnessConfig(
        repetitions=3, warmup_repetitions=1, respect_arrival_schedule=True,
        collect_gpu=True, collect_system_telemetry=False, sample_interval_seconds=0.25,
    ))
    summary = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": MODEL_ID, "revision": MODEL_REVISION, "gpu": torch.cuda.get_device_name(0),
        "repetitions": 3, "warmup_repetitions": 1, "hf_reference_check": False,
        "capture_policy": "lazy per bucket; retained across warmup and measured runs",
        "decode_rate_definition": "tokens after first / summed decode-only step wall seconds",
        "graph_replay_share_definition": "graph decode steps / (graph + eager decode steps), measured runs only",
        "gpu_busy_definition": "median across runs of sampled GPU utilization p50",
        "sections": {},
    }

    def save_summary():
        (args.output_dir / "cuda_graphs.json").write_text(json.dumps(summary, indent=2) + "\n")

    def run(section, name, requests, *, mode, policy="whole", **extra):
        options = {"prefill_chunk_size": None, "cuda_graphs": mode != "eager"}
        if policy == "mixed_adaptive":
            options.update(mixed_batch=True, adaptive_chunking=True, graph_mixed_decode=mode == "graph_split")
        workload = WorkloadSpec(
            name=f"{section}_{name}", seed=17, requests=tuple(requests), model_id=MODEL_ID,
            model_revision=MODEL_REVISION, dtype="fp8", device="cuda:0", arrival_pattern="fixed_rate",
            metadata={"section": section, "policy": policy, "mode": mode, **extra},
        )
        retries_before = torch.cuda.memory_stats().get("num_alloc_retries", 0)
        torch.cuda.reset_peak_memory_stats()
        runner = paged(model, requests, **options)
        try:
            result = harness.run_trace(workload, runner)
            scheduler_runs = list(runner.run_summaries[1:])
            warmup_summary = runner.run_summaries[0]
            # Read before cleanup; do not retain bound methods or graph tensors.
            memory = {
                "alloc_retries": torch.cuda.memory_stats().get("num_alloc_retries", 0) - retries_before,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            }
            overhead = {
                "case": name, "section": section, "policy": policy, "mode": mode, **extra,
                "captured_buckets": runner.last_summary["captured_buckets"],
                "capture_ms": runner.last_summary["capture_ms"],
                "graph_pool_memory_bytes_per_bucket": runner.last_summary["graph_pool_memory_bytes_per_bucket"],
                "graph_pool_memory_bytes": runner.last_summary["graph_pool_memory_bytes"],
                "scratch_pages": runner.last_summary["graph_scratch_pages"],
                "scratch_memory_bytes": runner.last_summary["graph_scratch_memory_bytes"],
                "static_buffer_memory_bytes_per_bucket": runner.last_summary["graph_static_buffer_memory_bytes_per_bucket"],
            }
        finally:
            runner.close()
            del runner
            gc.collect()
            torch.cuda.empty_cache()

        entry = {"engine": name, "label": name, "policy": policy, "mode": mode, **extra,
                 **case_metrics(result, requests, scheduler_runs, expected_runs=3), **memory,
                 "graph_fallback_reasons": sorted({reason for s in scheduler_runs for reason in s["graph_fallback_steps_by_reason"]}),
                 "captured_buckets": overhead["captured_buckets"],
                 "capture_ms": overhead["capture_ms"],
                 "graph_pool_memory_bytes": overhead["graph_pool_memory_bytes"]}
        reasons = []
        if mode != "eager" and section != "prefill" and not entry["graph_replays"]:
            reasons.append(f"no measured graph replays: {entry['graph_fallback_reasons']}")
        entry.update(case_validity(entry, *reasons))
        overhead.update({key: entry[key] for key in ("completed", "expected", "alloc_retries", "valid", "invalid_reason")})
        raw = result.to_dict()
        raw["cuda_graphs"] = {"metrics": entry, "warmup_scheduler": warmup_summary,
                              "measured_schedulers": scheduler_runs, "overhead": overhead}
        (args.output_dir / f"{section}_{name}.json").write_text(json.dumps(raw, default=str) + "\n")
        summary["sections"].setdefault(section, []).append(entry)
        if "overhead" in args.sections and mode != "eager":
            summary["sections"].setdefault("overhead", []).append(overhead)
        save_summary()
        print_invalid_case(f"[{section}] {name}", entry)
        print(f"[{section}] {name}: TTFT {entry['ttft_p50_ms'] or 0:.1f} ms, "
              f"TPOT {entry['tpot_p50_ms'] or 0:.1f} ms, {entry['tokens_per_s']:.1f} tok/s, "
              f"GPU busy {entry['gpu_busy_percent']}, replay share {entry['graph_replay_share']:.1%}, "
              f"retries {entry['alloc_retries']}, peak {entry['peak_allocated_gib']:.2f} GiB", flush=True)

    if "decode" in args.sections:
        for length, batches in ((512, BATCH_SIZES), (4096, (8,))):
            tokens = prompt(tokenizer, length, "decode")
            for batch in batches:
                requests = tuple(RequestSpec(f"decode-r{i}", tokens, 128) for i in range(batch))
                for mode in ("eager", "graph"):
                    run("decode", f"p{length}_b{batch}_{mode}", requests, mode=mode,
                        batch=batch, prompt_tokens=length)

    if "prefill" in args.sections:
        for length in PREFILL_LENGTHS:
            requests = (RequestSpec(f"prefill-{length}", prompt(tokenizer, length, "prefill"), 1),)
            for mode in ("eager", "graph"):
                run("prefill", f"p{length}_{mode}", requests, mode=mode, prompt_tokens=length)

    if "serving" in args.sections:
        for workload_name in ("serving", "longmix"):
            requests = serving_requests(tokenizer, workload_name)
            for policy, modes in (("whole", ("eager", "graph")),
                                  ("mixed_adaptive", ("eager", "graph", "graph_split"))):
                for mode in modes:
                    run("serving", f"{workload_name}_{policy}_{mode}", requests,
                        mode=mode, policy=policy, workload=workload_name)

    if "overhead" in args.sections and "decode" not in args.sections:
        # Explicit overhead-only invocation still captures every default bucket.
        tokens = prompt(tokenizer, 512, "decode")
        for batch in BATCH_SIZES:
            requests = tuple(RequestSpec(f"capture-r{i}", tokens, 2) for i in range(batch))
            run("capture", f"b{batch}_graph", requests, mode="graph", batch=batch, prompt_tokens=512)
    save_summary()
    print(f"Summary: {args.output_dir / 'cuda_graphs.json'}")
    exit_if_invalid(entry for entries in summary["sections"].values() for entry in entries)


if __name__ == "__main__":
    main()
