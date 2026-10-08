"""Prefill-only TTFT: how long until each request's first token.

Every request generates exactly one token, so a request's TTFT is its wait
plus its prompt prefill.  Requests run through the continuous paged engine
(resource admission: nothing queues while KV pages are free).  Workloads:

* uniform  - B prompts of one length arriving together (B=1 is "one prompt");
* mixed    - B prompts of different lengths arriving together in one batch;
* arrivals - prompts of mixed lengths arriving over time (continuous batching);
* intrusion - a long prompt starts on an idle engine and a 128-token request
  arrives a little later: how long does the newcomer wait?

Policies:

* sequential - each prompt in its own forward, FIFO;
* flattened  - prompts that are waiting share one packed forward (cu_seqlens,
  no padding), FIFO up to --packed-token-limit tokens per forward;
* mixed_N    - budgeted iteration batching: at most N tokens per forward,
  shortest remaining prompt first, long prompts split into chunks;
* adaptive   - chunk size chosen per step from a time limit (short when busy,
  longer when idle), a self-calibrating step-time model, aging, and free
  GPU memory (``engine/step_planner.py``).

No Hugging Face reference check is run (the engine paths are verified by the
test suite).  Loading and warm-up are excluded from every timing.

  .conda-env/bin/python -m minillm_l4.benchmarks.commands.run_prefill_ttft \
      --output-dir minillm_l4/results/prefill_ttft_20261007
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
from minillm_l4.benchmarks.core.schemas import HarnessConfig, RequestSpec, WorkloadSpec
from minillm_l4.benchmarks.core.synthetic import SyntheticSample, exact_token_ids
from minillm_l4.benchmarks.runners.chunked_prefill import ChunkedPrefillPagedRunner
from minillm_l4.benchmarks.runners.huggingface_baseline import MODEL_ID, MODEL_REVISION, load_qwen_fp8

LENGTHS = (128, 256, 512, 1024, 2048, 4096, 8192)
MIXES = {
    "short": (128, 256, 512),
    "medium": (512, 1024, 2048),
    "long": (2048, 4096, 8192),
    "all": LENGTHS,
}
SEED_TEXT = (
    "Explain how prefill, cached decoding, and batching affect language-model "
    "inference. Give concrete comparisons."
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=list(LENGTHS))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--mixes", nargs="+", choices=sorted(MIXES), default=["short", "medium", "long", "all"])
    parser.add_argument("--arrival-intervals-ms", type=float, nargs="+", default=[50.0, 200.0, 500.0],
                        help="continuous-arrival cases: one request every interval (mix 'all')")
    parser.add_argument("--arrival-requests", type=int, default=14)
    parser.add_argument("--policies", nargs="+", default=["sequential", "flattened", "mixed_2048"])
    parser.add_argument("--packed-token-limit", type=int, default=16384)
    parser.add_argument("--max-total-tokens", type=int, default=65536,
                        help="skip cases whose prompts exceed this many tokens in total (KV memory)")
    parser.add_argument("--workloads", nargs="+", choices=["uniform", "mixed", "arrivals", "intrusion"],
                        default=["uniform", "mixed", "arrivals"])
    parser.add_argument("--intrusion-long-lengths", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--intrusion-delays-ms", type=float, nargs="+", default=[20.0, 100.0, 400.0, 1000.0])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for policy in args.policies:
        if policy not in {"sequential", "flattened", "adaptive"} and not (
                policy.startswith("mixed_") and policy[6:].isdigit()):
            parser.error(f"unknown policy {policy!r}")
    return args


def plan_cases(args):
    rng = random.Random(args.seed)
    cases = []
    if "uniform" in args.workloads:
        for length in args.prompt_lengths:
            for batch in args.batch_sizes:
                cases.append({"workload": "uniform", "name": f"uniform_p{length}_b{batch}",
                              "lengths": [length] * batch, "interval_ms": 0.0})
    if "mixed" in args.workloads:
        for mix in args.mixes:
            for batch in args.batch_sizes:
                if batch < 2:
                    continue
                lengths = [MIXES[mix][i % len(MIXES[mix])] for i in range(batch)]
                rng.shuffle(lengths)
                cases.append({"workload": "mixed", "name": f"mixed_{mix}_b{batch}", "mix": mix,
                              "lengths": lengths, "interval_ms": 0.0})
    if "arrivals" in args.workloads:
        for interval in args.arrival_intervals_ms:
            lengths = [LENGTHS[i % len(LENGTHS)] for i in range(args.arrival_requests)]
            rng.shuffle(lengths)
            cases.append({"workload": "arrivals", "name": f"arrivals_all_every{interval:g}ms", "mix": "all",
                          "lengths": lengths, "interval_ms": float(interval)})
    if "intrusion" in args.workloads:
        for long in args.intrusion_long_lengths:
            for delay in args.intrusion_delays_ms:
                cases.append({"workload": "intrusion", "name": f"intrusion_p{long}_short_at{delay:g}ms",
                              "lengths": [long, 128], "interval_ms": float(delay)})
    planned = []
    for case in cases:
        total = sum(case["lengths"])
        for policy in args.policies:
            planned.append({**case, "policy": policy, "total_tokens": total,
                            "key": f"{case['name']}_{policy}",
                            "skipped": total > args.max_total_tokens})
    return planned


def build_runner(model, case, args):
    block = 16
    blocks = sum(math.ceil(length / block) for length in case["lengths"]) + len(case["lengths"])
    policy = case["policy"]
    budget = int(policy[6:]) if policy.startswith("mixed_") else None
    adaptive = policy == "adaptive"
    return ChunkedPrefillPagedRunner(
        model, block_size=block, num_blocks=blocks, max_batch_size=len(case["lengths"]),
        max_prefill_tokens=budget or 256, prefill_chunk_size=None, device="cuda",
        decode_backend="auto", decode_sdpa_compat=True, enable_prefix=False,
        batched_prefill=policy == "flattened", mixed_batch=budget is not None or adaptive,
        adaptive_chunking=adaptive,
        packed_prefill_token_limit=args.packed_token_limit if policy == "flattened" else None,
    )


def summarize_case(case, result):
    runs = result.runs
    ttft, wait, by_length, makespans = [], [], {}, []
    for run in runs:
        finish = 0.0
        for request, record in zip(result.workload.requests, run["requests"], strict=True):
            metrics = record["metrics"]
            value = metrics["ttft_ms"]
            ttft.append(value)
            if metrics.get("prefill_ms") is not None:
                wait.append(max(0.0, value - metrics["prefill_ms"]))
            by_length.setdefault(request.prompt_tokens, []).append(value)
            finish = max(finish, request.scheduled_arrival_ms + value)
        makespans.append(finish)
    ttft.sort()
    p95 = ttft[min(len(ttft) - 1, math.ceil(0.95 * len(ttft)) - 1)]
    makespan = statistics.median(makespans)
    return {
        "ttft_p50_ms": statistics.median(ttft), "ttft_p95_ms": p95, "ttft_max_ms": max(ttft),
        "ttft_mean_ms": statistics.mean(ttft),
        "wait_p50_ms": statistics.median(wait) if wait else None,
        "ttft_p50_by_length_ms": {str(k): statistics.median(v) for k, v in sorted(by_length.items())},
        "makespan_ms": makespan,
        "prompt_tokens_per_s": case["total_tokens"] / (makespan / 1000) if makespan else None,
    }


def write_markdown(path, manifest):
    cases = [c for c in manifest["cases"] if c.get("summary")]
    by_key = {c["key"]: c for c in cases}
    policies = manifest["policies"]
    present = sorted({n for c in cases for n in c["lengths"]})
    lines = ["# Prefill-only TTFT", "",
             f"Captured {manifest['created_at_utc']}. Model `{MODEL_ID}` at `{MODEL_REVISION}`, SM89 FP8 path, "
             f"{manifest['repetitions']} measured runs after {manifest['warmup_repetitions']} warm-up per case. "
             "Each request generates one token, so TTFT = wait + prompt prefill. No HF reference check. "
             f"`flattened` packs waiting prompts FIFO up to {manifest['packed_token_limit']} tokens per forward; "
             "`mixed_N` caps each forward at N tokens, shortest prompt first.", ""]

    lines += ["## One prompt (batch 1)", "", "| Prompt tokens | TTFT (ms) | Prompt tokens/s |", "|---|---|---|"]
    for length in manifest["prompt_lengths"]:
        case = by_key.get(f"uniform_p{length}_b1_sequential") or by_key.get(f"uniform_p{length}_b1_flattened")
        if case:
            s = case["summary"]
            lines.append(f"| {length} | {s['ttft_p50_ms']:,.1f} | {s['prompt_tokens_per_s']:,.0f} |")

    def table(title, rows, label):
        out = ["", f"## {title}", "",
               "| " + label + " | Batch | Policy | TTFT P50 / P95 / max (ms) | Wait P50 (ms) | All done (ms) | Prompt tokens/s |",
               "|---|---|---|---|---|---|---|"]
        for case in rows:
            s = case["summary"]
            wait = f"{s['wait_p50_ms']:,.0f}" if s["wait_p50_ms"] is not None else "-"
            out.append(f"| {case['row_label']} | {len(case['lengths'])} | {case['policy']} | "
                       f"{s['ttft_p50_ms']:,.0f} / {s['ttft_p95_ms']:,.0f} / {s['ttft_max_ms']:,.0f} | {wait} | "
                       f"{s['makespan_ms']:,.0f} | {s['prompt_tokens_per_s']:,.0f} |")
        return out

    order = {p: i for i, p in enumerate(policies)}
    uniform = sorted((c for c in cases if c["workload"] == "uniform"),
                     key=lambda c: (c["lengths"][0], len(c["lengths"]), order[c["policy"]]))
    for c in uniform:
        c["row_label"] = str(c["lengths"][0])
    lines += table("Same-length batches (all arrive together)", uniform, "Prompt")

    mixed = sorted((c for c in cases if c["workload"] == "mixed"),
                   key=lambda c: (list(MIXES).index(c["mix"]), len(c["lengths"]), order[c["policy"]]))
    for c in mixed:
        c["row_label"] = f"{c['mix']} {MIXES[c['mix']]}"
    lines += table("Mixed-length batches (all arrive together)", mixed, "Mix")
    if mixed:
        lines += ["", "TTFT P50 by prompt length inside mixed batches (ms):", "",
                  "| Mix | Batch | Policy | " + " | ".join(str(n) for n in present) + " |",
                  "|---|---|---|" + "---|" * len(present)]
        for c in mixed:
            per = c["summary"]["ttft_p50_by_length_ms"]
            cells = [f"{per[str(n)]:,.0f}" if str(n) in per else "" for n in present]
            lines.append(f"| {c['mix']} | {len(c['lengths'])} | {c['policy']} | " + " | ".join(cells) + " |")

    arrivals = sorted((c for c in cases if c["workload"] == "arrivals"),
                      key=lambda c: (c["interval_ms"], order[c["policy"]]))
    for c in arrivals:
        c["row_label"] = f"every {c['interval_ms']:g} ms"
    lines += table("Continuous arrivals (mixed lengths 128-8192)", arrivals, "Arrivals")
    if arrivals:
        lines += ["", "TTFT P50 by prompt length under continuous arrivals (ms):", "",
                  "| Arrivals | Policy | " + " | ".join(str(n) for n in present) + " |",
                  "|---|---|" + "---|" * len(present)]
        for c in arrivals:
            per = c["summary"]["ttft_p50_by_length_ms"]
            cells = [f"{per[str(n)]:,.0f}" if str(n) in per else "" for n in present]
            lines.append(f"| every {c['interval_ms']:g} ms | {c['policy']} | " + " | ".join(cells) + " |")

    intrusion = sorted((c for c in cases if c["workload"] == "intrusion"),
                       key=lambda c: (c["lengths"][0], c["interval_ms"], order[c["policy"]]))
    if intrusion:
        lines += ["", "## Intrusion: a 128-token request arrives while a long prompt prefills", "",
                  "| Long prompt | Short arrives at | Policy | Short request TTFT (ms) | Long prompt TTFT (ms) | All done (ms) |",
                  "|---|---|---|---|---|---|"]
        for c in intrusion:
            per = c["summary"]["ttft_p50_by_length_ms"]
            lines.append(f"| {c['lengths'][0]} | {c['interval_ms']:g} ms | {c['policy']} | "
                         f"{per['128']:,.0f} | {per[str(c['lengths'][0])]:,.0f} | {c['summary']['makespan_ms']:,.0f} |")

    skipped = [c["key"] for c in manifest["cases"] if c.get("skipped")]
    if skipped:
        lines += ["", f"Skipped (more than {manifest['max_total_tokens']} prompt tokens): " + ", ".join(skipped)]
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    cases = plan_cases(args)
    if args.dry_run:
        print(json.dumps([{k: c[k] for k in ("key", "total_tokens", "skipped")} for c in cases], indent=1))
        print(f"{sum(not c['skipped'] for c in cases)} cases to run, {sum(c['skipped'] for c in cases)} skipped")
        return
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not visible; run this on the L4 host.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"{args.output_dir} is not empty; choose a new dated directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    markdown = args.markdown_output or args.output_dir / "report.md"

    bundle = load_qwen_fp8(fp8_kernel_path="sm89", local_files_only=True,
                           pretune_tokens=max(16384, args.packed_token_limit))
    tokenizer = bundle.tokenizer
    prompts = {length: tuple(exact_token_ids(tokenizer, SyntheticSample(
        sample_id=f"prefill-ttft-{length}", category="prefill-only", target_prompt_tokens=length,
        target_output_tokens=1, seed_text=SEED_TEXT))) for length in sorted(set(args.prompt_lengths) | set(LENGTHS))}

    manifest = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "kernel_strategy": bundle.kernel_strategy},
        "prompt_lengths": args.prompt_lengths, "policies": args.policies,
        "packed_token_limit": args.packed_token_limit, "max_total_tokens": args.max_total_tokens,
        "repetitions": args.repetitions, "warmup_repetitions": args.warmup_repetitions,
        "hf_reference_check": False, "cases": [],
    }
    harness = BenchmarkHarness(HarnessConfig(
        repetitions=args.repetitions, warmup_repetitions=args.warmup_repetitions,
        respect_arrival_schedule=True, collect_gpu=True, collect_system_telemetry=False,
        sample_interval_seconds=0.25,
    ))
    for index, case in enumerate(cases, start=1):
        entry = {k: case[k] for k in ("key", "workload", "policy", "lengths", "interval_ms", "total_tokens", "skipped")}
        entry["mix"] = case.get("mix")
        if case["skipped"]:
            manifest["cases"].append(entry)
            continue
        requests = tuple(
            RequestSpec(f"{case['key']}-r{i}", prompts[length], 1,
                        scheduled_arrival_ms=i * case["interval_ms"], category=f"p{length}")
            for i, length in enumerate(case["lengths"]))
        workload = WorkloadSpec(name=case["key"], seed=args.seed, requests=requests, model_id=MODEL_ID,
                                model_revision=MODEL_REVISION, dtype="fp8", device="cuda:0",
                                arrival_pattern="fixed_rate" if case["interval_ms"] else "closed_loop",
                                metadata={"workload": case["workload"], "policy": case["policy"]})
        runner = build_runner(bundle.model, case, args)
        try:
            result = harness.run_trace(workload, runner)
            runner_summaries = list(getattr(runner, "run_summaries", []))
        finally:
            # Free this case's KV pool before the next case allocates its own.
            runner.close()
            del runner
            gc.collect()
            torch.cuda.empty_cache()
        # summary counts span every measured repetition.
        failed = result.summary.get("failed_requests", 0)
        completed = result.summary.get("completed_requests", 0)
        entry["status"] = "completed" if not failed and completed == len(requests) * args.repetitions else "failed"
        entry["summary"] = summarize_case(case, result) if entry["status"] == "completed" else None
        payload = result.to_dict()
        payload["prefill_ttft"] = entry
        payload["runner_summaries"] = runner_summaries
        (args.output_dir / f"{case['key']}.json").write_text(json.dumps(payload, default=str) + "\n")
        manifest["cases"].append(entry)
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
        write_markdown(markdown, manifest)
        s = entry["summary"]
        peak = torch.cuda.max_memory_allocated() / 2**30
        torch.cuda.reset_peak_memory_stats()
        held = torch.cuda.memory_allocated() / 2**30
        print(f"[{index}/{len(cases)}] {case['key']}: peak {peak:.1f} GiB, held after {held:.1f} GiB, " + (
            f"TTFT p50 {s['ttft_p50_ms']:,.0f} ms, max {s['ttft_max_ms']:,.0f} ms, "
            f"{s['prompt_tokens_per_s']:,.0f} prompt tok/s" if s else "FAILED"), flush=True)
    print(f"Report: {markdown}")


if __name__ == "__main__":
    main()
