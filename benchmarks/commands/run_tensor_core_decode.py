"""Tensor-core decode attention in the real engine: time per token and token agreement.

Runs the recommended engine configuration (mixed batching, adaptive chunks, CUDA Graph
decode, fused kernels, merged projections) with the default exact kernel and with
`decode_tensor_core=True`, for batch 1-32 x prompts of 512, 1,024 and 2,048 tokens.

Each case runs both modes in two blocks, order flipped between them, each block with a
fresh runner, one warm-up (captures graphs) and the measured runs. TPOT is the median
token gap while every request is decoding, so prefill does not enter it.

No Hugging Face reference is run: the check is agreement with the engine's exact path.

  .conda-env/bin/python -m minillm_l4.benchmarks.commands.run_tensor_core_decode \\
      --output-dir minillm_l4/results/tensor_core_decode_<date>
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

BATCHES = (1, 2, 4, 8, 16, 32)
PROMPTS = (512, 1024, 2048)
MODES = ("exact", "tensor_core")


def compare_tokens(exact: Sequence[Sequence[int]], other: Sequence[Sequence[int]]) -> dict[str, Any]:
    """How many requests match, and the earliest output position that differs."""

    if len(exact) != len(other):
        raise ValueError("both modes must return the same number of requests")
    first_difference = [next((i for i, (a, b) in enumerate(zip(x, y)) if a != b), None) for x, y in zip(exact, other)]
    differing = [position for position in first_difference if position is not None]
    return {"requests": len(exact), "identical": len(exact) - len(differing),
            "earliest_difference": min(differing) if differing else None}


def format_report(rows: list[dict[str, Any]]) -> str:
    total = sum(r["tokens"]["requests"] for r in rows)
    identical = sum(r["tokens"]["identical"] for r in rows)
    earliest = [r["tokens"]["earliest_difference"] for r in rows if r["tokens"]["earliest_difference"] is not None]
    lines = ["# Tensor-core decode attention in the engine", "",
             f"Requests with tokens identical to the exact path: {identical}/{total}"
             + (f"; earliest difference at output token {min(earliest)}." if earliest else "."), "",
             "| Batch | Prompt tokens | Exact TPOT (ms) | Tensor-core TPOT (ms) | Gain | Identical | First difference |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        first = r["tokens"]["earliest_difference"]
        lines.append(f"| {r['batch']} | {r['prompt_tokens']:,} | {r['exact_tpot_ms']:.2f} | {r['tensor_core_tpot_ms']:.2f} | "
                     f"{1 - r['tensor_core_tpot_ms'] / r['exact_tpot_ms']:+.1%} | "
                     f"{r['tokens']['identical']}/{r['tokens']['requests']} | {'-' if first is None else first} |")
    return "\n".join(lines) + "\n"


def run_block(model, requests, mode: str, repetitions: int) -> tuple[list[float], list[tuple[int, ...]], dict[str, Any]]:
    import torch

    from minillm_l4.benchmarks.commands.run_showcase import pages
    from minillm_l4.benchmarks.core.harness import RequestEventRecorder
    from minillm_l4.engine.serving.chunked_prefill import ChunkedPrefillPagedRunner

    import time

    runner = ChunkedPrefillPagedRunner(
        model, block_size=16, num_blocks=pages(requests), max_batch_size=32, max_prefill_tokens=256, device="cuda",
        decode_backend="auto", enable_prefix=False, prefill_chunk_size=None, mixed_batch=True, adaptive_chunking=True,
        cuda_graphs=True, decode_tensor_core=mode == "tensor_core")
    tpots: list[float] = []
    tokens: list[tuple[int, ...]] = []
    try:
        for repetition in range(repetitions + 1):
            start = time.perf_counter_ns()
            recorders = tuple(RequestEventRecorder(r.request_id, run_started_ns=start) for r in requests)
            outcomes = runner(requests, recorders)
            if any(outcome.status != "completed" for outcome in outcomes):
                raise RuntimeError(f"incomplete requests: {[outcome.status for outcome in outcomes]}")
            summary = dict(runner.last_summary)
            if repetition == 0:
                continue  # Warm-up: graph capture happens here.
            tokens = [tuple(outcome.generated_token_ids) for outcome in outcomes]
            times = [sorted(event.timestamp_ns / 1e6 for event in recorder.events if event.event == "token_ready")
                     for recorder in recorders]
            all_decoding = max(row[0] for row in times)
            tpots.append(statistics.median(b - a for a, b in zip(times[0], times[0][1:]) if a >= all_decoding))
    finally:
        runner.close()
        del runner
        gc.collect()
        torch.cuda.empty_cache()
    return tpots, tokens, summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=list(BATCHES))
    parser.add_argument("--prompts", nargs="+", type=int, default=list(PROMPTS))
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"{args.output_dir} is not empty; choose a new dated directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from minillm_l4.benchmarks.commands.run_showcase import load_qwen_fp8, prompt
    from minillm_l4.benchmarks.core.schemas import RequestSpec
    from minillm_l4.engine.kernels.fused import install_fused_kernels
    from minillm_l4.engine.kernels.merged_projections import install_merged_projections

    bundle = load_qwen_fp8(fp8_kernel_path="sm89", local_files_only=True)
    if not install_fused_kernels(bundle.model):
        raise RuntimeError("fused kernels unavailable")
    install_merged_projections(bundle.model)

    rows: list[dict[str, Any]] = []
    for batch in args.batches:
        for prompt_tokens in args.prompts:
            requests = [RequestSpec(f"b{batch}-p{prompt_tokens}-r{i}", prompt(bundle.tokenizer, prompt_tokens, f"tc-r{i}"),
                                    args.new_tokens, scheduled_arrival_ms=0.0) for i in range(batch)]
            tpots: dict[str, list[float]] = {mode: [] for mode in MODES}
            tokens: dict[str, list[tuple[int, ...]]] = {}
            summaries: dict[str, dict[str, Any]] = {}
            for block_order in (MODES, tuple(reversed(MODES))):
                for mode in block_order:
                    measured, outputs, summaries[mode] = run_block(bundle.model, requests, mode, args.repetitions)
                    tpots[mode] += measured
                    tokens[mode] = outputs
            row = {"batch": batch, "prompt_tokens": prompt_tokens, "new_tokens": args.new_tokens,
                   "exact_tpot_ms": statistics.median(tpots["exact"]),
                   "tensor_core_tpot_ms": statistics.median(tpots["tensor_core"]),
                   "exact_samples_ms": tpots["exact"], "tensor_core_samples_ms": tpots["tensor_core"],
                   "tokens": compare_tokens(tokens["exact"], tokens["tensor_core"]),
                   "graph_replays": {mode: summaries[mode].get("graph_replays") for mode in MODES},
                   "decode_mode_used": {mode: summaries[mode].get("decode_mode_used") for mode in MODES}}
            rows.append(row)
            (args.output_dir / "tensor_core_decode.json").write_text(json.dumps(rows, indent=1) + "\n")
            print(f"batch {batch} prompt {prompt_tokens}: exact {row['exact_tpot_ms']:.2f} ms, tensor-core "
                  f"{row['tensor_core_tpot_ms']:.2f} ms ({1 - row['tensor_core_tpot_ms'] / row['exact_tpot_ms']:+.1%}), "
                  f"identical {row['tokens']['identical']}/{row['tokens']['requests']}, modes {row['decode_mode_used']}", flush=True)
    report = format_report(rows)
    (args.output_dir / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
