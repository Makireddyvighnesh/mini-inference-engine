"""Measure exact-token prompt-prefix reuse against a cold paged request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter, perf_counter_ns

import torch

from minillm_l4.benchmarks.core.harness import RequestEventRecorder
from minillm_l4.benchmarks.core.schemas import RequestSpec
from minillm_l4.benchmarks.runners.huggingface_baseline import (
    BASELINE_BUCKETS,
    MODEL_ID,
    MODEL_REVISION,
    build_hf_workload,
    load_qwen_fp8,
)
from minillm_l4.benchmarks.runners.prefix_cache import PrefixCachedPagedRunner


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=[row[0] for row in BASELINE_BUCKETS], default="short")
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-pairs", type=int, default=1)
    parser.add_argument(
        "--divergent-last-token-check", action="store_true",
        help="Also compare warm and cold outputs after changing the prompt's final token.",
    )
    parser.add_argument("--fp8-kernel-path", choices=("auto", "sm89"), default="sm89")
    parser.add_argument("--warm-prefill-backend", choices=("sdpa", "paged"), default="sdpa")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "results/prefix_cache/summary.json",
    )
    return parser.parse_args()


def _run_one(
    runner: PrefixCachedPagedRunner,
    request: RequestSpec,
    *,
    device: torch.device,
) -> dict[str, object]:
    recorder = RequestEventRecorder(request.request_id, run_started_ns=perf_counter_ns())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()
    outcome = runner((request,), (recorder,))[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (perf_counter() - started) * 1000
    first = next(event for event in recorder.events if event.event == "first_token_ready")
    prefill = next(event for event in recorder.events if event.event == "prefill_start")
    return {
        "request_id": request.request_id,
        "generated_token_ids": list(outcome.generated_token_ids),
        "ttft_ms": (first.timestamp_ns - prefill.timestamp_ns) / 1_000_000,
        "e2e_ms": elapsed_ms,
        "cached_prefix_tokens": outcome.metadata["prefix_cache"]["cached_prefix_tokens"],
        "computed_prefill_tokens": outcome.metadata["prefix_cache"]["computed_prefill_tokens"],
    }


def main() -> None:
    args = parse_args()
    if args.output_tokens < 1 or args.block_size < 1 or args.num_blocks < 1:
        raise ValueError("output length and block dimensions must be positive")
    if args.repetitions < 1 or args.warmup_pairs < 0:
        raise ValueError("repetitions must be positive and warmup pairs non-negative")
    bucket = next(row for row in BASELINE_BUCKETS if row[0] == args.workload)
    if args.output_tokens > bucket[2]:
        raise ValueError("output length exceeds the saved reference corpus")

    bundle = load_qwen_fp8(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        model_path=args.model_path,
        device=args.device,
        local_files_only=not args.allow_download,
        fp8_kernel_path=args.fp8_kernel_path,
    )
    workload = build_hf_workload(
        bundle.tokenizer,
        PROJECT_ROOT / "data/synthetic/workloads_v1.jsonl",
        bucket_name=bucket[0],
        prompt_tokens=bucket[1],
        output_tokens=args.output_tokens,
        count=1,
        seed=17,
        device=args.device,
    )
    source = workload.requests[0]
    reference = json.loads(
        (PROJECT_ROOT / f"results/phase1/references_baseline/{bucket[0]}.json").read_text()
    )["requests"][source.request_id]
    if reference["prompt_sha256"] != source.prompt_sha256:
        raise RuntimeError("saved reference prompt digest does not match the current tokenizer")
    expected = tuple(reference["generated_token_ids"][: args.output_tokens])
    runner = PrefixCachedPagedRunner(
        bundle.model,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        device=args.device,
        warm_prefill_backend=args.warm_prefill_backend,
    )
    pairs: list[dict[str, object]] = []
    divergent_check: dict[str, object] | None = None
    try:
        for iteration in range(args.warmup_pairs + args.repetitions):
            runner.close()
            results = []
            for mode in ("cold", "warm"):
                request = RequestSpec(
                    f"prefix-{iteration}-{mode}",
                    source.prompt_token_ids,
                    args.output_tokens,
                )
                results.append(_run_one(runner, request, device=torch.device(args.device)))
            if iteration >= args.warmup_pairs:
                cold, warm = results
                pairs.append({
                    "cold": cold,
                    "warm": warm,
                    "cold_matches_reference": tuple(cold["generated_token_ids"]) == expected,
                    "warm_matches_reference": tuple(warm["generated_token_ids"]) == expected,
                })
        if args.divergent_last_token_check:
            changed = (
                *source.prompt_token_ids[:-1],
                (source.prompt_token_ids[-1] + 1) % int(bundle.tokenizer.vocab_size),
            )
            runner.close()
            _run_one(
                runner,
                RequestSpec("prefix-divergent-prime", source.prompt_token_ids, args.output_tokens),
                device=torch.device(args.device),
            )
            warm = _run_one(
                runner,
                RequestSpec("prefix-divergent-warm", changed, args.output_tokens),
                device=torch.device(args.device),
            )
            runner.close()
            cold = _run_one(
                runner,
                RequestSpec("prefix-divergent-cold", changed, args.output_tokens),
                device=torch.device(args.device),
            )
            divergent_check = {
                "cached_prefix_tokens": warm["cached_prefix_tokens"],
                "warm_token_ids": warm["generated_token_ids"],
                "cold_token_ids": cold["generated_token_ids"],
                "exact_match": warm["generated_token_ids"] == cold["generated_token_ids"],
            }
    finally:
        runner.close()

    passed = all(
        pair["cold_matches_reference"] and pair["warm_matches_reference"]
        for pair in pairs
    ) and (divergent_check is None or bool(divergent_check["exact_match"]))
    cold_ttft = median(pair["cold"]["ttft_ms"] for pair in pairs)
    warm_ttft = median(pair["warm"]["ttft_ms"] for pair in pairs)
    result = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "workload": args.workload,
        "prompt_tokens": bucket[1],
        "output_tokens": args.output_tokens,
        "block_size": args.block_size,
        "num_blocks": args.num_blocks,
        "warm_prefill_backend": args.warm_prefill_backend,
        "warmup_pairs": args.warmup_pairs,
        "repetitions": args.repetitions,
        "timing_scope": (
            "steady_state_after_warmup"
            if args.warmup_pairs else "includes_first_call_initialization"
        ),
        "exact_token_correctness": passed,
        "cold_ttft_p50_ms": cold_ttft,
        "warm_ttft_p50_ms": warm_ttft,
        "ttft_speedup": cold_ttft / warm_ttft,
        "cold_e2e_p50_ms": median(pair["cold"]["e2e_ms"] for pair in pairs),
        "warm_e2e_p50_ms": median(pair["warm"]["e2e_ms"] for pair in pairs),
        "pairs": pairs,
        "divergent_last_token_check": divergent_check,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "pairs"}, indent=2))
    if not passed:
        raise SystemExit("exact token correctness failed; inspect the saved pairs")


if __name__ == "__main__":
    main()
