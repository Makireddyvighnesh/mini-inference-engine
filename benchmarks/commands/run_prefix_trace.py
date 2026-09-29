"""Benchmark prefix reuse under continuous arrivals against an uncached control."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import random
from pathlib import Path
from typing import Any

import torch

from minillm_l4.benchmarks.core.harness import BenchmarkHarness, write_events_jsonl, write_result
from minillm_l4.benchmarks.core.schemas import HarnessConfig, RequestSpec, WorkloadSpec
from minillm_l4.benchmarks.runners.continuous_prefix import ContinuousPrefixPagedRunner
from minillm_l4.benchmarks.runners.huggingface_baseline import (
    MODEL_ID, MODEL_REVISION, load_qwen_fp8,
)
from minillm_l4.configs.loader import load_yaml_config
from minillm_l4.engine.generation.manual import manual_greedy_generate


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/workloads/qwen3_fp8_prefix.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/prefix_cache/continuous")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--reuse-rates", type=float, nargs="+", default=None)
    parser.add_argument("--request-count", type=int, default=None)
    parser.add_argument("--max-batch-size", type=int, default=None)
    parser.add_argument("--output-tokens", type=int, default=None)
    parser.add_argument("--decode-backend", choices=("auto", "torch", "triton"), default=None)
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--warmup-repetitions", type=int, default=None)
    parser.add_argument("--no-gpu-sampling", action="store_true")
    parser.add_argument(
        "--record-logit-gaps",
        action="store_true",
        help="record each decode step's top-2 logit gap (adds per-step host work)",
    )
    return parser.parse_args()


def build_shared_prefix_workload(
    tokenizer: Any,
    *,
    rate: float,
    count: int,
    prefix_tokens: int,
    suffix_tokens: int,
    output_tokens: int,
    arrival_interval_ms: float,
    seed: int,
    device: str,
) -> WorkloadSpec:
    """Create varied suffixes and output lengths with controlled prefix reuse."""

    if not 0 <= rate <= 1 or count < 2 or prefix_tokens < 1 or suffix_tokens < 1:
        raise ValueError("invalid reuse rate, request count, or prompt lengths")
    if output_tokens < 2 or arrival_interval_ms < 0:
        raise ValueError("output length must exceed one and arrivals must be non-negative")
    rng = random.Random(seed)
    vocab_size = int(tokenizer.vocab_size)
    if vocab_size < 128:
        raise ValueError("tokenizer vocabulary is too small for this workload")
    base = tokenizer.encode(
        "You are a careful technical assistant. Explain the answer clearly.",
        add_special_tokens=False,
    )
    if not base:
        raise ValueError("tokenizer produced an empty shared prefix")
    shared = tuple((base * ((prefix_tokens // len(base)) + 1))[:prefix_tokens])
    hot_count = round(count * rate)
    requests: list[RequestSpec] = []
    for index in range(count):
        hot = index < hot_count
        variant = rng.randrange(1_000_000)
        if hot:
            prefix = shared
        else:
            unique_text = (
                f"{index + 1}: Specialized system instruction {variant}. "
                "Give a precise, grounded answer to the user's technical question. "
            )
            unique_tokens = tokenizer.encode(unique_text, add_special_tokens=False)
            if not unique_tokens:
                raise ValueError("tokenizer produced an empty unique prefix")
            prefix = tuple(
                (unique_tokens * ((prefix_tokens // len(unique_tokens)) + 1))[:prefix_tokens]
            )
        suffix_length = suffix_tokens * (1 + index % 2)
        user_text = (
            f"User question {index + 1}, case {variant}: Explain how prefix caching "
            "changes prefill latency when several requests share a system prompt. "
            "Give one concrete example. "
        )
        user_tokens = tokenizer.encode(user_text, add_special_tokens=False)
        if not user_tokens:
            raise ValueError("tokenizer produced an empty user suffix")
        suffix = tuple(
            (user_tokens * ((suffix_length // len(user_tokens)) + 1))[:suffix_length]
        )
        requests.append(RequestSpec(
            request_id=f"prefix-{index:03d}",
            prompt_token_ids=prefix + suffix,
            max_new_tokens=output_tokens if index % 2 == 0 else max(2, output_tokens // 2),
            scheduled_arrival_ms=index * arrival_interval_ms,
            category="shared_prefix" if hot else "unique_prefix",
        ))
    return WorkloadSpec(
        name=f"prefix-reuse-{int(rate * 100):03d}", seed=seed,
        requests=tuple(requests), model_id=MODEL_ID,
        model_revision=MODEL_REVISION, dtype="fp8", device=device,
        arrival_pattern="fixed_rate",
        metadata={
            "target_reuse_rate": rate,
            "shared_prefix_tokens": prefix_tokens,
            "suffix_tokens": suffix_tokens,
            "first_shared_request_is_cold": True,
        },
    )


def _output_rows(result: Any) -> list[list[tuple[str, tuple[int, ...]]]]:
    return [
        [
            (row["outcome"]["status"], tuple(row["outcome"]["generated_token_ids"]))
            for row in run["requests"]
        ]
        for run in result.runs
    ]


def main() -> None:
    args = parse_args()
    payload = load_yaml_config(args.config, expected_phase=7, label="Prefix cache")
    model_config = payload["model"]
    workload_config = payload["workload"]
    engine_config = payload["engine"]
    benchmark_config = payload["benchmark"]
    if model_config["id"] != MODEL_ID or model_config["revision"] != MODEL_REVISION:
        raise ValueError("prefix trace must use the pinned model revision")
    device = str(model_config["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; cannot run the L4 prefix benchmark")
    rates = (
        args.reuse_rates if args.reuse_rates is not None
        else [float(value) for value in workload_config["reuse_rates"]]
    )
    count = (
        int(workload_config["request_count"])
        if args.request_count is None else args.request_count
    )
    repetitions = (
        int(benchmark_config["repetitions"])
        if args.repetitions is None else args.repetitions
    )
    warmups = (
        int(benchmark_config["warmup_repetitions"])
        if args.warmup_repetitions is None else args.warmup_repetitions
    )
    if any(not 0 <= rate <= 1 for rate in rates):
        raise ValueError("reuse rates must be in [0, 1]")
    if repetitions < 1 or warmups < 0:
        raise ValueError("repetitions must be positive and warmups non-negative")
    max_batch_size = (
        int(engine_config["max_batch_size"])
        if args.max_batch_size is None else args.max_batch_size
    )
    decode_backend = (
        str(engine_config["decode_backend"])
        if args.decode_backend is None else args.decode_backend
    )
    if max_batch_size < 1:
        raise ValueError("max batch size must be positive")
    output_tokens = (
        int(workload_config["output_tokens"])
        if args.output_tokens is None else args.output_tokens
    )
    bundle = load_qwen_fp8(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        model_path=args.model_path,
        device=device,
        local_files_only=bool(model_config.get("local_files_only", True)) and not args.allow_download,
        fp8_kernel_path=str(model_config.get("fp8_kernel_path", "auto")),
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "model": bundle.metadata(),
        "configuration": payload,
        "overrides": {
            "max_batch_size": max_batch_size,
            "decode_backend": decode_backend,
            "request_count": count,
            "output_tokens": output_tokens,
            "repetitions": repetitions,
            "warmup_repetitions": warmups,
        },
        "config_path": str(args.config),
        "workloads": [],
    }
    harness_config = HarnessConfig(
        warmup_repetitions=warmups, repetitions=repetitions,
        respect_arrival_schedule=True,
        sample_interval_seconds=float(benchmark_config["sample_interval_seconds"]),
        collect_gpu=bool(benchmark_config["collect_gpu"]) and not args.no_gpu_sampling,
        collect_system_telemetry=bool(benchmark_config["collect_system_telemetry"]),
        runner_name="continuous_paged_prefix", timing_mode="wall",
        seed=int(workload_config["seed"]),
        timer_overhead_iterations=int(benchmark_config["timer_overhead_iterations"]),
    )
    all_correct = True
    for rate in rates:
        workload = build_shared_prefix_workload(
            bundle.tokenizer, rate=rate, count=count,
            prefix_tokens=int(workload_config["shared_prefix_tokens"]),
            suffix_tokens=int(workload_config["suffix_tokens"]),
            output_tokens=output_tokens,
            arrival_interval_ms=float(workload_config["arrival_interval_ms"]),
            seed=int(workload_config["seed"]), device=device,
        )
        label = f"reuse_{int(rate * 100):03d}"
        (output_dir / f"{label}_workload.json").write_text(
            json.dumps(workload.to_dict(), indent=2, sort_keys=True) + "\n"
        )
        results: dict[str, Any] = {}
        run_summaries: dict[str, list[dict[str, Any]]] = {}
        for mode in ("uncached", "cached"):
            runner = ContinuousPrefixPagedRunner(
                bundle.model,
                block_size=int(engine_config["block_size"]),
                num_blocks=int(engine_config["num_blocks"]),
                max_entries=int(engine_config["max_entries"]),
                max_batch_size=max_batch_size,
                max_prefill_tokens=int(engine_config["max_prefill_tokens"]),
                device=device,
                decode_backend=decode_backend,
                decode_mode=str(engine_config["decode_mode"]),
                enable_prefix=mode == "cached",
                record_logit_gaps=args.record_logit_gaps,
            )
            try:
                result = BenchmarkHarness(
                    dataclasses.replace(harness_config, runner_name=runner.runner_name),
                    benchmark_name=f"prefix_trace_{mode}",
                ).run_trace(workload, runner)
                results[mode] = result
                write_result(result, output_dir / f"{label}_{mode}.json")
                write_events_jsonl(result, output_dir / f"{label}_{mode}_events.jsonl")
            finally:
                runner.close()
            run_summaries[mode] = runner.run_summaries[-repetitions:]
            del runner
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
        cached_vs_uncached = _output_rows(results["uncached"]) == _output_rows(results["cached"])
        cached_vs_uncached = cached_vs_uncached and all(
            status == "completed" for run in _output_rows(results["cached"])
            for status, _ in run
        )
        reference_tokens: dict[str, list[int]] = {}
        reference_matches = True
        reference_mismatch_ids: list[str] = []
        for index, request in enumerate(workload.requests):
            reference = manual_greedy_generate(
                bundle.model,
                {
                    "input_ids": torch.tensor(
                        [request.prompt_token_ids], dtype=torch.long, device=device
                    ),
                    "attention_mask": torch.ones(
                        (1, request.prompt_tokens), dtype=torch.long, device=device
                    ),
                },
                output_tokens=request.max_new_tokens,
            )
            token_ids = [int(value) for value in reference.row(0).tolist()]
            reference_tokens[request.request_id] = token_ids
            request_matches = all(
                run["requests"][index]["outcome"]["generated_token_ids"] == token_ids
                for run in results["cached"].runs
            )
            reference_matches = reference_matches and request_matches
            if not request_matches:
                reference_mismatch_ids.append(request.request_id)
        correct = cached_vs_uncached and reference_matches
        all_correct = all_correct and correct
        cache_runs = run_summaries["cached"]
        record = {
            "reuse_rate": rate,
            "correctness": bool(correct),
            "cached_vs_uncached_exact": bool(cached_vs_uncached),
            "dense_reference_exact": bool(reference_matches),
            "dense_reference_mismatch_ids": reference_mismatch_ids,
            "dense_reference_token_ids": reference_tokens,
            "uncached": results["uncached"].summary,
            "cached": results["cached"].summary,
            "cache_runs": cache_runs,
        }
        manifest["workloads"].append(record)
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        uncached_ttft = record["uncached"]["metrics"]["ttft_ms"]["p50"]
        cached_ttft = record["cached"]["metrics"]["ttft_ms"]["p50"]
        hit_rate = sum(item["hits"] for item in cache_runs) / max(
            1, sum(item["hits"] + item["misses"] for item in cache_runs)
        )
        print(
            f"reuse={rate:.0%} hit_rate={hit_rate:.1%} "
            f"TTFT_P50={uncached_ttft:.1f}->{cached_ttft:.1f} ms "
            f"cached_vs_uncached={'pass' if cached_vs_uncached else 'FAIL'} "
            f"dense_reference={'pass' if reference_matches else 'FAIL'}",
            flush=True,
        )
    print(f"Manifest: {output_dir / 'manifest.json'}")
    if not all_correct:
        raise RuntimeError("one or more correctness gates failed; inspect the manifest")


if __name__ == "__main__":
    main()
