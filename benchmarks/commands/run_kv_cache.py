"""Compare explicit contiguous KV reuse with full-prefix recomputation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from minillm_l4.benchmarks.core.harness import BenchmarkHarness, write_events_jsonl
from minillm_l4.benchmarks.core.schemas import HarnessConfig
from minillm_l4.benchmarks.core.workloads import save_workload
from minillm_l4.benchmarks.runners.huggingface_baseline import (
    BASELINE_BUCKETS,
    MODEL_ID,
    MODEL_REVISION,
    build_hf_workload,
    load_qwen_fp8,
    verify_or_write_reference,
)
from minillm_l4.benchmarks.runners.kv_cache import (
    CACHE_MODES,
    KvCacheBatchRunner,
    write_kv_result,
)
from minillm_l4.configs.loader import load_yaml_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/workloads/qwen3_fp8_kv_cache.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/kv_cache"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "results/phase1/references_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare explicit KV reuse against full-prefix recomputation."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--workload",
        choices=("short", "medium", "long", "all"),
        default="all",
    )
    parser.add_argument("--modes", nargs="+", choices=CACHE_MODES, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=None)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--warmup-repetitions", type=int, default=None)
    parser.add_argument("--cache-capacity-tokens", type=int, default=None)
    parser.add_argument("--no-gpu-sampling", action="store_true")
    parser.add_argument("--no-system-telemetry", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = load_yaml_config(path, expected_phase=3, label="KV cache")
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("KV cache config must contain a model object")
    if model.get("id") != MODEL_ID or model.get("revision") != MODEL_REVISION:
        raise ValueError("KV cache config must use the pinned Qwen model revision")
    if model.get("precision") != "fp8":
        raise ValueError("KV cache benchmark requires checkpoint-native FP8")
    cache = payload.get("cache")
    if not isinstance(cache, Mapping):
        raise ValueError("KV cache config must contain a cache object")
    modes = cache.get("modes")
    if not isinstance(modes, list) or not modes:
        raise ValueError("cache.modes must be a non-empty list")
    if any(mode not in CACHE_MODES for mode in modes):
        raise ValueError(f"cache.modes values must be in {CACHE_MODES}")
    capacity = cache.get("capacity", "request")
    if capacity != "request" and int(capacity) < 1:
        raise ValueError("cache.capacity must be 'request' or a positive integer")
    return payload


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == PROJECT_ROOT.name:
        return PROJECT_ROOT.parent / path
    return PROJECT_ROOT / path


def _selected_buckets(
    selection: str,
    configured_lengths: list[int],
) -> list[tuple[str, int, int]]:
    by_length = {bucket[1]: bucket for bucket in BASELINE_BUCKETS}
    unknown = [value for value in configured_lengths if value not in by_length]
    if unknown:
        raise ValueError(f"Unsupported prompt lengths: {unknown}")
    selected = [by_length[value] for value in configured_lengths]
    if selection == "all":
        return selected
    return [bucket for bucket in selected if bucket[0] == selection]


def _display(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _print_summary(
    mode: str,
    workload_name: str,
    batch_size: int,
    summary: Mapping[str, Any],
    correctness: Mapping[str, Any],
) -> None:
    metrics = summary["metrics"]
    print(
        f"{mode:10s} {workload_name:6s} batch={batch_size:<2d} "
        f"requests/run={summary['run_summaries'][0]['completed_requests']:3d} "
        f"repetitions={summary['repetitions']} "
        f"TTFT_P50={_display(metrics['ttft_ms'].get('median'))} ms "
        f"TPOT_P50={_display(metrics['tpot_ms'].get('median'))} ms "
        f"TPS_P50={_display(summary['tokens_per_second'].get('median'))} "
        f"correctness={correctness['status']}"
    )


def _ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator is None or float(denominator) == 0:
        return None
    return float(numerator) / float(denominator)


def _comparison(
    contiguous: Mapping[str, Any],
    recompute: Mapping[str, Any],
) -> dict[str, Any]:
    cached_tpot = contiguous["metrics"]["tpot_ms"].get("median")
    recompute_tpot = recompute["metrics"]["tpot_ms"].get("median")
    cached_tps = contiguous["tokens_per_second"].get("median")
    recompute_tps = recompute["tokens_per_second"].get("median")
    return {
        "contiguous_tpot_ms": cached_tpot,
        "recompute_tpot_ms": recompute_tpot,
        "decode_tpot_speedup": _ratio(recompute_tpot, cached_tpot),
        "contiguous_tps": cached_tps,
        "recompute_tps": recompute_tps,
        "throughput_speedup": _ratio(cached_tps, recompute_tps),
    }


def _harness_config(
    config: Mapping[str, Any],
    workload: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[HarnessConfig, int, int]:
    repetitions = int(
        config["repetitions"] if args.repetitions is None else args.repetitions
    )
    warmups = int(
        config["warmup_repetitions"]
        if args.warmup_repetitions is None
        else args.warmup_repetitions
    )
    return (
        HarnessConfig(
            warmup_repetitions=warmups,
            repetitions=repetitions,
            respect_arrival_schedule=False,
            sample_interval_seconds=float(config["sample_interval_seconds"]),
            collect_gpu=bool(config["collect_gpu"]) and not args.no_gpu_sampling,
            collect_system_telemetry=(
                bool(config["collect_system_telemetry"])
                and not args.no_system_telemetry
            ),
            runner_name="kv_cache_comparison",
            timing_mode="wall",
            seed=int(workload["seed"]),
            timer_overhead_iterations=int(config["timer_overhead_iterations"]),
        ),
        repetitions,
        warmups,
    )


def main() -> None:
    args = parse_args()
    config_path = _project_path(args.config)
    payload = load_config(config_path)
    model_config = payload["model"]
    workload_config = payload["workloads"]
    cache_config = payload["cache"]
    benchmark_config = payload["benchmark"]

    prompt_lengths = (
        [int(value) for value in workload_config["prompt_lengths"]]
        if args.prompt_lengths is None
        else [int(value) for value in args.prompt_lengths]
    )
    batch_sizes = (
        [int(value) for value in workload_config["batch_sizes"]]
        if args.batch_sizes is None
        else [int(value) for value in args.batch_sizes]
    )
    modes = (
        [str(value) for value in cache_config["modes"]]
        if args.modes is None
        else list(args.modes)
    )
    if not prompt_lengths or any(value < 1 for value in prompt_lengths):
        raise ValueError("prompt lengths must be positive")
    if not batch_sizes or any(value < 1 for value in batch_sizes):
        raise ValueError("batch sizes must be positive")
    selected = _selected_buckets(args.workload, prompt_lengths)
    if not selected:
        raise ValueError(f"No prompt lengths match workload {args.workload}")

    configured_capacity = cache_config.get("capacity", "request")
    capacity_tokens = args.cache_capacity_tokens
    if capacity_tokens is None and configured_capacity != "request":
        capacity_tokens = int(configured_capacity)

    output_dir = _project_path(args.output_dir)
    reference_dir = _project_path(args.reference_dir)
    dataset_path = _project_path(str(workload_config["dataset"]))
    count = int(workload_config["count"] if args.count is None else args.count)
    if count < 1:
        raise ValueError("count must be positive")
    device = str(model_config["device"] if args.device is None else args.device)
    local_files_only = bool(model_config.get("local_files_only", True)) and not (
        args.allow_download
    )

    load_started = perf_counter()
    bundle = load_qwen_fp8(
        model_id=str(model_config["id"]),
        revision=str(model_config["revision"]),
        model_path=(
            _project_path(args.model_path) if args.model_path is not None else None
        ),
        device=device,
        local_files_only=local_files_only,
        fp8_fallback_dtype=str(model_config.get("fp8_fallback_dtype", "auto")),
        fp8_kernel_path=str(model_config.get("fp8_kernel_path", "auto")),
    )
    initialization = {
        "model_load_wall_time_ms": (perf_counter() - load_started) * 1000.0,
        **bundle.metadata(),
    }
    harness_config, repetitions, warmups = _harness_config(
        benchmark_config,
        workload_config,
        args,
    )
    eos_token_id = None
    if bool(benchmark_config.get("eos_stopping", False)):
        eos_token_id = getattr(bundle.tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise ValueError("EOS stopping requested but tokenizer has no EOS ID")
    pad_token_id = getattr(bundle.tokenizer, "pad_token_id", None)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "project": "MiniLLM-L4",
        "phase": 3,
        "status": "running",
        "config": str(config_path),
        "initialization": initialization,
        "configuration": {
            "modes": modes,
            "capacity_tokens": capacity_tokens or "request",
            "prompt_lengths": prompt_lengths,
            "batch_sizes": batch_sizes,
            "count": count,
            "repetitions": repetitions,
            "warmup_repetitions": warmups,
            "dataset": str(dataset_path),
            "sampling": "greedy argmax",
        },
        "workloads": [],
        "comparisons": [],
    }

    for bucket_name, prompt_tokens, default_output_tokens in selected:
        output_tokens = int(
            workload_config["output_tokens"].get(
                str(prompt_tokens),
                default_output_tokens,
            )
        )
        workload = build_hf_workload(
            bundle.tokenizer,
            dataset_path,
            bucket_name=bucket_name,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            count=max(count, max(batch_sizes)),
            seed=int(workload_config["seed"]),
            model_id=str(model_config["id"]),
            revision=str(model_config["revision"]),
            device=device,
        )
        workload_path = output_dir / f"workload_{bucket_name}.json"
        if not workload_path.exists():
            save_workload(workload, workload_path)

        for batch_size in batch_sizes:
            summaries: dict[str, Mapping[str, Any]] = {}
            for mode in modes:
                runner = KvCacheBatchRunner(
                    bundle.model,
                    mode=mode,
                    device=device,
                    logits_mode=str(benchmark_config.get("logits_mode", "last")),
                    eos_token_id=eos_token_id,
                    pad_token_id=0 if pad_token_id is None else int(pad_token_id),
                    capacity_tokens=(capacity_tokens if mode == "contiguous" else None),
                )
                result = BenchmarkHarness(
                    harness_config,
                    benchmark_name=f"minillm_l4_kv_{mode}",
                ).run_batched(workload, batch_size, runner)
                correctness = verify_or_write_reference(
                    result,
                    reference_dir / f"{bucket_name}.json",
                )
                result_path = output_dir / f"{mode}_{bucket_name}_b{batch_size}.json"
                events_path = (
                    output_dir / f"{mode}_{bucket_name}_b{batch_size}_events.jsonl"
                )
                write_kv_result(
                    result,
                    result_path,
                    model_metadata=initialization,
                    correctness=correctness,
                    mode=mode,
                    cache_snapshot=runner.last_cache_snapshot,
                )
                write_events_jsonl(result, events_path)
                summaries[mode] = result.summary
                manifest["workloads"].append(
                    {
                        "mode": mode,
                        "name": bucket_name,
                        "prompt_tokens": prompt_tokens,
                        "output_tokens": output_tokens,
                        "batch_size": batch_size,
                        "workload": str(workload_path),
                        "result": str(result_path),
                        "events": str(events_path),
                        "reference": str(reference_dir / f"{bucket_name}.json"),
                        "correctness": correctness,
                        "cache_snapshot": runner.last_cache_snapshot,
                        "summary": result.summary,
                    }
                )
                _print_summary(
                    mode,
                    bucket_name,
                    batch_size,
                    result.summary,
                    correctness,
                )

            if "contiguous" in summaries and "recompute" in summaries:
                manifest["comparisons"].append(
                    {
                        "name": bucket_name,
                        "batch_size": batch_size,
                        **_comparison(summaries["contiguous"], summaries["recompute"]),
                    }
                )

    manifest["status"] = (
        "completed"
        if all(
            item["correctness"]["status"] == "pass"
            for item in manifest["workloads"]
        )
        else "failed_correctness"
    )
    manifest_path = output_dir / "kv_cache_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")
    if manifest["status"] != "completed":
        raise RuntimeError("KV cache correctness checks failed; see the manifest")


if __name__ == "__main__":
    main()
