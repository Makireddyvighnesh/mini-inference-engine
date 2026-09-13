"""Compare contiguous KV allocation with fixed-block paged allocation."""

from __future__ import annotations

import argparse
import json
import math
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
from minillm_l4.benchmarks.runners.kv_cache import KvCacheBatchRunner, write_kv_result
from minillm_l4.benchmarks.runners.paged_cuda_graph import PagedCudaGraphBatchRunner
from minillm_l4.benchmarks.runners.paged_kv import (
    PagedAttentionBatchRunner,
    PagedHybridBatchRunner,
    PagedKvBatchRunner,
    write_paged_result,
)
from minillm_l4.configs.loader import load_yaml_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/workloads/qwen3_fp8_paged.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/paged_kv"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "results/phase1/references_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare contiguous KV storage with fixed-block paged storage."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--workload", choices=("short", "medium", "long", "all"), default="all"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=(
            "contiguous",
            "paged_graph",
            "paged",
            "paged_gather",
            "paged_direct",
            "paged_hybrid",
        ),
        default=None,
    )
    parser.add_argument("--block-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--capacity-token-slots", type=int, default=None)
    parser.add_argument(
        "--decode-backend",
        choices=("auto", "torch", "triton"),
        default=None,
        help="Paged decode implementation; auto selects Triton on supported CUDA inputs.",
    )
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--warmup-repetitions", type=int, default=None)
    parser.add_argument("--no-gpu-sampling", action="store_true")
    parser.add_argument("--no-system-telemetry", action="store_true")
    return parser.parse_args()


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == PROJECT_ROOT.name:
        return PROJECT_ROOT.parent / path
    return PROJECT_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    payload = load_yaml_config(path, expected_phase=6, label="Paged KV")
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("paged KV config must contain a model object")
    if model.get("id") != MODEL_ID or model.get("revision") != MODEL_REVISION:
        raise ValueError("paged KV config must use the pinned Qwen model revision")
    if model.get("precision") != "fp8":
        raise ValueError("paged KV benchmark requires checkpoint-native FP8")
    cache = payload.get("cache")
    if not isinstance(cache, Mapping):
        raise ValueError("paged KV config must contain a cache object")
    modes = cache.get("modes")
    if not isinstance(modes, list) or not modes:
        raise ValueError("cache.modes must be a non-empty list")
    if any(
        mode
        not in {
            "contiguous",
            "paged_graph",
            "paged",
            "paged_gather",
            "paged_direct",
            "paged_hybrid",
        }
        for mode in modes
    ):
        raise ValueError(
            "cache.modes values must be contiguous, paged_graph, paged, paged_gather, "
            "paged_direct, or paged_hybrid"
        )
    block_sizes = cache.get("block_sizes")
    if not isinstance(block_sizes, list) or not block_sizes:
        raise ValueError("cache.block_sizes must be a non-empty list")
    if any(int(value) < 1 for value in block_sizes):
        raise ValueError("cache.block_sizes must be positive")
    if int(cache.get("capacity_token_slots", 0)) < 1:
        raise ValueError("cache.capacity_token_slots must be positive")
    return payload


def _selected_buckets(selection: str, configured_lengths: list[int]) -> list[tuple[str, int, int]]:
    by_length = {bucket[1]: bucket for bucket in BASELINE_BUCKETS}
    unknown = [value for value in configured_lengths if value not in by_length]
    if unknown:
        raise ValueError(f"unsupported prompt lengths: {unknown}")
    selected = [by_length[value] for value in configured_lengths]
    if selection == "all":
        return selected
    return [bucket for bucket in selected if bucket[0] == selection]


def _display(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _harness_config(
    benchmark: Mapping[str, Any],
    *,
    repetitions: int,
    warmups: int,
    seed: int,
    runner_name: str,
    args: argparse.Namespace,
) -> HarnessConfig:
    return HarnessConfig(
        warmup_repetitions=warmups,
        repetitions=repetitions,
        respect_arrival_schedule=False,
        sample_interval_seconds=float(benchmark["sample_interval_seconds"]),
        collect_gpu=bool(benchmark["collect_gpu"]) and not args.no_gpu_sampling,
        collect_system_telemetry=(
            bool(benchmark["collect_system_telemetry"])
            and not args.no_system_telemetry
        ),
        runner_name=runner_name,
        timing_mode="wall",
        seed=seed,
        timer_overhead_iterations=int(benchmark["timer_overhead_iterations"]),
    )


def main() -> None:
    args = parse_args()
    config_path = _project_path(args.config)
    payload = load_config(config_path)
    model_config = payload["model"]
    workload_config = payload["workloads"]
    cache_config = payload["cache"]
    benchmark_config = payload["benchmark"]

    modes = list(cache_config["modes"] if args.modes is None else args.modes)
    block_sizes = [
        int(value)
        for value in (
            cache_config["block_sizes"] if args.block_sizes is None else args.block_sizes
        )
    ]
    batch_sizes = [
        int(value)
        for value in (
            workload_config["batch_sizes"]
            if args.batch_sizes is None
            else args.batch_sizes
        )
    ]
    if any(value < 1 for value in block_sizes + batch_sizes):
        raise ValueError("block sizes and batch sizes must be positive")
    paged_modes = tuple(
        mode
        for mode in modes
        if mode in {
            "paged",
            "paged_graph",
            "paged_gather",
            "paged_direct",
            "paged_hybrid",
        }
    )
    if paged_modes and not block_sizes:
        raise ValueError("paged modes require at least one block size")
    prompt_lengths = [int(value) for value in workload_config["prompt_lengths"]]
    selected = _selected_buckets(args.workload, prompt_lengths)
    if not selected:
        raise ValueError(f"no prompt lengths match workload {args.workload}")
    count = int(workload_config["count"] if args.count is None else args.count)
    if count < 1:
        raise ValueError("count must be positive")
    capacity_token_slots = int(
        cache_config["capacity_token_slots"]
        if args.capacity_token_slots is None
        else args.capacity_token_slots
    )
    if capacity_token_slots < 1:
        raise ValueError("capacity-token-slots must be positive")
    decode_backend = str(
        cache_config.get("decode_backend", "auto")
        if args.decode_backend is None
        else args.decode_backend
    )
    if decode_backend not in {"auto", "torch", "triton"}:
        raise ValueError("cache.decode_backend must be auto, torch, or triton")

    output_dir = _project_path(args.output_dir)
    reference_dir = _project_path(args.reference_dir)
    dataset_path = _project_path(str(workload_config["dataset"]))
    device = str(model_config["device"] if args.device is None else args.device)
    local_files_only = bool(model_config.get("local_files_only", True)) and not args.allow_download

    load_started = perf_counter()
    bundle = load_qwen_fp8(
        model_id=str(model_config["id"]),
        revision=str(model_config["revision"]),
        model_path=_project_path(args.model_path) if args.model_path is not None else None,
        device=device,
        local_files_only=local_files_only,
        fp8_fallback_dtype=str(model_config.get("fp8_fallback_dtype", "auto")),
        fp8_kernel_path=str(model_config.get("fp8_kernel_path", "auto")),
    )
    initialization = {
        "model_load_wall_time_ms": (perf_counter() - load_started) * 1000.0,
        **bundle.metadata(),
    }
    repetitions = int(
        benchmark_config["repetitions"]
        if args.repetitions is None
        else args.repetitions
    )
    warmups = int(
        benchmark_config["warmup_repetitions"]
        if args.warmup_repetitions is None
        else args.warmup_repetitions
    )
    if repetitions < 1 or warmups < 0:
        raise ValueError("repetitions must be positive and warmups non-negative")
    if "paged_graph" in modes and warmups < 1:
        raise ValueError(
            "paged_graph requires at least one warm-up so capture cost is "
            "kept separate from steady-state metrics"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "project": "MiniLLM-L4",
        "phase": 6,
        "status": "running",
        "config": str(config_path),
        "initialization": initialization,
        "configuration": {
            "modes": modes,
            "block_sizes": block_sizes,
            "capacity_token_slots": capacity_token_slots,
            "decode_backend": decode_backend,
            "batch_sizes": batch_sizes,
            "count": count,
            "repetitions": repetitions,
            "warmup_repetitions": warmups,
            "dataset": str(dataset_path),
            "sampling": "greedy argmax",
        },
        "workloads": [],
    }

    for bucket_name, prompt_tokens, default_output_tokens in selected:
        output_tokens = int(
            workload_config["output_tokens"].get(str(prompt_tokens), default_output_tokens)
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
            if "contiguous" in modes:
                runner = KvCacheBatchRunner(
                    bundle.model,
                    mode="contiguous",
                    device=device,
                    logits_mode=str(benchmark_config.get("logits_mode", "last")),
                    pad_token_id=0,
                )
                result = BenchmarkHarness(
                    _harness_config(
                        benchmark_config,
                        repetitions=repetitions,
                        warmups=warmups,
                        seed=int(workload.seed),
                        runner_name="contiguous_dynamic_kv",
                        args=args,
                    ),
                    benchmark_name="minillm_l4_paged_comparison",
                ).run_batched(workload, batch_size, runner)
                correctness = verify_or_write_reference(
                    result,
                    reference_dir / f"{bucket_name}.json",
                )
                result_path = output_dir / f"contiguous_{bucket_name}_b{batch_size}.json"
                events_path = output_dir / f"contiguous_{bucket_name}_b{batch_size}_events.jsonl"
                write_kv_result(
                    result,
                    result_path,
                    model_metadata=initialization,
                    correctness=correctness,
                    mode="contiguous",
                    cache_snapshot=runner.last_cache_snapshot,
                )
                write_events_jsonl(result, events_path)
                manifest["workloads"].append(
                    {
                        "mode": "contiguous",
                        "name": bucket_name,
                        "prompt_tokens": prompt_tokens,
                        "output_tokens": output_tokens,
                        "batch_size": batch_size,
                        "result": str(result_path),
                        "events": str(events_path),
                        "reference": str(reference_dir / f"{bucket_name}.json"),
                        "correctness": correctness,
                        "summary": result.summary,
                    }
                )
                print(
                    f"contiguous {bucket_name:6s} batch={batch_size:<2d} "
                    f"TTFT_P50={_display(result.summary['metrics']['ttft_ms'].get('median'))} ms "
                    f"TPOT_P50={_display(result.summary['metrics']['tpot_ms'].get('median'))} ms "
                    f"correctness={correctness['status']}"
                )

            for paged_mode in paged_modes:
                for block_size in block_sizes:
                    num_blocks = math.ceil(capacity_token_slots / block_size)
                    is_direct = paged_mode == "paged_direct"
                    is_hybrid = paged_mode == "paged_hybrid"
                    is_graph = paged_mode == "paged_graph"
                    runner_class = (
                        PagedCudaGraphBatchRunner
                        if is_graph
                        else PagedAttentionBatchRunner
                        if is_direct
                        else PagedHybridBatchRunner
                        if is_hybrid
                        else PagedKvBatchRunner
                    )
                    runner_kwargs = {
                        "block_size": block_size,
                        "num_blocks": num_blocks,
                        "device": device,
                        "logits_mode": str(
                            benchmark_config.get("logits_mode", "last")
                        ),
                    }
                    if is_direct or is_hybrid or is_graph:
                        runner_kwargs["decode_backend"] = decode_backend
                    runner = runner_class(bundle.model, **runner_kwargs)
                    runner_name = runner.runner_name
                    result = BenchmarkHarness(
                        _harness_config(
                            benchmark_config,
                            repetitions=repetitions,
                            warmups=warmups,
                            seed=int(workload.seed),
                            runner_name=runner_name,
                            args=args,
                        ),
                        benchmark_name="minillm_l4_paged_comparison",
                    ).run_batched(workload, batch_size, runner)
                    correctness = verify_or_write_reference(
                        result,
                        reference_dir / f"{bucket_name}.json",
                    )
                    result_prefix = (
                        "paged_direct"
                        if is_direct
                        else "paged_graph"
                        if is_graph
                        else "paged_hybrid"
                        if is_hybrid
                        else "paged"
                    )
                    result_path = (
                        output_dir
                        / f"{result_prefix}_s{block_size}_{bucket_name}_b{batch_size}.json"
                    )
                    events_path = (
                        output_dir
                        / f"{result_prefix}_s{block_size}_{bucket_name}_b{batch_size}_events.jsonl"
                    )
                    write_paged_result(
                        result,
                        result_path,
                        model_metadata=initialization,
                        correctness=correctness,
                        block_size=block_size,
                        num_blocks=num_blocks,
                        cache_snapshot=runner.last_cache_snapshot,
                    )
                    write_events_jsonl(result, events_path)
                    snapshot = runner.last_cache_snapshot or {}
                    manifest["workloads"].append(
                        {
                            "mode": paged_mode,
                            "block_size": block_size,
                            "num_blocks": num_blocks,
                            "name": bucket_name,
                            "prompt_tokens": prompt_tokens,
                            "output_tokens": output_tokens,
                            "batch_size": batch_size,
                            "result": str(result_path),
                            "events": str(events_path),
                            "reference": str(reference_dir / f"{bucket_name}.json"),
                            "correctness": correctness,
                            "cache_snapshot": snapshot,
                            "summary": result.summary,
                        }
                    )
                    display_mode = (
                        "paged-direct"
                        if is_direct
                        else "paged-graph"
                        if is_graph
                        else "paged-hybrid"
                        if is_hybrid
                        else "paged-gather"
                    )
                    print(
                        f"{display_mode:12s} {bucket_name:6s} block={block_size:<3d} "
                        f"batch={batch_size:<2d} "
                        f"TTFT_P50={_display(result.summary['metrics']['ttft_ms'].get('median'))} ms "
                        f"TPOT_P50={_display(result.summary['metrics']['tpot_ms'].get('median'))} ms "
                        f"waste={float(snapshot.get('internal_fragmentation', 0.0)):.3f} "
                        f"correctness={correctness['status']}"
                    )

    manifest["status"] = (
        "completed"
        if all(item["correctness"]["status"] == "pass" for item in manifest["workloads"])
        else "failed_correctness"
    )
    manifest_path = output_dir / "paged_kv_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    if manifest["status"] != "completed":
        raise RuntimeError("paged KV correctness checks failed; see the manifest")


if __name__ == "__main__":
    main()
