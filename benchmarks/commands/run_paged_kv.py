"""Compare contiguous KV allocation with fixed-block paged allocation."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
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
from minillm_l4.benchmarks.runners.packed_paged import (
    PackedPagedPrefillBatchRunner,
)
from minillm_l4.configs.loader import load_yaml_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/workloads/qwen3_fp8_paged.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/paged_kv"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "results/phase1/references_baseline"

# The regular Phase 6 matrix stops at 2,048 prompt tokens.  Keep these
# additional shapes local to the paged benchmark so the original Phase 1–5
# workload definitions and their reports remain unchanged.
LONG_CONTEXT_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("xlong", 3072, 64),
    ("xxlong", 4608, 32),
)
PAGED_CONTEXT_BUCKETS = BASELINE_BUCKETS + LONG_CONTEXT_BUCKETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare contiguous KV storage with fixed-block paged storage."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--workload",
        choices=tuple(bucket[0] for bucket in PAGED_CONTEXT_BUCKETS) + ("all",),
        default="all",
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
            "paged_packed",
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
    parser.add_argument(
        "--decode-numerics", choices=("auto", "accurate", "sdpa_compat"), default=None,
        help="Auto matches dense BF16 SDPA in hybrid/graph paths; accurate preserves FP32 page reductions.",
    )
    parser.add_argument(
        "--prefill-backend",
        choices=(
            "auto",
            "torch",
            "triton",
            "sdpa",
            "sdpa_math",
        ),
        default=None,
        help=(
            "Packed-prefill implementation; auto selects fused SDPA on CUDA, "
            "while triton keeps the educational page-walking kernel available."
        ),
    )
    parser.add_argument(
        "--graph-prefill-backend",
        choices=("auto", "packed", "dense"),
        default=None,
        help=(
            "CUDA-Graph prefill path; auto selects dense SDPA for the fixed-shape "
            "graph runner, packed is the educational flat-token path, and dense "
            "is the explicit control."
        ),
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
    parser.add_argument(
        "--trace-summary",
        action="store_true",
        help=(
            "Enable synchronized component tracing and print the measured "
            "runner timeline and aggregates; diagnostic mode adds overhead."
        ),
    )
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
            "paged_packed",
        }
        for mode in modes
    ):
        raise ValueError(
            "cache.modes values must be contiguous, paged_graph, paged, paged_gather, "
            "paged_direct, paged_hybrid, or paged_packed"
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
    by_length = {bucket[1]: bucket for bucket in PAGED_CONTEXT_BUCKETS}
    unknown = [value for value in configured_lengths if value not in by_length]
    if unknown:
        raise ValueError(f"unsupported prompt lengths: {unknown}")
    selected = [by_length[value] for value in configured_lengths]
    if selection == "all":
        return selected
    return [bucket for bucket in selected if bucket[0] == selection]


def _display(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _print_trace_summary(result: Any) -> None:
    """Print the last measured batch trace without dumping every span."""

    traces: list[Mapping[str, Any]] = []
    diagnostics: list[Mapping[str, Any]] = []
    for run in getattr(result, "runs", ()):
        for batch in run.get("runner_diagnostics", ()):
            if isinstance(batch, Mapping):
                diagnostics.append(batch)
            trace = batch.get("execution_trace")
            if isinstance(trace, Mapping):
                traces.append(trace)
    if not traces:
        print("  trace: unavailable for this runner")
        return
    trace = traces[-1]
    totals = trace.get("timing_totals", {})
    if diagnostics:
        diagnostic = diagnostics[-1]
        batch_start = float(diagnostic.get("start_ms", 0.0))
        batch_end = float(diagnostic.get("end_ms", batch_start))
        spans = trace.get("spans", ())
        span_starts = [
            float(span.get("start_ms", 0.0))
            for span in spans
            if isinstance(span, Mapping)
        ]
        span_ends = [
            float(span.get("end_ms", 0.0))
            for span in spans
            if isinstance(span, Mapping)
        ]
        trace_start = min(span_starts, default=batch_start)
        trace_end = max(span_ends, default=batch_start)
        covered_start = max(batch_start, trace_start)
        covered_end = min(batch_end, trace_end)
        covered_ms = max(0.0, covered_end - covered_start)
        outside_ms = max(0.0, (batch_end - batch_start) - covered_ms)
        print(
            "  runner invocation="
            f"{_display(diagnostic.get('batch_runner_wall_ms'))} ms "
            "trace-covered="
            f"{_display(covered_ms)} ms "
            "outside-trace="
            f"{_display(outside_ms)} ms"
        )
    print(
        "  trace window="
        f"{_display(totals.get('trace_window_ms'))} ms "
        "unattributed="
        f"{_display(totals.get('unattributed_wall_ms'))} ms"
    )
    components = trace.get("components", {})
    if not isinstance(components, Mapping):
        return
    for name, component in sorted(
        components.items(),
        key=lambda item: float(item[1].get("wall_ms_total", 0.0))
        if isinstance(item[1], Mapping)
        else 0.0,
        reverse=True,
    ):
        if not isinstance(component, Mapping):
            continue
        print(
            f"    {name:44s} "
            f"wall={_display(component.get('wall_ms_total'))} ms "
            f"device={_display(component.get('device_ms_total'))} ms "
            f"host_gap={_display(component.get('host_overhead_ms_total'))} ms "
            f"share={_display(component.get('share_of_recorded_span_wall_percent'))}%"
        )


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
            "paged_packed",
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
    decode_numerics = str(
        cache_config.get("decode_numerics", "auto")
        if args.decode_numerics is None else args.decode_numerics
    )
    if decode_numerics not in {"auto", "accurate", "sdpa_compat"}:
        raise ValueError("cache.decode_numerics must be auto, accurate, or sdpa_compat")
    prefill_backend = str(
        cache_config.get("packed_prefill_backend", "auto")
        if args.prefill_backend is None
        else args.prefill_backend
    )
    if prefill_backend not in {
        "auto",
        "torch",
        "triton",
        "sdpa",
        "sdpa_math",
    }:
        raise ValueError(
            "cache.packed_prefill_backend must be auto, torch, triton, "
            "sdpa, or sdpa_math"
        )
    graph_prefill_backend = str(
        cache_config.get("graph_prefill_backend", "packed")
        if args.graph_prefill_backend is None
        else args.graph_prefill_backend
    )
    if graph_prefill_backend not in {"auto", "packed", "dense"}:
        raise ValueError("cache.graph_prefill_backend must be auto, packed, or dense")

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
            "decode_numerics": decode_numerics,
            "packed_prefill_backend": prefill_backend,
            "graph_prefill_backend": graph_prefill_backend,
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
                    trace_enabled=bool(args.trace_summary),
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
                    f"ITL_P50={_display(result.summary['metrics']['itl_ms'].get('median'))} ms "
                    f"TPOT_P50={_display(result.summary['metrics']['tpot_ms'].get('median'))} ms "
                    f"TPS_P50={_display(result.summary['metrics']['tokens_per_second'].get('median'))} "
                    f"E2E_TPS_P50={_display(result.summary['metrics']['e2e_tokens_per_second'].get('median'))} "
                    f"correctness={correctness['status']}"
                )
                if args.trace_summary:
                    _print_trace_summary(result)

            for paged_mode in paged_modes:
                for block_size in block_sizes:
                    num_blocks = math.ceil(capacity_token_slots / block_size)
                    is_direct = paged_mode == "paged_direct"
                    is_hybrid = paged_mode == "paged_hybrid"
                    is_graph = paged_mode == "paged_graph"
                    is_packed = paged_mode == "paged_packed"
                    runner_class = (
                        PagedCudaGraphBatchRunner
                        if is_graph
                        else PackedPagedPrefillBatchRunner
                        if is_packed
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
                        "trace_enabled": bool(args.trace_summary),
                    }
                    if is_direct or is_hybrid or is_graph or is_packed:
                        runner_kwargs["decode_backend"] = decode_backend
                        runner_kwargs["decode_sdpa_compat"] = (
                            None if decode_numerics == "auto" else decode_numerics == "sdpa_compat"
                        )
                    if is_packed:
                        runner_kwargs["prefill_backend"] = prefill_backend
                    if is_graph:
                        runner_kwargs["prefill_backend"] = graph_prefill_backend
                    runner = runner_class(bundle.model, **runner_kwargs)
                    runner_name = runner.runner_name
                    benchmark_workload = workload
                    if is_graph and len(workload.requests) != batch_size:
                        # A captured graph owns the request-specific page-table
                        # addresses.  Keep one stable group for every measured
                        # repetition instead of sending later harness batches
                        # with different request IDs to the same graph.
                        benchmark_workload = replace(
                            workload,
                            requests=workload.requests[:batch_size],
                        )
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
                    ).run_batched(benchmark_workload, batch_size, runner)
                    correctness = verify_or_write_reference(
                        result,
                        reference_dir / f"{bucket_name}.json",
                    )
                    result_prefix = (
                        "paged_direct"
                        if is_direct
                        else "paged_graph"
                        if is_graph
                        else "paged_packed"
                        if is_packed
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
                            "request_count_measured": len(benchmark_workload.requests),
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
                        else "paged-packed"
                        if is_packed
                        else "paged-hybrid"
                        if is_hybrid
                        else "paged-gather"
                    )
                    print(
                        f"{display_mode:12s} {bucket_name:6s} block={block_size:<3d} "
                        f"batch={batch_size:<2d} "
                        f"TTFT_P50={_display(result.summary['metrics']['ttft_ms'].get('median'))} ms "
                        f"ITL_P50={_display(result.summary['metrics']['itl_ms'].get('median'))} ms "
                        f"TPOT_P50={_display(result.summary['metrics']['tpot_ms'].get('median'))} ms "
                        f"TPS_P50={_display(result.summary['metrics']['tokens_per_second'].get('median'))} "
                        f"E2E_TPS_P50={_display(result.summary['metrics']['e2e_tokens_per_second'].get('median'))} "
                        f"waste={float(snapshot.get('internal_fragmentation', 0.0)):.3f} "
                        f"correctness={correctness['status']}"
                    )
                    if args.trace_summary:
                        _print_trace_summary(result)

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
