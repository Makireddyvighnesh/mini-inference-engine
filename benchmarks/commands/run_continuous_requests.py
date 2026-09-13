"""Run staggered request traces with iteration-level continuous batching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from minillm_l4.benchmarks.core.harness import BenchmarkHarness, write_events_jsonl
from minillm_l4.benchmarks.core.schemas import HarnessConfig
from minillm_l4.benchmarks.core.workloads import save_workload
from minillm_l4.benchmarks.runners.concurrent_requests import (
    verify_concurrent_references,
)
from minillm_l4.benchmarks.runners.continuous_requests import (
    ContinuousRequestTraceRunner,
    write_continuous_result,
)
from minillm_l4.benchmarks.runners.huggingface_baseline import (
    MODEL_ID,
    MODEL_REVISION,
    load_qwen_fp8,
)
from minillm_l4.configs.loader import load_yaml_config

from .run_concurrent_requests import (
    PROJECT_ROOT,
    _display,
    _project_path,
    build_trace_workloads,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/workloads/qwen3_fp8_continuous.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/continuous_requests"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "results/phase1/references_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark iteration-level continuous batching."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--workload", choices=("uniform", "mixed", "all"), default="all"
    )
    parser.add_argument("--max-batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--max-prefill-tokens", type=int, default=None)
    parser.add_argument("--max-wait-ms", type=float, default=None)
    parser.add_argument("--count-per-bucket", type=int, default=None)
    parser.add_argument("--arrival-interval-ms", type=float, default=None)
    parser.add_argument(
        "--arrival-pattern",
        choices=("fixed_rate", "poisson"),
        default=None,
    )
    parser.add_argument("--arrival-rate-per-second", type=float, default=None)
    parser.add_argument("--cancel-request-indices", type=int, nargs="*", default=())
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


def load_config(path: Path) -> dict[str, Any]:
    payload = load_yaml_config(path, expected_phase=5, label="Continuous batching")
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Continuous batching config must contain a model object")
    if model.get("id") != MODEL_ID or model.get("revision") != MODEL_REVISION:
        raise ValueError("Continuous batching config must use the pinned Qwen revision")
    scheduler = payload.get("scheduler")
    if not isinstance(scheduler, Mapping):
        raise ValueError(
            "Continuous batching config must contain a scheduler object"
        )
    if scheduler.get("policy") != "continuous_in_flight_batching":
        raise ValueError(
            "Phase 5 supports only continuous_in_flight_batching"
        )
    if int(scheduler.get("max_prefill_tokens", 0)) < 1:
        raise ValueError("max_prefill_tokens must be positive")
    if float(scheduler.get("max_wait_ms", 0.0)) < 0:
        raise ValueError("max_wait_ms must be non-negative")
    return payload


def main() -> None:
    args = parse_args()
    config_path = _project_path(args.config)
    payload = load_config(config_path)
    model_config = payload["model"]
    workload_config = payload["workloads"]
    scheduler_config = payload["scheduler"]
    benchmark_config = payload["benchmark"]
    output_dir = _project_path(args.output_dir)
    reference_dir = _project_path(args.reference_dir)
    dataset_path = _project_path(str(workload_config["dataset"]))
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
    arrival_interval_ms = float(
        workload_config["arrival_interval_ms"]
        if args.arrival_interval_ms is None
        else args.arrival_interval_ms
    )
    arrival_pattern = str(
        workload_config.get("arrival_pattern", "fixed_rate")
        if args.arrival_pattern is None
        else args.arrival_pattern
    )
    configured_rate = workload_config.get("arrival_rate_per_second")
    arrival_rate_per_second = (
        None
        if args.arrival_rate_per_second is None and configured_rate is None
        else float(
            configured_rate
            if args.arrival_rate_per_second is None
            else args.arrival_rate_per_second
        )
    )
    workloads = build_trace_workloads(
        bundle.tokenizer,
        dataset_path,
        workload_config,
        seed=int(workload_config["seed"]),
        model_id=str(model_config["id"]),
        revision=str(model_config["revision"]),
        device=device,
        count_per_bucket_override=args.count_per_bucket,
        arrival_interval_ms=arrival_interval_ms,
        arrival_pattern=arrival_pattern,
        arrival_rate_per_second=arrival_rate_per_second,
    )
    selected_names = (
        tuple(workloads) if args.workload == "all" else (args.workload,)
    )
    max_batch_sizes = (
        [int(value) for value in scheduler_config["max_batch_sizes"]]
        if args.max_batch_sizes is None
        else list(args.max_batch_sizes)
    )
    if not max_batch_sizes or any(value < 1 for value in max_batch_sizes):
        raise ValueError("max batch sizes must be positive")
    max_prefill_tokens = int(
        scheduler_config["max_prefill_tokens"]
        if args.max_prefill_tokens is None
        else args.max_prefill_tokens
    )
    max_wait_ms = float(
        scheduler_config.get("max_wait_ms", 0.0)
        if args.max_wait_ms is None
        else args.max_wait_ms
    )
    if max_prefill_tokens < 1:
        raise ValueError("max prefill tokens must be positive")
    if max_wait_ms < 0:
        raise ValueError("max wait ms must be non-negative")
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
    harness_config = HarnessConfig(
        warmup_repetitions=warmups,
        repetitions=repetitions,
        respect_arrival_schedule=True,
        sample_interval_seconds=float(benchmark_config["sample_interval_seconds"]),
        collect_gpu=bool(benchmark_config["collect_gpu"]) and not args.no_gpu_sampling,
        collect_system_telemetry=(
            bool(benchmark_config["collect_system_telemetry"])
            and not args.no_system_telemetry
        ),
        runner_name="continuous_in_flight_batching",
        timing_mode="wall",
        seed=int(workload_config["seed"]),
        timer_overhead_iterations=int(benchmark_config["timer_overhead_iterations"]),
    )
    eos_token_id = None
    if bool(benchmark_config.get("eos_stopping", False)):
        eos_token_id = bundle.tokenizer.eos_token_id
    pad_token_id = bundle.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = bundle.tokenizer.eos_token_id or 0

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "project": "MiniLLM-L4",
        "phase": 5,
        "status": "running",
        "config": str(config_path),
        "initialization": initialization,
        "configuration": {
            "workload_selection": args.workload,
            "count_per_bucket": args.count_per_bucket,
            "arrival_interval_ms": arrival_interval_ms,
            "arrival_pattern": arrival_pattern,
            "arrival_rate_per_second": arrival_rate_per_second,
            "max_batch_sizes": max_batch_sizes,
            "max_prefill_tokens": max_prefill_tokens,
            "max_wait_ms": max_wait_ms,
            "repetitions": repetitions,
            "warmup_repetitions": warmups,
            "sample_interval_seconds": harness_config.sample_interval_seconds,
            "collect_gpu": harness_config.collect_gpu,
            "collect_system_telemetry": harness_config.collect_system_telemetry,
            "eos_stopping": eos_token_id is not None,
            "sampling": "greedy argmax",
        },
        "workloads": [],
    }
    manifest_path = output_dir / "continuous_manifest.json"

    def write_manifest() -> None:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    write_manifest()
    for workload_name in selected_names:
        workload = workloads[workload_name]
        save_workload(workload, output_dir / f"workload_{workload_name}.json")
        cancellation_ids: tuple[str, ...] = ()
        if args.cancel_request_indices:
            try:
                cancellation_ids = tuple(
                    workload.requests[index].request_id
                    for index in args.cancel_request_indices
                )
            except IndexError as error:
                raise ValueError("cancel request index is outside the workload") from error

        for max_batch_size in max_batch_sizes:
            runner = ContinuousRequestTraceRunner(
                bundle.model,
                max_batch_size=max_batch_size,
                max_prefill_tokens=max_prefill_tokens,
                max_wait_ms=max_wait_ms,
                device=device,
                logits_mode=str(benchmark_config.get("logits_mode", "last")),
                eos_token_id=eos_token_id,
                pad_token_id=int(pad_token_id),
                cancel_request_ids=cancellation_ids,
            )
            result = BenchmarkHarness(
                harness_config,
                benchmark_name="minillm_l4_continuous_requests",
            ).run_trace(workload, runner)
            correctness = verify_concurrent_references(
                result,
                reference_dir,
                cancelled_request_ids=cancellation_ids,
            )
            result_path = output_dir / f"continuous_{workload_name}_b{max_batch_size}.json"
            events_path = output_dir / (
                f"continuous_{workload_name}_b{max_batch_size}_events.jsonl"
            )
            write_continuous_result(
                result,
                result_path,
                model_metadata=initialization,
                correctness=correctness,
                scheduler_summary=runner.last_summary,
            )
            write_events_jsonl(result, events_path)
            manifest["workloads"].append(
                {
                    "name": workload_name,
                    "max_batch_size": max_batch_size,
                    "max_prefill_tokens": max_prefill_tokens,
                    "max_wait_ms": max_wait_ms,
                    "cancelled_request_ids": list(cancellation_ids),
                    "result": str(result_path),
                    "events": str(events_path),
                    "correctness": correctness,
                    "scheduler": runner.last_summary,
                    "summary": result.summary,
                }
            )
            write_manifest()
            metrics = result.summary["metrics"]
            scheduler = runner.last_summary or {}
            print(
                f"{workload_name:7s} max_batch={max_batch_size:<2d} "
                f"active_max={scheduler.get('maximum_active_batch_size', 'n/a')!s:<2} "
                f"TTFT_P50={_display(metrics['ttft_ms'].get('median'))} ms "
                f"E2E_P95={_display(metrics['e2e_latency_ms'].get('p95'))} ms "
                f"TPS_P50={_display(result.summary['tokens_per_second'].get('median'))} "
                f"queue_P95={_display(metrics['queue_delay_ms'].get('p95'))} ms "
                f"correctness={correctness['status']}"
            )

    manifest["status"] = (
        "completed"
        if all(item["correctness"]["status"] == "pass" for item in manifest["workloads"])
        else "failed_correctness"
    )
    write_manifest()
    print(f"Manifest: {manifest_path}")
    if manifest["status"] != "completed":
        raise RuntimeError("Continuous batching correctness failed; see the manifest")


if __name__ == "__main__":
    main()
