"""Run staggered uniform and mixed request traces with static batching."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

from minillm_l4.benchmarks.core.harness import BenchmarkHarness, write_events_jsonl
from minillm_l4.benchmarks.core.schemas import HarnessConfig, RequestSpec, WorkloadSpec
from minillm_l4.benchmarks.core.workloads import save_workload
from minillm_l4.benchmarks.runners.concurrent_requests import (
    StaticRequestTraceRunner,
    verify_concurrent_references,
    write_concurrent_result,
)
from minillm_l4.benchmarks.runners.huggingface_baseline import (
    BASELINE_BUCKETS,
    MODEL_ID,
    MODEL_REVISION,
    build_hf_workload,
    load_qwen_fp8,
)
from minillm_l4.configs.loader import load_yaml_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/workloads/qwen3_fp8_concurrent.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/concurrent_requests"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "results/phase1/references_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark FIFO static batching with staggered request arrivals."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--workload", choices=("uniform", "mixed", "all"), default="all"
    )
    parser.add_argument("--max-batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--count-per-bucket", type=int, default=None)
    parser.add_argument("--arrival-interval-ms", type=float, default=None)
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


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == PROJECT_ROOT.name:
        return PROJECT_ROOT.parent / path
    return PROJECT_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    payload = load_yaml_config(path, expected_phase=4, label="Concurrent requests")
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Concurrent request config must contain a model object")
    if model.get("id") != MODEL_ID or model.get("revision") != MODEL_REVISION:
        raise ValueError("Concurrent request config must use the pinned Qwen revision")
    scheduler = payload.get("scheduler")
    if not isinstance(scheduler, Mapping):
        raise ValueError("Concurrent request config must contain a scheduler object")
    if scheduler.get("policy") != "fifo_padded_static_batch":
        raise ValueError("Phase 4 supports only fifo_padded_static_batch")
    return payload


def _bucket_map() -> dict[str, tuple[str, int, int]]:
    return {name: (name, prompt, output) for name, prompt, output in BASELINE_BUCKETS}


def _materialize_bucket(
    tokenizer: Any,
    dataset_path: Path,
    *,
    bucket_name: str,
    count: int,
    output_tokens: int,
    seed: int,
    model_id: str,
    revision: str,
    device: str,
) -> tuple[RequestSpec, ...]:
    try:
        _, prompt_tokens, _ = _bucket_map()[bucket_name]
    except KeyError as error:
        raise ValueError(f"Unknown workload bucket: {bucket_name}") from error
    return build_hf_workload(
        tokenizer,
        dataset_path,
        bucket_name=bucket_name,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        count=count,
        seed=seed,
        model_id=model_id,
        revision=revision,
        device=device,
    ).requests


def build_trace_workloads(
    tokenizer: Any,
    dataset_path: Path,
    config: Mapping[str, Any],
    *,
    seed: int,
    model_id: str,
    revision: str,
    device: str,
    count_per_bucket_override: int | None,
    arrival_interval_ms: float,
) -> dict[str, WorkloadSpec]:
    if arrival_interval_ms < 0:
        raise ValueError("arrival_interval_ms must be non-negative")
    output_map = config["output_tokens"]
    uniform_config = config["uniform"]
    uniform_bucket = str(uniform_config["bucket"])
    uniform_count = int(
        uniform_config["count"]
        if count_per_bucket_override is None
        else count_per_bucket_override
    )
    if uniform_count < 1:
        raise ValueError("uniform request count must be positive")
    uniform_requests = _materialize_bucket(
        tokenizer,
        dataset_path,
        bucket_name=uniform_bucket,
        count=uniform_count,
        output_tokens=int(output_map[uniform_bucket]),
        seed=seed,
        model_id=model_id,
        revision=revision,
        device=device,
    )

    mixed_config = config["mixed"]
    mixed_buckets = tuple(str(value) for value in mixed_config["buckets"])
    mixed_count = int(
        mixed_config["count_per_bucket"]
        if count_per_bucket_override is None
        else count_per_bucket_override
    )
    if mixed_count < 1:
        raise ValueError("mixed count_per_bucket must be positive")
    by_bucket = {
        bucket: _materialize_bucket(
            tokenizer,
            dataset_path,
            bucket_name=bucket,
            count=mixed_count,
            output_tokens=int(output_map[bucket]),
            seed=seed,
            model_id=model_id,
            revision=revision,
            device=device,
        )
        for bucket in mixed_buckets
    }
    mixed_requests = tuple(
        by_bucket[bucket][index]
        for index in range(mixed_count)
        for bucket in mixed_buckets
    )

    def make_workload(name: str, requests: Sequence[RequestSpec]) -> WorkloadSpec:
        scheduled = tuple(
            replace(request, scheduled_arrival_ms=index * arrival_interval_ms)
            for index, request in enumerate(requests)
        )
        return WorkloadSpec(
            name=name,
            seed=seed,
            requests=scheduled,
            model_id=model_id,
            model_revision=revision,
            dtype="fp8",
            device=device,
            arrival_pattern="fixed_rate",
            metadata={
                "arrival_interval_ms": arrival_interval_ms,
                "request_shapes": [
                    {
                        "request_id": request.request_id,
                        "prompt_tokens": request.prompt_tokens,
                        "output_tokens": request.max_new_tokens,
                    }
                    for request in scheduled
                ],
            },
        )

    return {
        "uniform": make_workload("concurrent_uniform", uniform_requests),
        "mixed": make_workload("concurrent_mixed", mixed_requests),
    }


def _display(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


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
        runner_name="fifo_padded_static_batch",
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
        "phase": 4,
        "status": "running",
        "config": str(config_path),
        "initialization": initialization,
        "configuration": {
            "workload_selection": args.workload,
            "count_per_bucket": args.count_per_bucket,
            "arrival_interval_ms": arrival_interval_ms,
            "max_batch_sizes": max_batch_sizes,
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
    manifest_path = output_dir / "concurrent_manifest.json"

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
            runner = StaticRequestTraceRunner(
                bundle.model,
                max_batch_size=max_batch_size,
                device=device,
                logits_mode=str(benchmark_config.get("logits_mode", "last")),
                eos_token_id=eos_token_id,
                pad_token_id=int(pad_token_id),
                cancel_request_ids=cancellation_ids,
            )
            result = BenchmarkHarness(
                harness_config,
                benchmark_name="minillm_l4_concurrent_requests",
            ).run_trace(workload, runner)
            correctness = verify_concurrent_references(
                result,
                reference_dir,
                cancelled_request_ids=cancellation_ids,
            )
            result_path = output_dir / f"static_{workload_name}_b{max_batch_size}.json"
            events_path = output_dir / (
                f"static_{workload_name}_b{max_batch_size}_events.jsonl"
            )
            write_concurrent_result(
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
            print(
                f"{workload_name:7s} max_batch={max_batch_size:<2d} "
                f"TTFT_P50={_display(metrics['ttft_ms'].get('median'))} ms "
                f"E2E_P95={_display(metrics['e2e_latency_ms'].get('p95'))} ms "
                f"TPS_P50={_display(result.summary['tokens_per_second'].get('median'))} "
                f"padding={_display((runner.last_summary or {}).get('padding_waste_ratio'))} "
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
        raise RuntimeError("Concurrent request correctness failed; see the manifest")


if __name__ == "__main__":
    main()
