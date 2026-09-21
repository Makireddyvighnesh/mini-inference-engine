"""Stress-test continuous batching capacity and in-flight admissions."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import torch

from minillm_l4.benchmarks.core.harness import BenchmarkHarness
from minillm_l4.benchmarks.core.schemas import HarnessConfig, RequestSpec, WorkloadSpec
from minillm_l4.benchmarks.core.workloads import save_workload
from minillm_l4.benchmarks.runners.continuous_requests import (
    ContinuousRequestTraceRunner,
    write_continuous_result,
)
from minillm_l4.benchmarks.runners.huggingface_baseline import (
    BASELINE_BUCKETS,
    MODEL_ID,
    MODEL_REVISION,
    build_hf_workload,
    load_qwen_fp8,
)
from minillm_l4.configs.loader import load_yaml_config

from .run_concurrent_requests import PROJECT_ROOT, _project_path


DEFAULT_CONFIG = PROJECT_ROOT / "configs/workloads/qwen3_fp8_stress.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/capacity_stress"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "results/phase1/references_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stress-test continuous batching concurrency, prefill-token budgets, "
            "and admissions while decode is active."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--scenario",
        choices=("concurrency", "input_budget", "interleave", "all"),
        default="all",
    )
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
        "--continue-after-failure",
        action="store_true",
        help="Continue larger cases after a failed correctness or runtime case.",
    )
    return parser.parse_args()


def _display(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _bucket_lengths() -> dict[str, int]:
    return {name: prompt for name, prompt, _ in BASELINE_BUCKETS}


def _reference_index(reference_dir: Path) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for path in sorted(reference_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload.get("requests", {}).values():
            digest = str(record.get("prompt_sha256", ""))
            if digest:
                references[digest] = dict(record)
    return references


def _verify_stress_references(
    result: Any,
    references: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify by prompt digest and allow a shorter output prefix.

    Stress cases intentionally reuse the Phase 1 prompt corpus with new request
    IDs and shorter output limits. Matching by prompt digest keeps correctness
    independent of the stress matrix while preserving exact token equality.
    """

    request_specs = {
        request.request_id: request for request in result.workload.requests
    }
    errors: list[str] = []
    first_outputs: dict[str, tuple[int, ...]] = {}
    checked = 0
    for run in result.runs:
        for record in run["requests"]:
            request_id = str(record["request_id"])
            outcome = record["outcome"]
            if outcome["status"] != "completed":
                errors.append(f"{request_id}: status={outcome['status']}")
                continue
            request = request_specs[request_id]
            reference = references.get(request.prompt_sha256)
            if reference is None:
                errors.append(f"{request_id}: missing prompt reference")
                continue
            expected = tuple(int(value) for value in reference.get("generated_token_ids", ()))
            actual = tuple(int(value) for value in outcome["generated_token_ids"])
            if len(expected) < request.max_new_tokens:
                errors.append(
                    f"{request_id}: reference has {len(expected)} tokens, "
                    f"needed {request.max_new_tokens}"
                )
            elif actual != expected[: request.max_new_tokens]:
                errors.append(f"{request_id}: generated token mismatch")
            prior = first_outputs.setdefault(request_id, actual)
            if prior != actual:
                errors.append(f"{request_id}: output changed across repetitions")
            checked += 1
    return {
        "status": "pass" if not errors else "fail",
        "comparison_mode": "exact_prefix_of_prompt_reference",
        "completed_outputs_checked": checked,
        "reference_prompt_count": len(references),
        "errors": errors,
    }


def _prompt_pool(
    tokenizer: Any,
    dataset_path: Path,
    *,
    bucket: str,
    seed: int,
    model_id: str,
    revision: str,
) -> tuple[RequestSpec, ...]:
    lengths = _bucket_lengths()
    if bucket not in lengths:
        raise ValueError(f"unknown prompt bucket: {bucket}")
    workload = build_hf_workload(
        tokenizer,
        dataset_path,
        bucket_name=bucket,
        prompt_tokens=lengths[bucket],
        output_tokens=1,
        count=4,
        seed=seed,
        model_id=model_id,
        revision=revision,
        device="cuda:0",
    )
    return workload.requests


def _make_workload(
    tokenizer: Any,
    dataset_path: Path,
    *,
    name: str,
    bucket_sequence: Sequence[str],
    requests_per_bucket: int,
    output_tokens: int,
    arrival_interval_ms: float,
    seed: int,
    model_id: str,
    revision: str,
) -> WorkloadSpec:
    if not bucket_sequence:
        raise ValueError("bucket_sequence must not be empty")
    if requests_per_bucket < 1 or output_tokens < 1:
        raise ValueError("requests_per_bucket and output_tokens must be positive")
    pools = {
        bucket: _prompt_pool(
            tokenizer,
            dataset_path,
            bucket=bucket,
            seed=seed,
            model_id=model_id,
            revision=revision,
        )
        for bucket in set(bucket_sequence)
    }
    requests: list[RequestSpec] = []
    for cycle in range(requests_per_bucket):
        for bucket in bucket_sequence:
            source = pools[bucket][cycle % len(pools[bucket])]
            index = len(requests)
            requests.append(
                replace(
                    source,
                    request_id=f"{name}-{index:04d}",
                    max_new_tokens=output_tokens,
                    scheduled_arrival_ms=index * arrival_interval_ms,
                    metadata={
                        **dict(source.metadata),
                        "stress_workload": name,
                        "stress_source_request_id": source.request_id,
                    },
                )
            )
    return WorkloadSpec(
        name=name,
        seed=seed,
        requests=tuple(requests),
        model_id=model_id,
        model_revision=revision,
        dtype="fp8",
        device="cuda:0",
        arrival_pattern="fixed_rate",
        metadata={
            "stress_workload": name,
            "bucket_sequence": list(bucket_sequence),
            "requests_per_bucket": requests_per_bucket,
            "output_tokens": output_tokens,
            "arrival_interval_ms": arrival_interval_ms,
            "total_prompt_tokens": sum(request.prompt_tokens for request in requests),
        },
    )


def _scenario_cases(
    tokenizer: Any,
    dataset_path: Path,
    config: Mapping[str, Any],
    *,
    seed: int,
    model_id: str,
    revision: str,
    selected: str,
) -> list[dict[str, Any]]:
    stress = config.get("stress")
    if not isinstance(stress, Mapping):
        raise ValueError("stress config must contain a mapping")
    cases: list[dict[str, Any]] = []

    if selected in {"all", "concurrency"}:
        item = stress["concurrency"]
        workload = _make_workload(
            tokenizer,
            dataset_path,
            name="stress_concurrency",
            bucket_sequence=(str(item["prompt_bucket"]),),
            requests_per_bucket=int(item["request_count"]),
            output_tokens=int(item["output_tokens"]),
            arrival_interval_ms=float(item["arrival_interval_ms"]),
            seed=seed,
            model_id=model_id,
            revision=revision,
        )
        for max_batch_size in item["max_batch_sizes"]:
            cases.append(
                {
                    "scenario": "concurrency",
                    "case_name": f"concurrency_b{int(max_batch_size)}",
                    "workload": workload,
                    "max_batch_size": int(max_batch_size),
                    "max_prefill_tokens": int(item["max_prefill_tokens"]),
                    "max_wait_ms": float(item.get("max_wait_ms", 0.0)),
                }
            )

    if selected in {"all", "input_budget"}:
        item = stress["input_budget"]
        workload = _make_workload(
            tokenizer,
            dataset_path,
            name="stress_input_budget",
            bucket_sequence=(str(item["prompt_bucket"]),),
            requests_per_bucket=int(item["request_count"]),
            output_tokens=int(item["output_tokens"]),
            arrival_interval_ms=float(item["arrival_interval_ms"]),
            seed=seed,
            model_id=model_id,
            revision=revision,
        )
        for budget in item["max_prefill_tokens"]:
            cases.append(
                {
                    "scenario": "input_budget",
                    "case_name": f"input_budget_t{int(budget)}",
                    "workload": workload,
                    "max_batch_size": int(item["max_batch_size"]),
                    "max_prefill_tokens": int(budget),
                    "max_wait_ms": float(item.get("max_wait_ms", 0.0)),
                }
            )

    if selected in {"all", "interleave"}:
        item = stress["interleave"]
        buckets = tuple(str(value) for value in item["buckets"])
        workload = _make_workload(
            tokenizer,
            dataset_path,
            name="stress_interleave",
            bucket_sequence=buckets,
            requests_per_bucket=int(item["cycles"]),
            output_tokens=int(item["output_tokens"]),
            arrival_interval_ms=float(item["arrival_interval_ms"]),
            seed=seed,
            model_id=model_id,
            revision=revision,
        )
        for max_batch_size in item["max_batch_sizes"]:
            cases.append(
                {
                    "scenario": "interleave",
                    "case_name": f"interleave_b{int(max_batch_size)}",
                    "workload": workload,
                    "max_batch_size": int(max_batch_size),
                    "max_prefill_tokens": int(item["max_prefill_tokens"]),
                    "max_wait_ms": float(item.get("max_wait_ms", 0.0)),
                }
            )
    return cases


def main() -> None:
    args = parse_args()
    config_path = _project_path(args.config)
    payload = load_yaml_config(config_path, expected_phase=5, label="Capacity stress")
    model_config = payload["model"]
    workload_config = payload["workloads"]
    benchmark_config = payload["benchmark"]
    output_dir = _project_path(args.output_dir)
    reference_dir = _project_path(args.reference_dir)
    dataset_path = _project_path(str(workload_config["dataset"]))
    device = str(model_config["device"] if args.device is None else args.device)
    local_files_only = bool(model_config.get("local_files_only", True)) and not args.allow_download

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
    cases = _scenario_cases(
        bundle.tokenizer,
        dataset_path,
        payload,
        seed=int(workload_config["seed"]),
        model_id=str(model_config["id"]),
        revision=str(model_config["revision"]),
        selected=args.scenario,
    )
    if not cases:
        raise ValueError("stress matrix produced no cases")
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
        runner_name="continuous_in_flight_batching_capacity_stress",
        timing_mode="wall",
        seed=int(workload_config["seed"]),
        timer_overhead_iterations=int(benchmark_config["timer_overhead_iterations"]),
    )
    pad_token_id = bundle.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = bundle.tokenizer.eos_token_id or 0
    references = _reference_index(reference_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "project": "MiniLLM-L4",
        "phase": 5,
        "benchmark": "capacity_stress",
        "status": "running",
        "config": str(config_path),
        "initialization": initialization,
        "configuration": {
            "scenario": args.scenario,
            "repetitions": repetitions,
            "warmup_repetitions": warmups,
            "continue_after_failure": args.continue_after_failure,
            "correctness_mode": "exact_prefix_of_prompt_reference",
        },
        "cases": [],
    }
    manifest_path = output_dir / "capacity_stress_manifest.json"

    def write_manifest() -> None:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    workload_paths: set[str] = set()
    write_manifest()
    for case in cases:
        workload = case["workload"]
        workload_path = output_dir / f"workload_{workload.name}.json"
        if str(workload_path) not in workload_paths:
            save_workload(workload, workload_path)
            workload_paths.add(str(workload_path))
        case_name = str(case["case_name"])
        runner = ContinuousRequestTraceRunner(
            bundle.model,
            max_batch_size=int(case["max_batch_size"]),
            max_prefill_tokens=int(case["max_prefill_tokens"]),
            max_wait_ms=float(case["max_wait_ms"]),
            device=device,
            logits_mode=str(benchmark_config.get("logits_mode", "last")),
            pad_token_id=int(pad_token_id),
        )
        try:
            result = BenchmarkHarness(
                harness_config,
                benchmark_name="minillm_l4_capacity_stress",
            ).run_trace(workload, runner)
            correctness = _verify_stress_references(result, references)
            result_path = output_dir / f"{case_name}.json"
            write_continuous_result(
                result,
                result_path,
                model_metadata=initialization,
                correctness=correctness,
                scheduler_summary=runner.last_summary,
            )
            scheduler = runner.last_summary or {}
            metrics = result.summary["metrics"]
            row = {
                "scenario": case["scenario"],
                "case_name": case_name,
                "max_batch_size": case["max_batch_size"],
                "max_prefill_tokens": case["max_prefill_tokens"],
                "max_wait_ms": case["max_wait_ms"],
                "request_count": len(workload.requests),
                "total_prompt_tokens": sum(
                    request.prompt_tokens for request in workload.requests
                ),
                "maximum_concurrent_requests": scheduler.get(
                    "maximum_concurrent_requests", 0
                ),
                "maximum_prefill_input_tokens": scheduler.get(
                    "maximum_prefill_input_tokens", 0
                ),
                "maximum_prefill_compute_slots": scheduler.get(
                    "maximum_prefill_compute_slots", 0
                ),
                "maximum_prefill_padding_slots": scheduler.get(
                    "maximum_prefill_padding_slots", 0
                ),
                "maximum_active_prompt_tokens": scheduler.get(
                    "maximum_active_prompt_tokens", 0
                ),
                "maximum_active_cached_tokens": scheduler.get(
                    "maximum_active_cached_tokens", 0
                ),
                "maximum_queue_depth": scheduler.get("maximum_queue_depth", 0),
                "prefill_batches_while_decoding": scheduler.get(
                    "prefill_batches_while_decoding", 0
                ),
                "requests_prefilled_while_decoding": scheduler.get(
                    "requests_prefilled_while_decoding", 0
                ),
                "peak_allocated_bytes": result.summary["memory"].get(
                    "peak_allocated_bytes"
                ),
                "peak_reserved_bytes": result.summary["memory"].get(
                    "peak_reserved_bytes"
                ),
                "gpu_utilization_p50": result.summary[
                    "gpu_utilization_percent"
                ].get("median"),
                "ttft_p50_ms": metrics["ttft_ms"].get("median"),
                "ttft_p95_ms": metrics["ttft_ms"].get("p95"),
                "tpot_p50_ms": metrics["tpot_ms"].get("median"),
                "e2e_p95_ms": metrics["e2e_latency_ms"].get("p95"),
                "tokens_per_second_p50": result.summary["tokens_per_second"].get("median"),
                "requests_per_second_p50": result.summary["requests_per_second"].get("median"),
                "correctness": correctness,
                "result": str(result_path),
            }
        except Exception as error:
            row = {
                "scenario": case["scenario"],
                "case_name": case_name,
                "max_batch_size": case["max_batch_size"],
                "max_prefill_tokens": case["max_prefill_tokens"],
                "request_count": len(workload.requests),
                "correctness": {"status": "error", "errors": [f"{type(error).__name__}: {error}"]},
                "error": f"{type(error).__name__}: {error}",
            }
        manifest["cases"].append(row)
        write_manifest()
        print(
            f"{row['scenario']:11s} {row['case_name']:20s} "
            f"active={row.get('maximum_concurrent_requests', 'n/a')!s:<2} "
            f"prefill_tokens={row.get('maximum_prefill_input_tokens', 'n/a')!s:<5} "
            f"interleave={row.get('prefill_batches_while_decoding', 'n/a')!s:<3} "
            f"TTFT_P50={_display(row.get('ttft_p50_ms'))} ms "
            f"TPS_P50={_display(row.get('tokens_per_second_p50'))} "
            f"correctness={row['correctness']['status']}"
        )
        if row["correctness"]["status"] != "pass" and not args.continue_after_failure:
            break
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest["status"] = (
        "completed"
        if manifest["cases"]
        and all(case["correctness"]["status"] == "pass" for case in manifest["cases"])
        else "failed"
    )
    write_manifest()
    print(f"Manifest: {manifest_path}")
    if manifest["status"] != "completed":
        raise RuntimeError("Capacity stress failed; see the manifest")


if __name__ == "__main__":
    main()
