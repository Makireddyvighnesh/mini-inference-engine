"""Run the explicit manual prefill/decode benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from minillm_l4.configs.loader import load_yaml_config
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
from minillm_l4.benchmarks.runners.manual_decode import (
    ManualGreedyBatchRunner,
    write_manual_result,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/workloads/qwen3_fp8_manual.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/manual_decode"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "results/phase1/references_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MiniLLM-L4 explicit manual decode benchmark."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--workload",
        choices=("short", "medium", "long", "all"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit Hugging Face Hub access if the pinned local snapshot is absent.",
    )
    parser.add_argument("--device", default=None, help="CUDA device override, e.g. cuda:0")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=None)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--warmup-repetitions", type=int, default=None)
    parser.add_argument("--no-gpu-sampling", action="store_true")
    parser.add_argument("--no-system-telemetry", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = load_yaml_config(path, expected_phase=2, label="Manual decode")
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Manual decode config must contain a model object")
    if model.get("id") != MODEL_ID:
        raise ValueError(
            "Manual decode is pinned to "
            f"{MODEL_ID}; update the config only with an intentional model change"
        )
    if model.get("revision") != MODEL_REVISION:
        raise ValueError(
            "Manual decode config must pin the local checkpoint revision "
            f"{MODEL_REVISION}"
        )
    if model.get("precision") != "fp8":
        raise ValueError(
            "Manual decode must use the checkpoint-native fp8 precision path"
        )
    return payload


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == PROJECT_ROOT.name:
        return PROJECT_ROOT.parent / path
    return PROJECT_ROOT / path


def _bucket_by_prompt_length(prompt_tokens: int) -> tuple[str, int, int]:
    for bucket in BASELINE_BUCKETS:
        if bucket[1] == prompt_tokens:
            return bucket
    allowed = ", ".join(str(bucket[1]) for bucket in BASELINE_BUCKETS)
    raise ValueError(
        f"Unsupported prompt length {prompt_tokens}; expected {allowed}"
    )


def _selected_buckets(
    selection: str,
    configured_lengths: list[int],
) -> list[tuple[str, int, int]]:
    selected = [
        _bucket_by_prompt_length(prompt_tokens)
        for prompt_tokens in configured_lengths
    ]
    if selection == "all":
        return selected
    return [bucket for bucket in selected if bucket[0] == selection]


def _print_summary(
    workload_name: str,
    batch_size: int,
    summary: Mapping[str, Any],
    correctness: Mapping[str, Any],
) -> None:
    metrics = summary["metrics"]
    print(
        f"{workload_name:6s} batch={batch_size:<2d} "
        f"requests/run={summary['run_summaries'][0]['completed_requests']:3d} "
        f"repetitions={summary['repetitions']} "
        f"TTFT_P50={_display(metrics['ttft_ms'].get('median'))} ms "
        f"TPOT_P50={_display(metrics['tpot_ms'].get('median'))} ms "
        f"TPS_P50={_display(summary['tokens_per_second'].get('median'))} "
        f"correctness={correctness['status']}"
    )


def _display(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def main() -> None:
    args = parse_args()
    config_path = _project_path(args.config)
    payload = load_config(config_path)
    model_config = payload["model"]
    workload_config = payload["workloads"]
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
    if not prompt_lengths or any(value <= 0 for value in prompt_lengths):
        raise ValueError("prompt lengths must be positive")
    if not batch_sizes or any(value <= 0 for value in batch_sizes):
        raise ValueError("batch sizes must be positive")
    if len(set(prompt_lengths)) != len(prompt_lengths):
        raise ValueError("prompt lengths must be unique")
    if len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError("batch sizes must be unique")

    selected = _selected_buckets(args.workload, prompt_lengths)
    if not selected:
        raise ValueError(
            f"No configured prompt lengths match --workload {args.workload!r}"
        )
    output_dir = _project_path(args.output_dir)
    reference_dir = _project_path(args.reference_dir)
    dataset_path = _project_path(str(workload_config["dataset"]))
    count = int(workload_config["count"] if args.count is None else args.count)
    if count <= 0:
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
        respect_arrival_schedule=False,
        sample_interval_seconds=float(benchmark_config["sample_interval_seconds"]),
        collect_gpu=bool(benchmark_config["collect_gpu"]) and not args.no_gpu_sampling,
        collect_system_telemetry=(
            bool(benchmark_config["collect_system_telemetry"])
            and not args.no_system_telemetry
        ),
        runner_name=str(benchmark_config["runner"]),
        timing_mode="wall",
        seed=int(workload_config["seed"]),
        timer_overhead_iterations=int(benchmark_config["timer_overhead_iterations"]),
    )
    eos_token_id = None
    if bool(benchmark_config.get("eos_stopping", False)):
        eos_token_id = getattr(bundle.tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise ValueError(
                "eos_stopping is enabled but the tokenizer has no eos_token_id"
            )
    pad_token_id = getattr(bundle.tokenizer, "pad_token_id", None)
    runner = ManualGreedyBatchRunner(
        bundle.model,
        device=device,
        logits_mode=str(benchmark_config.get("logits_mode", "last")),
        eos_token_id=eos_token_id,
        pad_token_id=0 if pad_token_id is None else int(pad_token_id),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "project": "MiniLLM-L4",
        "phase": 2,
        "status": "running",
        "config": str(config_path),
        "initialization": initialization,
        "configuration": {
            "prompt_lengths": prompt_lengths,
            "batch_sizes": batch_sizes,
            "count": count,
            "repetitions": repetitions,
            "warmup_repetitions": warmups,
            "dataset": str(dataset_path),
            "local_files_only": local_files_only,
            "sampling": "greedy argmax",
            "early_stopping": (
                "enabled from tokenizer eos_token_id"
                if eos_token_id is not None
                else "disabled; emit exactly requested output tokens"
            ),
            "timing_boundary": (
                "request event execution through manual prefill/decode return; "
                "tokenization is outside measured runs"
            ),
        },
        "workloads": [],
    }

    for bucket_name, prompt_tokens, default_output_tokens in selected:
        output_map = workload_config["output_tokens"]
        output_tokens = int(
            output_map.get(str(prompt_tokens), default_output_tokens)
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

        for batch_size in batch_sizes:
            result = BenchmarkHarness(
                harness_config,
                benchmark_name="minillm_l4_manual_decode",
            ).run_batched(workload, batch_size, runner)
            reference_path = reference_dir / f"{bucket_name}.json"
            correctness = verify_or_write_reference(result, reference_path)
            result_path = output_dir / f"manual_{bucket_name}_b{batch_size}.json"
            events_path = (
                output_dir / f"manual_{bucket_name}_b{batch_size}_events.jsonl"
            )
            write_manual_result(
                result,
                result_path,
                model_metadata=initialization,
                correctness=correctness,
            )
            write_events_jsonl(result, events_path)
            workload_path = output_dir / f"workload_{bucket_name}.json"
            if not workload_path.exists():
                save_workload(workload, workload_path)
            manifest["workloads"].append(
                {
                    "name": bucket_name,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "batch_size": batch_size,
                    "workload": str(workload_path),
                    "result": str(result_path),
                    "events": str(events_path),
                    "reference": str(reference_path),
                    "correctness": correctness,
                    "summary": result.summary,
                }
            )
            _print_summary(bucket_name, batch_size, result.summary, correctness)

    manifest["status"] = (
        "completed"
        if all(item["correctness"]["status"] == "pass" for item in manifest["workloads"])
        else "failed_correctness"
    )
    manifest_path = output_dir / "manual_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")
    if manifest["status"] != "completed":
        raise RuntimeError("Manual decode correctness checks failed; see the manifest")


if __name__ == "__main__":
    main()
