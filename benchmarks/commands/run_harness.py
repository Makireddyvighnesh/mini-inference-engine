from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..core.harness import BenchmarkHarness, write_events_jsonl, write_result
from ..core.schemas import HarnessConfig
from ..core.workloads import (
    build_fixture_workloads,
    save_workload,
)
from ..runners.simulated import make_simulated_runner
from minillm_l4.configs.loader import load_yaml_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/workloads/minillm_l4_harness.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/phase0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MiniLLM-L4 Phase 0 benchmark harness."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--workload",
        choices=("short", "medium", "long", "mixed", "all"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--warmup-repetitions", type=int, default=None)
    parser.add_argument(
        "--respect-arrivals",
        action="store_true",
        help="Sleep to replay the workload's scheduled arrival timestamps.",
    )
    parser.add_argument(
        "--no-gpu-sampling",
        action="store_true",
        help="Skip CUDA allocator and nvidia-smi sampling.",
    )
    parser.add_argument(
        "--no-system-telemetry",
        action="store_true",
        help="Keep CUDA allocator snapshots but skip nvidia-smi queries.",
    )
    parser.add_argument("--timer-overhead-iterations", type=int, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return load_yaml_config(path, expected_phase=0, label="Harness")


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == PROJECT_ROOT.name:
        return PROJECT_ROOT.parent / path
    return PROJECT_ROOT / path


def selected_workload_names(selection: str) -> list[str]:
    return ["short", "medium", "long", "mixed"] if selection == "all" else [selection]


def main() -> None:
    args = parse_args()
    payload = load_config(project_path(args.config))
    model = payload["model"]
    workload_config = payload["workloads"]
    benchmark_config = payload["benchmark"]

    workloads = build_fixture_workloads(
        seed=int(workload_config["seed"]),
        fixed_count=int(workload_config["fixed_count"]),
        mixed_count=int(workload_config["mixed_count"]),
        arrival_pattern=str(workload_config["arrival_pattern"]),
        arrival_rate_per_second=workload_config.get("arrival_rate_per_second"),
        model_id=str(model["id"]),
        model_revision=str(model["revision"]),
        dtype=str(model["dtype"]),
        device=str(model["device"]),
        vocab_size=int(workload_config["vocab_size"]),
    )
    repetitions = (
        int(benchmark_config["repetitions"])
        if args.repetitions is None
        else args.repetitions
    )
    warmups = (
        int(benchmark_config["warmup_repetitions"])
        if args.warmup_repetitions is None
        else args.warmup_repetitions
    )
    timer_overhead_iterations = (
        int(benchmark_config.get("timer_overhead_iterations", 1000))
        if args.timer_overhead_iterations is None
        else args.timer_overhead_iterations
    )
    configuration = HarnessConfig(
        warmup_repetitions=warmups,
        repetitions=repetitions,
        respect_arrival_schedule=(
            bool(benchmark_config["respect_arrival_schedule"])
            or args.respect_arrivals
        ),
        sample_interval_seconds=float(benchmark_config["sample_interval_seconds"]),
        collect_gpu=bool(benchmark_config["collect_gpu"]) and not args.no_gpu_sampling,
        collect_system_telemetry=(
            bool(benchmark_config["collect_system_telemetry"])
            and not args.no_system_telemetry
        ),
        runner_name=str(benchmark_config["runner"]),
        timing_mode="wall",
        seed=int(workload_config["seed"]),
        timer_overhead_iterations=timer_overhead_iterations,
    )
    runner = make_simulated_runner(
        prefill_ms=float(benchmark_config["simulated_prefill_ms"]),
        token_ms=float(benchmark_config["simulated_token_ms"]),
    )
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = selected_workload_names(args.workload)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "project": "MiniLLM-L4",
        "phase": 0,
        "config": str(args.config),
        "workloads": [],
    }

    for name in selected:
        workload = workloads[name]
        save_workload(workload, output_dir / f"workload_{name}.json")
        result = BenchmarkHarness(configuration).run(workload, runner)
        result_path = output_dir / f"harness_{name}.json"
        events_path = output_dir / f"harness_{name}_events.jsonl"
        write_result(result, result_path)
        write_events_jsonl(result, events_path)
        manifest["workloads"].append(
            {
                "name": name,
                "workload": str(output_dir / f"workload_{name}.json"),
                "result": str(result_path),
                "events": str(events_path),
                "summary": result.summary,
            }
        )
        _print_summary(name, result.summary)

    manifest_path = output_dir / "harness_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")


def _print_summary(name: str, summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    first_run = summary["run_summaries"][0]
    ttft = metrics["ttft_ms"].get("median")
    tpot = metrics["tpot_ms"].get("median")
    print(
        f"{name:6s} requests/run={first_run['completed_requests']:3d} "
        f"repetitions={summary['repetitions']} "
        f"TTFT_P50={_display(ttft)} ms "
        f"TPOT_P50={_display(tpot)} ms "
        f"TPS_P50={_display(summary['tokens_per_second'].get('median'))} "
        f"VRAM={'available' if summary['memory']['available'] else 'unavailable'}"
    )


def _display(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    main()
