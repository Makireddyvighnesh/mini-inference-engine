"""Compare whole-prompt and chunked prefill on identical scheduled traffic."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from minillm_l4.benchmarks.core.harness import BenchmarkHarness, write_events_jsonl
from minillm_l4.benchmarks.core.schemas import HarnessConfig
from minillm_l4.benchmarks.core.workloads import save_workload
from minillm_l4.benchmarks.runners.chunked_prefill import ChunkedPrefillPagedRunner
from minillm_l4.benchmarks.runners.concurrent_requests import verify_concurrent_references
from minillm_l4.benchmarks.runners.huggingface_baseline import MODEL_ID, MODEL_REVISION, load_qwen_fp8
from minillm_l4.configs.loader import load_yaml_config
from .run_concurrent_requests import PROJECT_ROOT, _project_path, build_trace_workloads
from .run_prefix_trace import _output_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/workloads/qwen3_fp8_chunked.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--reference-dir", type=Path, default=PROJECT_ROOT / "results/phase1/references_baseline")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--workload", choices=("uniform", "mixed", "all"), default="all")
    parser.add_argument("--chunk-sizes", type=int, nargs="+")
    parser.add_argument("--max-prefill-tokens", type=int)
    parser.add_argument("--max-batch-size", type=int)
    parser.add_argument("--arrival-interval-ms", type=float)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--warmup-repetitions", type=int)
    parser.add_argument("--enable-prefix", action="store_true")
    parser.add_argument("--no-gpu-sampling", action="store_true")
    parser.add_argument("--no-system-telemetry", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate configuration and show the matrix without loading weights")
    return parser.parse_args(argv)


def resolved_configuration(args: argparse.Namespace) -> dict[str, Any]:
    payload = load_yaml_config(args.config, expected_phase=8, label="Chunked prefill")
    if payload["model"]["id"] != MODEL_ID or payload["model"]["revision"] != MODEL_REVISION:
        raise ValueError("chunked benchmark requires the pinned model revision")
    engine, workload, benchmark = payload["engine"], payload["workloads"], payload["benchmark"]
    for attr, section, key in [
        ("max_batch_size", engine, "max_batch_size"),
        ("max_prefill_tokens", engine, "max_prefill_tokens"),
        ("arrival_interval_ms", workload, "arrival_interval_ms"),
        ("repetitions", benchmark, "repetitions"),
        ("warmup_repetitions", benchmark, "warmup_repetitions"),
    ]:
        value = getattr(args, attr)
        if value is not None:
            section[key] = value
    requested = args.chunk_sizes if args.chunk_sizes is not None else engine["chunk_sizes"]
    if not requested or any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in requested):
        raise ValueError("chunk sizes must be non-negative integers; 0 denotes the control")
    engine["chunk_sizes"] = list(dict.fromkeys([0, *requested]))
    engine["enable_prefix"] = bool(engine.get("enable_prefix", False) or args.enable_prefix)
    if min(int(engine[key]) for key in ["max_batch_size", "max_prefill_tokens", "block_size", "num_blocks", "max_entries"]) < 1:
        raise ValueError("engine capacities and token budgets must be positive")
    if int(benchmark["repetitions"]) < 1 or int(benchmark["warmup_repetitions"]) < 0:
        raise ValueError("invalid measured/warm-up repetition count")
    if float(workload["arrival_interval_ms"]) < 0 or float(engine["max_wait_ms"]) < 0:
        raise ValueError("arrival interval and batching wait must be non-negative")
    if engine["decode_backend"] not in {"auto", "torch", "triton"}:
        raise ValueError("invalid decode backend")
    return payload


def validate_reference_identity(reference_dir: Path) -> None:
    """Require an existing, pinned corpus; never silently create a reference."""
    for bucket in ["short", "medium", "long"]:
        path = reference_dir / f"{bucket}.json"
        reference = json.loads(path.read_text())
        if reference.get("model_id") != MODEL_ID or reference.get("model_revision") != MODEL_REVISION or not reference.get("requests"):
            raise ValueError(f"invalid or empty pinned reference corpus: {path}")


def snapshot_sources(output_dir: Path, payload: dict[str, Any], reference_dir: Path) -> dict[str, Any]:
    """Keep the actual runtime source and inputs, including uncommitted files."""
    destination = output_dir / "source_snapshot" / "minillm_l4"
    files = [PROJECT_ROOT / "__init__.py", PROJECT_ROOT / "requirements.lock"]
    for directory in ["engine", "benchmarks", "configs"]:
        files.extend(p for p in (PROJECT_ROOT / directory).rglob("*") if p.is_file() and p.suffix in {".py", ".yaml", ".json"})
    digests = {}
    for path in sorted(files):
        relative = path.relative_to(PROJECT_ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        digests[str(Path("minillm_l4") / relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    for path, relative in [
        (_project_path(payload["workloads"]["dataset"]), Path("inputs/dataset.jsonl")),
        *[(reference_dir / f"{bucket}.json", Path(f"inputs/references/{bucket}.json")) for bucket in ["short", "medium", "long"]],
    ]:
        target = output_dir / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        digests[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    git = subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"], capture_output=True, text=True, check=False)
    provenance = {
        "git_commit_sha": git.stdout.strip() if git.returncode == 0 else None,
        "git_worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "snapshot_directory": "source_snapshot", "sha256": digests,
        "resolved_configuration": payload,
    }
    (output_dir / "run_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return provenance


def write_markdown(manifest: dict[str, Any], path: Path, output_dir: Path) -> None:
    def value(case, name, q="p50"):
        v = case["summary"]["metrics"].get(name, {}).get(q)
        return "—" if v is None else f"{v:,.2f}"

    payload = manifest["configuration"]
    benchmark = payload["benchmark"]
    lines = [
        "# Phase 8 — Chunked prefill measurements", "",
        f"Captured {manifest['created_at_utc']}. Model `{MODEL_ID}` at `{MODEL_REVISION}`.", "",
        f"One warm-up/measurement configuration across every case: {benchmark['warmup_repetitions']} warm-up runs and {benchmark['repetitions']} measured runs. Same prompts, outputs, arrivals, FP8 path, decode policy, page pool, and active limit. Prefix reuse enabled: {payload['engine']['enable_prefix']}. Loading and warm-up are excluded.", "",
        "Chunk 0 is the unchunked control, which permits an oversized prompt only when decode is idle. Positive chunk sizes respect the per-iteration token budget. Cases run in the recorded order, with the control first. Raw JSON retains environment/Git identity, request events, all percentiles, repeat variance, and scheduler/chunk records. `source_snapshot/` and `run_provenance.json` preserve the actual runtime source, input dataset, reference corpora, resolved configuration, and file hashes.", "",
        "| Workload | Chunk tokens | TTFT P50 / P95 / P99 (ms) | ITL P50 / P95 / P99 (ms) | TPOT P50 (ms) | E2E P95 (ms) | Output TPS P50 | Exact HF / control |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in manifest["cases"]:
        summary = case["summary"]
        lines.append("| " + " | ".join([
            case["workload"], str(case["chunk_size"]),
            " / ".join(value(case, "ttft_ms", q) for q in ["p50", "p95", "p99"]),
            " / ".join(value(case, "itl_ms", q) for q in ["p50", "p95", "p99"]),
            value(case, "tpot_ms"), value(case, "e2e_latency_ms", "p95"),
            f"{summary['tokens_per_second']['p50']:.2f}",
            ("pass" if case["correctness"]["status"] == "pass" else "FAIL") + " / " + ("pass" if case["matches_unchunked"] else "FAIL"),
        ]) + " |")
    lines += ["", "A failed correctness gate invalidates a performance claim; its measurements are retained for diagnosis. Three repetitions of this fixed corpus do not establish production tail latency.", "",
              "## Capacity and telemetry", "",
              "| Workload | Chunk tokens | Requests/s P50 | GPU utilization P50 (%) | Peak allocated / reserved GiB | Maximum chunk tokens | Chunks during decode |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for case in manifest["cases"]:
        summary = case["summary"]
        memory = summary["memory"]
        gib = " / ".join("—" if memory.get(key) is None else f"{memory[key] / 2**30:.2f}" for key in ["peak_allocated_bytes", "peak_reserved_bytes"])
        gpu = summary["gpu_utilization_percent"].get("p50")
        schedules = case["scheduler_runs"]
        lines.append(f"| {case['workload']} | {case['chunk_size']} | {summary['requests_per_second']['p50']:.3f} | {'—' if gpu is None else f'{gpu:.1f}'} | {gib} | {max((s['maximum_prefill_chunk_tokens'] for s in schedules), default=0)} | {max((s['prefill_chunks_while_decoding'] for s in schedules), default=0)} |")
    lines += ["", f"Raw sources: `{output_dir}`; configuration and case index: `chunked_prefill_manifest.json`.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = resolved_configuration(args)
    if args.dry_run:
        print(json.dumps({"configuration": payload, "workload_selection": args.workload, "reference_dir": str(args.reference_dir)}, indent=2))
        return
    validate_reference_identity(args.reference_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or PROJECT_ROOT / "results" / f"phase8_chunked_prefill_{stamp}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("choose an empty output directory to preserve saved runs")
    markdown = args.markdown_output or PROJECT_ROOT / "benchmark_results" / f"phase_08_chunked_prefill_{stamp}.md"
    if markdown.exists():
        raise FileExistsError("choose a new Markdown path to preserve saved reports")
    model_config, workload_config = payload["model"], payload["workloads"]
    engine, benchmark = payload["engine"], payload["benchmark"]
    device = str(model_config["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this benchmark in the L4 host environment")
    bundle = load_qwen_fp8(model_id=MODEL_ID, revision=MODEL_REVISION, model_path=args.model_path,
        device=device, local_files_only=bool(model_config.get("local_files_only", True)) and not args.allow_download,
        fp8_kernel_path=str(model_config["fp8_kernel_path"]))
    workloads = build_trace_workloads(bundle.tokenizer, _project_path(workload_config["dataset"]), workload_config,
        seed=int(workload_config["seed"]), model_id=MODEL_ID, revision=MODEL_REVISION, device=device,
        count_per_bucket_override=None, arrival_interval_ms=float(workload_config["arrival_interval_ms"]))
    harness = HarnessConfig(warmup_repetitions=int(benchmark["warmup_repetitions"]), repetitions=int(benchmark["repetitions"]),
        respect_arrival_schedule=True, collect_gpu=bool(benchmark["collect_gpu"]) and not args.no_gpu_sampling,
        collect_system_telemetry=bool(benchmark["collect_system_telemetry"]) and not args.no_system_telemetry,
        sample_interval_seconds=float(benchmark["sample_interval_seconds"]), seed=int(workload_config["seed"]),
        timer_overhead_iterations=int(benchmark["timer_overhead_iterations"]))
    manifest = {"schema_version": 1, "phase": 8, "project": "MiniLLM-L4", "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "model": bundle.metadata(),
        "configuration": payload, "reference_dir": str(args.reference_dir), "cases": []}
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest["provenance"] = snapshot_sources(output_dir, payload, args.reference_dir)
    manifest_path = output_dir / "chunked_prefill_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    all_correct = True
    for name in (tuple(workloads) if args.workload == "all" else (args.workload,)):
        workload = workloads[name]
        save_workload(workload, output_dir / f"workload_{name}.json")
        control = None
        for size in engine["chunk_sizes"]:
            runner = ChunkedPrefillPagedRunner(bundle.model, block_size=int(engine["block_size"]), num_blocks=int(engine["num_blocks"]),
                max_entries=int(engine["max_entries"]), max_batch_size=int(engine["max_batch_size"]),
                max_prefill_tokens=int(engine["max_prefill_tokens"]), max_wait_ms=float(engine["max_wait_ms"]),
                prefill_chunk_size=size or None, device=device, decode_backend=engine["decode_backend"],
                decode_sdpa_compat=bool(engine["decode_sdpa_compat"]), enable_prefix=bool(engine["enable_prefix"]))
            try:
                result = BenchmarkHarness(dataclasses.replace(harness, runner_name=runner.runner_name), benchmark_name="minillm_l4_chunked_prefill").run_trace(workload, runner)
                correctness = verify_concurrent_references(result, args.reference_dir)
                rows = _output_rows(result)
                if size == 0:
                    control = rows
                matches = rows == control and all(status == "completed" for run in rows for status, _ in run)
                schedules = runner.run_summaries[-int(benchmark["repetitions"]):]
                correct = (
                    correctness["status"] == "pass" and matches
                    and len(schedules) == int(benchmark["repetitions"])
                    and all(s["status"] == "completed" for s in schedules)
                )
                all_correct = all_correct and correct
                label = f"{name}_chunk{size}"
                result_path = output_dir / f"{label}.json"
                data = result.to_dict()
                data["chunked_prefill"] = {"correctness": correctness, "matches_unchunked": matches, "scheduler_runs": schedules, "model": manifest["model"]}
                result_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
                write_events_jsonl(result, output_dir / f"{label}_events.jsonl")
                manifest["cases"].append({"workload": name, "chunk_size": size, "result": str(result_path),
                    "correctness": correctness, "matches_unchunked": matches, "summary": result.summary, "scheduler_runs": schedules})
                manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
                write_markdown(manifest, markdown, output_dir)
                print(f"{name} chunk={size}: exact HF/control={'pass' if correct else 'FAIL'}; TPS={result.summary['tokens_per_second']['p50']:.2f}", flush=True)
            finally:
                runner.close()
                del runner
                gc.collect()
    manifest["status"] = "completed" if all_correct else "failed_correctness"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Results: {manifest_path}\nMarkdown: {markdown}")
    if not all_correct:
        raise RuntimeError("chunked prefill correctness gate failed; inspect saved results")


if __name__ == "__main__":
    main()
