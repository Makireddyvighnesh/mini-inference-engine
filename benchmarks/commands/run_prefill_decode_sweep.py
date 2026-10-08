"""Separate model prefill/decode timings and compare matched serving traces."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

from minillm_l4.benchmarks.core.harness import BenchmarkHarness, write_events_jsonl
from minillm_l4.benchmarks.core.schemas import HarnessConfig, RequestSpec, WorkloadSpec
from minillm_l4.benchmarks.core.synthetic import SyntheticSample, exact_token_ids
from minillm_l4.benchmarks.core.workloads import save_workload
from minillm_l4.benchmarks.runners.chunked_prefill import ChunkedPrefillPagedRunner
from minillm_l4.benchmarks.runners.huggingface_baseline import MODEL_ID, MODEL_REVISION, load_qwen_fp8
from minillm_l4.benchmarks.runners.manual_decode import ManualGreedyBatchRunner
from minillm_l4.benchmarks.runners.phase_sweep import MeteredRunner, StaticPagedBatchRunner, StaticPagedTraceRunner
from minillm_l4.configs.loader import load_yaml_config
from minillm_l4.engine.generation.huggingface import transformers_greedy_generate
from .run_chunked_prefill import snapshot_sources
from .run_concurrent_requests import PROJECT_ROOT


MODES = ("prefill", "isolated", "static", "continuous", "chunked_128", "chunked_256", "chunked_512",
         "mixed_512", "mixed_2048", "adaptive")
# capped: batch size caps in-flight requests and sizes the page pool (original sweep).
# resource: batch size is offered load; admission is limited only by KV pages,
# the pool fits every request, and a chunk size is the per-iteration prompt budget.
# mixed_N: one forward per iteration holding every decode token plus prompt
# chunks (shortest first) within an N-token budget, in either admission policy.
ADMISSION_POLICIES = ("capped", "resource")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/workloads/qwen3_fp8_phase_sweep.yaml")
    parser.add_argument("--prompt-lengths", type=int, nargs="+")
    parser.add_argument("--generation-lengths", type=int, nargs="+")
    parser.add_argument("--batch-sizes", type=int, nargs="+")
    parser.add_argument("--modes", choices=MODES, nargs="+")
    parser.add_argument("--admission", choices=ADMISSION_POLICIES)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--warmup-repetitions", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wait-for-idle-gpu", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-gpu-sampling", action="store_true")
    return parser.parse_args(argv)


def configuration(args):
    data = load_yaml_config(args.config, expected_phase=8)
    if data["model"]["id"] != MODEL_ID or data["model"]["revision"] != MODEL_REVISION:
        raise ValueError("the phase sweep requires the pinned model")
    workloads = data["workloads"]
    for key in ["prompt_lengths", "generation_lengths", "batch_sizes"]:
        values = getattr(args, key) or workloads[key]
        if not values or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values):
            raise ValueError(f"{key} must contain positive integers")
        workloads[key] = list(dict.fromkeys(values))
    if max(workloads["generation_lengths"]) > 1048:
        raise ValueError("generation lengths must not exceed the requested 1048-token maximum")
    data["engine"]["modes"] = list(dict.fromkeys(args.modes or data["engine"]["modes"]))
    if any(mode not in MODES for mode in data["engine"]["modes"]):
        raise ValueError("invalid sweep mode")
    for key in ["repetitions", "warmup_repetitions"]:
        value = getattr(args, key)
        if value is not None:
            data["benchmark"][key] = value
    data["engine"]["admission"] = args.admission or data["engine"].get("admission", "capped")
    if data["engine"]["admission"] not in ADMISSION_POLICIES:
        raise ValueError("invalid admission policy")
    if data["engine"]["admission"] == "capped":
        data["engine"].pop("admission")  # keep saved capped configurations identical
    if data["benchmark"]["repetitions"] < 1 or data["benchmark"]["warmup_repetitions"] < 0:
        raise ValueError("invalid repetition count")
    if data["engine"]["block_size"] < 1 or data["engine"]["max_prefill_tokens"] < 1:
        raise ValueError("page size and prefill budget must be positive")
    if workloads["mixed_arrival_interval_ms"] < 0 or workloads["mixed_short_output_tokens"] < 1:
        raise ValueError("invalid mixed trace parameters")
    return data


def planned_cases(data):
    w = data["workloads"]
    prefill = [{"key": f"p{p}_g1_b{b}_prefill", "prompt_tokens": p, "generation_cap": 1,
                "batch_limit": b, "mode": "prefill"} for p in w["prompt_lengths"]
               for b in w["batch_sizes"]] if "prefill" in data["engine"]["modes"] else []
    return prefill + [{"key": f"p{p}_g{g}_b{b}_{mode}", "prompt_tokens": p,
             "generation_cap": g, "batch_limit": b, "mode": mode}
            for p in w["prompt_lengths"] for g in w["generation_lengths"]
            for b in w["batch_sizes"] for mode in data["engine"]["modes"] if mode != "prefill"]


def build_prompts(tokenizer, data):
    lengths = sorted(set(data["workloads"]["prompt_lengths"]) | {min(128, min(data["workloads"]["prompt_lengths"]))})
    return {str(length): exact_token_ids(tokenizer, SyntheticSample(
        sample_id=f"phase-sweep-{length}", category="controlled-phase-probe",
        target_prompt_tokens=length, target_output_tokens=max(data["workloads"]["generation_lengths"]),
        seed_text="Explain how prefill, cached decoding, and batching affect language-model inference. Give concrete comparisons.",
    )) for length in lengths}


def build_workload(case, prompts, data):
    p, g, b = case["prompt_tokens"], case["generation_cap"], case["batch_limit"]
    short = min(128, min(data["workloads"]["prompt_lengths"]))
    if case["mode"] in {"isolated", "prefill"}:
        requests = tuple(RequestSpec(f"{case['key']}-r{i}", tuple(prompts[str(p)]), g, category=f"p{p}") for i in range(b))
    else:
        requests = [RequestSpec(f"{case['key']}-r0", tuple(prompts[str(short)]), g, category="decode_anchor")]
        arrivals = b if data["engine"].get("admission") == "resource" else max(3, max(data["workloads"]["batch_sizes"]) + 1) - 1
        for index in range(1, arrivals + 1):
            length = p if index % 2 else short
            requests.append(RequestSpec(f"{case['key']}-r{index}", tuple(prompts[str(length)]),
                min(g, data["workloads"]["mixed_short_output_tokens"]),
                scheduled_arrival_ms=index * data["workloads"]["mixed_arrival_interval_ms"], category=f"p{length}"))
        requests = tuple(requests)
    return WorkloadSpec(name=case["key"], seed=data["workloads"]["seed"], requests=requests,
        model_id=MODEL_ID, model_revision=MODEL_REVISION, dtype="fp8", device=data["model"]["device"],
        arrival_pattern="closed_loop" if case["mode"] in {"isolated", "prefill"} else "fixed_rate",
        metadata={"generation_cap": g, "profile": "same-length static phase isolation" if case["mode"] in {"isolated", "prefill"} else "long decoding anchor with later prefills and shorter completions"})


def gpu_occupants():
    query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    if query.returncode:
        raise RuntimeError("GPU driver query failed; use the L4 host execution context")
    occupants = []
    for line in query.stdout.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if fields and fields[0].isdigit() and int(fields[0]) != os.getpid():
            occupants.append({"pid": int(fields[0]), "process_name": fields[1], "memory_mib": fields[2]})
    return occupants


def verify_reference(result, references):
    errors, checked, prior = [], 0, {}
    specs = {r.request_id: r for r in result.workload.requests}
    for run in result.runs:
        for row in run["requests"]:
            request = specs[row["request_id"]]
            expected = references.get(request.prompt_sha256)
            outcome = row["outcome"]
            tokens = outcome["generated_token_ids"]
            checked += 1
            if outcome["status"] != "completed" or not expected or tokens != expected["generated_token_ids"][:request.max_new_tokens] or len(tokens) != request.max_new_tokens:
                errors.append(f"{request.request_id}: exact HF prefix or completion check failed")
            if request.request_id in prior and prior[request.request_id] != tokens:
                errors.append(f"{request.request_id}: output changed across repetitions")
            prior[request.request_id] = tokens
    return {"status": "pass" if not errors else "fail", "checked_outputs": checked,
            "origin": "independent HF model.generate; exact prefix for shorter requests", "errors": errors}


def phase_summary(result, forwards):
    prefill, decode, decode_rates, input_rates = [], [], [], []
    for run, timing in zip(result.runs, forwards, strict=True):
        rows = run["requests"]
        prefill.append(statistics.median(row["metrics"]["prefill_ms"] for row in rows if row["metrics"]["prefill_ms"] is not None) if any(row["metrics"]["prefill_ms"] is not None for row in rows) else None)
        endpoints = {}
        for event in run["events"]:
            if event["event"] == "token_ready":
                endpoints.setdefault(event["request_id"], []).append(event["timestamp_ns"])
        if endpoints:
            start = min(min(times) for times in endpoints.values())
            end = max(max(times) for times in endpoints.values())
            duration = (end - start) / 1e6
            decode.append(duration)
            tokens = sum(max(0, row["outcome"]["generated_tokens"] - 1) for row in rows)
            decode_rates.append(tokens / (duration / 1000) if duration > 0 else None)
        else:
            decode.append(None)
            decode_rates.append(None)
        cuda = timing["prefill"]["cuda_elapsed_total_ms"]
        input_rates.append(sum(r.prompt_tokens for r in result.workload.requests) / (cuda / 1000) if cuda else None)
    median = lambda values: statistics.median(v for v in values if v is not None) if any(v is not None for v in values) else None
    return {"prefill_request_wall_ms_p50": median(prefill), "decode_window_wall_ms_p50": median(decode),
            "decode_window_tokens_per_second_p50": median(decode_rates), "prefill_forward_input_tokens_per_second_p50": median(input_rates),
            "prefill_forward_cuda_total_ms_p50": median([f["prefill"]["cuda_elapsed_total_ms"] for f in forwards]),
            "decode_forward_cuda_total_ms_p50": median([f["decode"]["cuda_elapsed_total_ms"] for f in forwards])}


def write_report(manifest, path):
    def fmt(v):
        return "—" if v is None else f"{v:,.2f}"
    lines = ["# Prefill, decode, static batching, and continuous batching sweep", "",
             f"Status: **{manifest['status']}**. Created {manifest['created_at_utc']}. Model `{MODEL_ID}` at `{MODEL_REVISION}`.", "",
             "Dedicated pure prefill generates only the first token and has no cached decode. Isolated static generation separates prefill from decode after KV setup and the first token; 1048 outputs contain 1047 cached decode steps. These isolation rows use the native Transformers DynamicCache, while the matched mixed policies all use paged KV. Differences between the isolated and serving rows therefore include the cache backend. CUDA forward elapsed time excludes token sampling and page-copy code, includes stream launch gaps, and is not kernel-busy time. Request wall timings include scheduling and output synchronization.", "",
             "Static and continuous mixed rows use identical request lists, arrival times, output limits, FP8 projections, paged KV storage, and decode numerics. The mixed profile has a long decoding anchor and later requests capped at 128 outputs. All row-generation limits stay at or below the reported cap. Prefix reuse and CUDA Graph replay are disabled.", "",
             "Repeated prompts within an isolated batch are intentional shape controls, not a diverse serving corpus. Case order is fixed. Raw records preserve P50/P95/P99, repeat variance, forward phase records, model outputs, telemetry, references, and source snapshots. Failed or contended rows are not accepted performance claims.", "",
             "## Isolated prefill and decode", "",
             "| Prompt | Batch | Output tokens | Mode | Prefill wall P50 (ms) | Prefill forward CUDA (ms) | Prefill input tokens/s | Decode wall (ms) | Decode tokens/s | TPOT P50 (ms) | Exact HF |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in manifest["completed_cases"]:
        if c["mode"] not in {"isolated", "prefill"}:
            continue
        p = c.get("phase_summary", {})
        metrics = c.get("summary", {}).get("metrics", {})
        lines.append("| " + " | ".join([str(c["prompt_tokens"]), str(c["batch_limit"]), str(c["generation_cap"]), c["mode"],
            fmt(p.get("prefill_request_wall_ms_p50")), fmt(p.get("prefill_forward_cuda_total_ms_p50")), fmt(p.get("prefill_forward_input_tokens_per_second_p50")),
            fmt(p.get("decode_window_wall_ms_p50")), fmt(p.get("decode_window_tokens_per_second_p50")), fmt(metrics.get("tpot_ms", {}).get("p50")), c["status"]]) + " |")
    lines += ["", "## Matched mixed traces", "",
              "| Largest prompt | Batch limit | Generation cap | Mode | TTFT P50 / P95 (ms) | ITL P95 / P99 (ms) | E2E P95 (ms) | Output tokens/s | GPU P50 (%) | Peak allocated GiB | Exact HF |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in manifest["completed_cases"]:
        if c["mode"] in {"isolated", "prefill"}:
            continue
        s = c.get("summary", {})
        metric = lambda k, q="p50": s.get("metrics", {}).get(k, {}).get(q)
        allocated = s.get("memory", {}).get("peak_allocated_bytes")
        lines.append("| " + " | ".join([str(c["prompt_tokens"]), str(c["batch_limit"]), str(c["generation_cap"]), c["mode"],
            fmt(metric("ttft_ms")) + " / " + fmt(metric("ttft_ms", "p95")), fmt(metric("itl_ms", "p95")) + " / " + fmt(metric("itl_ms", "p99")),
            fmt(metric("e2e_latency_ms", "p95")), fmt(s.get("tokens_per_second", {}).get("p50")), fmt(s.get("gpu_utilization_percent", {}).get("p50")),
            fmt(allocated / 2**30 if allocated is not None else None), c["status"]]) + " |")
    if not manifest["completed_cases"]:
        lines += ["", "No timings have been collected. Planned cases: " + str(len(manifest["planned_cases"])) + "."]
    lines += ["", "## Availability and failures", "", *[f"- `{c['key']}`: {c['status']}; {c.get('reason', '')}" for c in manifest["completed_cases"] if c["status"] != "pass"],
              "", f"Raw evidence: `{manifest['output_dir']}`.", ""]
    if manifest.get("gpu_occupants"):
        lines.append("GPU occupied by: " + json.dumps(manifest["gpu_occupants"]) + "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def checkpoint(manifest, output, markdown):
    (output / "phase_sweep_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "run_provenance.json").write_text(json.dumps(manifest["provenance"], indent=2, sort_keys=True) + "\n")
    write_report(manifest, markdown)


def main(argv=None):
    args = parse_args(argv)
    data = configuration(args)
    cases = planned_cases(data)
    if args.dry_run:
        print(json.dumps({"configuration": data, "case_count": len(cases), "cases": cases}, indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = args.output_dir or PROJECT_ROOT / "results" / f"phase_sweep_{stamp}"
    markdown = args.markdown_output or PROJECT_ROOT / "benchmark_results" / f"prefill_decode_sweep_{stamp}.md"
    manifest_path = output / "phase_sweep_manifest.json"
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if manifest["configuration"] != data:
            raise ValueError("resume requires the identical saved configuration")
        markdown = Path(manifest["markdown_output"])
        prompts = json.loads((output / "prompts.json").read_text())
    else:
        if output.exists() and any(output.iterdir()) or markdown.exists():
            raise FileExistsError("choose new output paths or use --resume")
        output.mkdir(parents=True, exist_ok=True)
        (output / "configuration.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
        prompts = build_prompts(tokenizer, data)
        (output / "prompts.json").write_text(json.dumps(prompts, indent=2) + "\n")
        manifest = {"schema_version": 1, "project": "MiniLLM-L4", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "prepared", "configuration": data, "output_dir": str(output), "markdown_output": str(markdown),
            "planned_cases": cases, "completed_cases": []}
        for case in cases:
            save_workload(build_workload(case, prompts, data), output / f"{case['key']}_workload.json")
        manifest["provenance"] = snapshot_sources(output, data, PROJECT_ROOT / "results/phase1/references_baseline")
        target = output / "source_snapshot/inputs/prompts.json"
        shutil.copyfile(output / "prompts.json", target)
        manifest["provenance"]["sha256"]["inputs/prompts.json"] = hashlib.sha256(target.read_bytes()).hexdigest()
    if hashlib.sha256((output / "prompts.json").read_bytes()).hexdigest() != manifest["provenance"]["sha256"]["inputs/prompts.json"]:
        raise ValueError("saved prompt inputs changed since preparation")
    if args.resume and "inputs/hf_references.json" in manifest["provenance"]["sha256"]:
        if hashlib.sha256((output / "hf_references.json").read_bytes()).hexdigest() != manifest["provenance"]["sha256"]["inputs/hf_references.json"]:
            raise ValueError("saved HF references changed since the last checkpoint")
    checkpoint(manifest, output, markdown)
    if args.prepare_only:
        print(f"Prepared {len(cases)} cases: {manifest_path}")
        return
    device = data["model"]["device"]
    if device.startswith("cuda"):
        while True:
            occupants = gpu_occupants()
            if not occupants:
                break
            manifest.update(status="waiting_for_gpu", gpu_occupants=occupants)
            checkpoint(manifest, output, markdown)
            if not args.wait_for_idle_gpu:
                raise RuntimeError("the GPU is occupied; resume with --wait-for-idle-gpu after reviewing the saved plan")
            print("Waiting for exclusive GPU access: " + json.dumps(occupants), flush=True)
            time.sleep(30)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use the host execution context")
    # Never overwrite earlier results after a code change.
    for relative, digest in manifest["provenance"]["sha256"].items():
        if relative.startswith("minillm_l4/"):
            current = PROJECT_ROOT / relative.removeprefix("minillm_l4/")
            if hashlib.sha256(current.read_bytes()).hexdigest() != digest:
                raise ValueError("runtime source changed since preparation; use a new output directory")
    bundle = load_qwen_fp8(model_id=MODEL_ID, revision=MODEL_REVISION, device=device,
        local_files_only=True, fp8_kernel_path=data["model"]["fp8_kernel_path"])
    manifest["model"] = bundle.metadata()
    manifest.pop("gpu_occupants", None)
    reference_path = output / "hf_references.json"
    references = json.loads(reference_path.read_text()) if reference_path.exists() else {}
    manifest["status"] = "building_hf_references"
    manifest["reference_count"] = len(references)
    checkpoint(manifest, output, markdown)
    for length, tokens in prompts.items():
        request = RequestSpec(f"reference-p{length}", tuple(tokens), max(data["workloads"]["generation_lengths"]))
        if request.prompt_sha256 in references:
            reference = references[request.prompt_sha256]
            if (reference.get("model_id") != MODEL_ID or reference.get("revision") != MODEL_REVISION
                    or reference.get("origin") != "HF model.generate"
                    or reference.get("output_cap", 0) < request.max_new_tokens
                    or len(reference.get("generated_token_ids", [])) != reference.get("output_cap")):
                raise ValueError("saved sweep reference has inconsistent identity or output length")
            continue
        print(f"HF reference: prompt={length}, outputs={request.max_new_tokens}", flush=True)
        manifest["active_reference_prompt_tokens"] = int(length)
        checkpoint(manifest, output, markdown)
        generation = transformers_greedy_generate(bundle.model, {
            "input_ids": torch.tensor([tokens], device=device), "attention_mask": torch.ones((1, len(tokens)), dtype=torch.long, device=device),
        }, output_tokens=request.max_new_tokens)
        references[request.prompt_sha256] = {"prompt_tokens": len(tokens), "generated_token_ids": generation.token_ids[0].tolist(),
            "model_id": MODEL_ID, "revision": MODEL_REVISION, "origin": "HF model.generate", "output_cap": request.max_new_tokens}
        reference_path.write_text(json.dumps(references, indent=2, sort_keys=True) + "\n")
        manifest["reference_count"] = len(references)
        checkpoint(manifest, output, markdown)
    shutil.copyfile(reference_path, output / "source_snapshot/inputs/hf_references.json")
    manifest["provenance"]["sha256"]["inputs/hf_references.json"] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    manifest["status"] = "running"
    checkpoint(manifest, output, markdown)
    for case in cases:
        if any(c["key"] == case["key"] for c in manifest["completed_cases"]):
            continue
        if device.startswith("cuda") and gpu_occupants():
            manifest.update(status="waiting_for_gpu", gpu_occupants=gpu_occupants())
            checkpoint(manifest, output, markdown)
            raise RuntimeError("another GPU job started; saved results can be resumed later")
        print(f"Case {case['key']} starting", flush=True)
        workload = build_workload(case, prompts, data)
        block = data["engine"]["block_size"]
        largest = max(r.prompt_tokens + r.max_new_tokens - 1 for r in workload.requests)
        resource = data["engine"].get("admission") == "resource" and case["mode"] not in {"isolated", "prefill"}
        if resource:
            blocks = sum(math.ceil((r.prompt_tokens + r.max_new_tokens - 1) / block) for r in workload.requests)
            active_limit = len(workload.requests)
        else:
            blocks = case["batch_limit"] * math.ceil(largest / block)
            active_limit = case["batch_limit"]
        if device.startswith("cuda"):
            cfg = bundle.model.config
            per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * 2
            dense_slots = sum(r.prompt_tokens for r in workload.requests) if resource else case["batch_limit"] * (
                largest if case["mode"] == "isolated" else max(r.prompt_tokens for r in workload.requests)
            )
            pool_slots = 0 if case["mode"] in {"isolated", "prefill"} else blocks * block
            extra = (pool_slots + dense_slots) * per_token + 2 * 2**30
            if extra * 1.1 > torch.cuda.mem_get_info(device)[0]:
                manifest["completed_cases"].append({**case, "status": "skipped_memory_guard", "reason": "estimated page pool + dense prefill cache + workspace exceeds free memory"})
                checkpoint(manifest, output, markdown)
                continue
        chunk = int(case["mode"].split("_")[-1]) if case["mode"].startswith("chunked_") else None
        budget = int(case["mode"].split("_")[-1]) if case["mode"].startswith("mixed_") else None
        options = dict(block_size=block, num_blocks=blocks, max_batch_size=active_limit,
            max_prefill_tokens=budget or (chunk if resource and chunk else data["engine"]["max_prefill_tokens"]), device=device,
            decode_backend=data["engine"]["decode_backend"], decode_sdpa_compat=data["engine"]["decode_sdpa_compat"], enable_prefix=False)
        if case["mode"] in {"isolated", "prefill"}:
            backend = ManualGreedyBatchRunner(bundle.model, device=device)
            runner = backend
        elif case["mode"] == "static":
            backend = StaticPagedBatchRunner(bundle.model, **options)
            runner = backend if case["mode"] == "isolated" else StaticPagedTraceRunner(backend)
        else:
            adaptive = case["mode"] == "adaptive"
            backend = ChunkedPrefillPagedRunner(bundle.model, prefill_chunk_size=chunk,
                                                mixed_batch=budget is not None or adaptive,
                                                adaptive_chunking=adaptive, **options)
            runner = backend
        wrapped = MeteredRunner(runner, bundle.model, device)
        bench = data["benchmark"]
        harness = BenchmarkHarness(HarnessConfig(warmup_repetitions=bench["warmup_repetitions"], repetitions=bench["repetitions"],
            respect_arrival_schedule=case["mode"] not in {"isolated", "prefill"}, collect_gpu=bench["collect_gpu"] and not args.no_gpu_sampling,
            collect_system_telemetry=bench["collect_system_telemetry"], sample_interval_seconds=bench["sample_interval_seconds"],
            timer_overhead_iterations=bench["timer_overhead_iterations"], runner_name=case["mode"], seed=data["workloads"]["seed"]))
        try:
            result = harness.run_batched(workload, case["batch_limit"], wrapped) if case["mode"] in {"isolated", "prefill"} else harness.run_trace(workload, wrapped)
            forwards = wrapped.run_summaries[-bench["repetitions"]:]
            gate = verify_reference(result, references)
            contended = bool(gpu_occupants()) if device.startswith("cuda") else False
            record = {**case, "status": "contended" if contended else gate["status"], "correctness": gate,
                "summary": result.summary, "phase_summary": phase_summary(result, forwards), "forward_runs": forwards,
                "scheduler": getattr(runner, "last_summary", None), "result": str(output / f"{case['key']}.json")}
            record["phase_summary"]["decode_window_has_interleaved_prefill"] = case["mode"] not in {"isolated", "prefill"}
            record["cache_backend"] = "transformers_dynamic" if case["mode"] in {"isolated", "prefill"} else "paged_sdpa_compat"
            raw = result.to_dict()
            raw["phase_sweep"] = record
            Path(record["result"]).write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
            write_events_jsonl(result, output / f"{case['key']}_events.jsonl")
            manifest["completed_cases"].append(record)
            checkpoint(manifest, output, markdown)
            print(f"Case {case['key']}: {record['status']}", flush=True)
        finally:
            if hasattr(backend, "close"):
                backend.close()
            del wrapped, runner, backend
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    manifest["status"] = "completed" if all(c["status"] == "pass" for c in manifest["completed_cases"]) else "completed_with_failures_or_skips"
    checkpoint(manifest, output, markdown)
    print(f"Results: {manifest_path}\nMarkdown: {markdown}")
    if any(c["status"] in {"fail", "contended"} for c in manifest["completed_cases"]):
        raise RuntimeError("one or more sweep cases failed correctness or exclusivity; inspect the saved manifest")


if __name__ == "__main__":
    main()
