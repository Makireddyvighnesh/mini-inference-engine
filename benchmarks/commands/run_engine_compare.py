"""Hugging Face generate() vs vLLM vs SGLang vs MiniLLM-L4 at 512, 1,024 and 2,048 prompt tokens.

One command runs everything (each engine in its own process and Python env):

  .conda-env/bin/python minillm_l4/benchmarks/commands/run_engine_compare.py \\
      --output-dir minillm_l4/results/engine_compare_<date>

Matched for every engine: pinned checkpoint, prompt token IDs (saved once in
workloads.json), output length (EOS ignored), greedy decoding, prefix caching
off, requests sent all at once, 1 warm-up + 3 measured runs per case.
Each engine keeps its own defaults otherwise:

  hf       stock Transformers FP8 path, one static batch through model.generate
  vllm     torch.compile + CUDA graphs, chunked prefill, its FP8 kernels
  sglang   sgl.Engine defaults (its CUDA graphs, overlap schedule, FP8 kernels)
  minillm  our sm89 FP8, fused and merged kernels, mixed batching, adaptive
           chunks, CUDA Graph decode

Fairness on the power-capped L4: engines cannot alternate replays because each
needs its own process, so rounds alternate the order (round 1 hf, vllm, sglang,
minillm; round 2 reversed) and each session starts only when the GPU is free and
cool. nvidia-smi logs power and clocks from a separate process for every
session. Compare the two rounds in comparison.md before trusting a small gap.

Timestamps: hf and minillm when the token is read back on the host in the same
process; vllm and sglang when the streamed token reaches the caller, so their
numbers include engine IPC. TTFT, TPOT and tok/s definitions are the ones in
run_vllm_compare.py.

SGLang env (.sglang-env; it pins torch 2.14, so it cannot share an env). FlashInfer
and SGLang JIT-compile kernels, so the CUDA toolkit packages must all be 13.0, and
glibc >= 2.42 needs a fix NVIDIA only shipped in later CUDA headers:

  uv venv --python 3.12 .sglang-env
  VIRTUAL_ENV=$PWD/.sglang-env uv pip install --prerelease=if-necessary-or-explicit \\
      --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130 \\
      sglang==0.5.23 "flash-attn-4>=4.0.0b18" nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 \\
      "nvidia-nvvm==13.0.*"
  # then in .sglang-env/lib/python3.12/site-packages/nvidia/cu13/include/crt/math_functions.h
  # declare rsqrt and rsqrtf noexcept(true) when __GLIBC_PREREQ(2,42), as CUDA 13.4 does.
  # Replace the file instead of editing in place: uv hard-links it to its cache.

Only the standard library is imported at module level so every engine env can
run this file directly.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

if __package__:
    from . import run_vllm_compare as base
else:
    import run_vllm_compare as base

THIS_FILE = Path(__file__).resolve()
ROOT = base.ROOT
ENGINES = ("hf", "vllm", "sglang", "minillm")
LABELS = {"hf": "HF generate", "vllm": "vLLM", "sglang": "SGLang", "minillm": "MiniLLM (ours)"}
PROMPT_LENGTHS = (512, 1024, 2048)
BATCH_SIZES = (1, 8)
CONDA_PYTHON = ROOT / ".conda-env/bin/python"
VLLM_ENV = Path("/home/ubuntu/miniconda3/envs/nemo35_vllm")
SGLANG_ENV = ROOT / ".sglang-env"
GPU_FIELDS = ("temperature.gpu", "power.draw", "clocks.sm", "clocks.mem", "memory.used", "utilization.gpu",
              "clocks_event_reasons.sw_power_cap")
COOL_TEMP_C = 50
COOL_TIMEOUT_S = 240


def engine_environment(engine: str) -> tuple[Path, dict[str, str]]:
    """The interpreter and environment variables one engine session needs."""

    env = dict(os.environ)
    if engine in ("hf", "minillm"):
        return CONDA_PYTHON, env
    if engine == "vllm":
        env.update({"PATH": f"{VLLM_ENV}/bin:{env['PATH']}", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                    "VLLM_USE_FLASHINFER_SAMPLER": "0",
                    "LD_LIBRARY_PATH": f"{VLLM_ENV}/lib/python3.12/site-packages/nvidia/cu13/lib"})
        return VLLM_ENV / "bin/python", env
    if engine == "sglang":
        # FlashInfer JIT-compiles its kernels and needs nvcc >= 12.8; /usr/bin/nvcc is 12.4.
        env["PATH"] = f"{SGLANG_ENV}/lib/python3.12/site-packages/nvidia/cu13/bin:{SGLANG_ENV}/bin:{env['PATH']}"
        return SGLANG_ENV / "bin/python", env
    raise ValueError(f"unknown engine {engine}")


def session_order(round_index: int) -> tuple[str, ...]:
    """Round 1 runs in ENGINES order, round 2 reversed, and so on."""

    return ENGINES if round_index % 2 == 1 else tuple(reversed(ENGINES))


def build_workloads(tokenizer, output_tokens: int) -> list[dict[str, Any]]:
    from minillm_l4.benchmarks.commands.run_showcase import prompt

    cases = []
    for length in PROMPT_LENGTHS:
        for batch in BATCH_SIZES:
            name = f"p{length}_b{batch}"
            cases.append({"section": "compare", "name": name, "prompt_tokens": length, "batch": batch,
                          "output_tokens": output_tokens,
                          "label": f"{length:,}-token prompt, {batch} request{'s' if batch > 1 else ''}",
                          "requests": [{"id": f"{name}-r{i}", "prompt_token_ids": list(prompt(tokenizer, length, f"r{i}")),
                                        "max_new_tokens": output_tokens, "arrival_ms": 0.0} for i in range(batch)]})
    return cases


# ----------------------------------------------------------------------------- one engine session

def package_versions(*names: str) -> dict[str, str | None]:
    import importlib.metadata as metadata

    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


class _TimedStreamer:
    """Records when generate() hands each token row to the host."""

    def __init__(self, recorder: list[tuple[int, list[int]]]) -> None:
        self.recorder = recorder
        self.prompt_seen = False

    def put(self, value) -> None:
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        self.recorder.append((time.perf_counter_ns(), value.view(-1).tolist()))

    def end(self) -> None:
        return None


def hf_case(model, work: dict[str, Any]) -> dict[str, Any]:
    import torch

    from minillm_l4.engine.generation.huggingface import transformers_greedy_generate

    requests = work["requests"]
    steps: list[tuple[int, list[int]]] = []
    error = None
    start = time.perf_counter_ns()
    try:
        input_ids = torch.tensor([r["prompt_token_ids"] for r in requests], dtype=torch.long, device="cuda")
        transformers_greedy_generate(model, {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)},
                                     output_tokens=requests[0]["max_new_tokens"], logits_mode="last",
                                     streamer=_TimedStreamer(steps))
    except torch.cuda.OutOfMemoryError as exc:
        error = f"CUDA OOM: {exc}"
        torch.cuda.empty_cache()
    times = [t for t, _ in steps]
    gaps = [(b - a) / 1e6 for a, b in zip(times, times[1:])]
    rows = []
    for index, request in enumerate(requests):
        tokens = [row[index] for _, row in steps]
        rows.append({"id": request["id"], "ttft_ms": (times[0] - start) / 1e6 if times else None,
                     "tpot_ms": (times[-1] - times[0]) / 1e6 / (len(times) - 1) if len(times) > 1 else None,
                     "max_gap_ms": max(gaps) if gaps else None, "tokens": len(tokens),
                     "expected_tokens": request["max_new_tokens"], "token_ids": tokens,
                     "status": "completed" if error is None and len(tokens) == request["max_new_tokens"] else "failed",
                     "error": error})
    return {"duration_ms": (times[-1] - start) / 1e6 if times else 0.0, "requests": rows}


def run_hf(args, workloads: list[dict[str, Any]]) -> None:
    import gc

    import torch

    from minillm_l4.benchmarks.runners.huggingface_baseline import load_qwen_fp8

    bundle = load_qwen_fp8(fp8_kernel_path="auto", local_files_only=True)
    results: dict[str, Any] = {"engine": "hf", "config": {
        "fp8_kernel_path": "auto", "kernel_strategy": bundle.kernel_strategy,
        "attn_implementation": getattr(bundle.model.config, "_attn_implementation", None),
        "logits_to_keep": 1, "torch": torch.__version__,
        "versions": package_versions("transformers", "torch", "triton")}, "cases": {}}
    for work in workloads:
        retries = torch.cuda.memory_stats().get("num_alloc_retries", 0)
        for _ in range(args.warmup_repetitions):
            hf_case(bundle.model, work)
        runs = [hf_case(bundle.model, work) for _ in range(args.repetitions)]
        entry = {**base.summarize(runs, expected=len(work["requests"]) * args.repetitions),
                 "alloc_retries": torch.cuda.memory_stats().get("num_alloc_retries", 0) - retries, "runs": runs}
        entry.update(base.case_validity(entry))
        results["cases"][work["name"]] = entry
        (args.output_dir / "hf.json").write_text(json.dumps(results) + "\n")
        base.print_invalid_case(f"[hf] {work['name']}", entry)
        print_case("hf", work["name"], entry)
        gc.collect()
        torch.cuda.empty_cache()
    base.exit_if_invalid(results["cases"].values())


async def _sglang_case(engine, work: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter_ns()

    async def one(request):
        params = {"max_new_tokens": request["max_new_tokens"], "temperature": 0.0, "ignore_eos": True}
        tokens, times = [], []
        error = None
        try:
            stream = await engine.async_generate(input_ids=request["prompt_token_ids"], sampling_params=params,
                                                 stream=True)
            async for output in stream:
                delta = list(output.get("output_ids") or [])
                if delta:
                    now = time.perf_counter_ns()
                    tokens += delta
                    times += [now] * len(delta)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        gaps = [(b - a) / 1e6 for a, b in zip(times, times[1:])]
        return {"id": request["id"], "ttft_ms": (times[0] - start) / 1e6 if times else None,
                "tpot_ms": (times[-1] - times[0]) / 1e6 / (len(times) - 1) if len(times) > 1 else None,
                "max_gap_ms": max(gaps) if gaps else None, "tokens": len(tokens),
                "expected_tokens": request["max_new_tokens"], "token_ids": tokens,
                "status": "completed" if error is None and len(tokens) == request["max_new_tokens"] else "failed",
                "error": error, "last_ns": times[-1] if times and error is None else time.perf_counter_ns()}

    rows = await asyncio.gather(*(one(r) for r in work["requests"]))
    return {"duration_ms": (max(r.pop("last_ns") for r in rows) - start) / 1e6, "requests": rows}


async def _run_sglang(args, workloads: list[dict[str, Any]]) -> None:
    import sglang

    engine = sglang.Engine(
        model_path=base.MODEL_PATH, context_length=base.MAX_MODEL_LEN, max_running_requests=base.MAX_IN_FLIGHT,
        disable_radix_cache=True, skip_tokenizer_init=True, incremental_streaming_output=True, random_seed=0,
        log_level="info")
    # The scheduler resolves "auto" settings at startup; engine.server_args still holds the unset values.
    server = engine.tokenizer_manager.server_args.resolved_dict()
    results: dict[str, Any] = {"engine": "sglang", "config": {
        "versions": package_versions("sglang", "torch", "transformers", "flashinfer-python", "triton"),
        "max_running_requests": base.MAX_IN_FLIGHT, "context_length": base.MAX_MODEL_LEN, "prefix_cache": False,
        **{key: str(server.get(key)) for key in (
            "attention_backend", "sampling_backend", "fp8_gemm_runner_backend", "quantization", "mem_fraction_static",
            "chunked_prefill_size", "max_prefill_tokens", "disable_overlap_schedule", "disable_cuda_graph",
            "enable_torch_compile", "stream_interval")}}, "cases": {}}
    print(f"[sglang] config: {json.dumps(results['config'])}", flush=True)
    try:
        for work in workloads:
            for _ in range(args.warmup_repetitions):
                await _sglang_case(engine, work)
            runs = [await _sglang_case(engine, work) for _ in range(args.repetitions)]
            entry = {**base.summarize(runs, expected=len(work["requests"]) * args.repetitions), "runs": runs}
            results["cases"][work["name"]] = entry
            (args.output_dir / "sglang.json").write_text(json.dumps(results) + "\n")
            base.print_invalid_case(f"[sglang] {work['name']}", entry)
            print_case("sglang", work["name"], entry)
    finally:
        engine.shutdown()
    base.exit_if_invalid(results["cases"].values())


def print_case(engine: str, name: str, entry: dict[str, Any]) -> None:
    print(f"[{engine}] {name}: TTFT p50 {entry['ttft_p50_ms'] or 0:.1f} ms, TPOT p50 {entry['tpot_p50_ms'] or 0:.2f} ms, "
          f"{entry['tokens_per_s']:.1f} tok/s, {entry['completed']}/{entry['expected']} complete", flush=True)


def run_session(args) -> None:
    workloads = json.loads((args.output_dir / "workloads.json").read_text())
    if args.cases:
        workloads = [w for w in workloads if w["name"] in args.cases]
    if args.engine == "hf":
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        run_hf(args, workloads)
    elif args.engine == "minillm":
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        args.sections = None
        base.run_minillm(args, workloads=workloads)
    elif args.engine == "vllm":
        if args.cases:
            raise SystemExit("--cases is not supported for vllm; it reads the whole workloads.json")
        args.sections = None
        asyncio.run(base._run_vllm(args))
    else:
        asyncio.run(_run_sglang(args, workloads))


# ----------------------------------------------------------------------------- all engines, all rounds

def gpu_compute_processes() -> list[str]:
    output = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                             "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout
    return [line for line in output.splitlines() if line.strip()]


def gpu_temperature() -> int:
    output = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True).stdout
    return int(output.strip().splitlines()[0])


def wait_until_gpu_ready(timeout_s: float = COOL_TIMEOUT_S) -> dict[str, Any]:
    """Wait for leftover engine processes to exit and the GPU to cool."""

    start = time.monotonic()
    while True:
        processes, temperature = gpu_compute_processes(), gpu_temperature()
        waited = time.monotonic() - start
        if not processes and temperature <= COOL_TEMP_C:
            return {"waited_s": round(waited, 1), "start_temp_c": temperature, "cool": True}
        if waited >= timeout_s:
            if processes:
                raise RuntimeError(f"GPU is still in use after {timeout_s:.0f} s: {processes}")
            return {"waited_s": round(waited, 1), "start_temp_c": temperature, "cool": False}
        time.sleep(5)


def git_state() -> dict[str, Any]:
    def git(*command: str) -> str:
        return subprocess.run(["git", "-C", str(ROOT / "minillm_l4"), *command], capture_output=True, text=True).stdout

    return {"sha": git("rev-parse", "HEAD").strip(), "dirty_files": git("status", "--porcelain").splitlines()}


def run_all(args) -> int:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"{args.output_dir} is not empty; choose a new dated directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base.MODEL_PATH, local_files_only=True)
    workloads = build_workloads(tokenizer, args.output_tokens)
    if args.cases:
        workloads = [w for w in workloads if w["name"] in args.cases]
    (args.output_dir / "workloads.json").write_text(json.dumps(workloads) + "\n")
    manifest: dict[str, Any] = {"created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "model": base.MODEL_ID, "revision": base.MODEL_REVISION,
                                "repetitions": args.repetitions, "warmup_repetitions": args.warmup_repetitions,
                                "output_tokens": args.output_tokens, "git": git_state(), "sessions": []}
    failed = 0
    for round_index in range(1, args.rounds + 1):
        for engine in session_order(round_index):
            if args.engines and engine not in args.engines:
                continue
            ready = wait_until_gpu_ready()
            session_dir = args.output_dir / f"round{round_index}" / engine
            session_dir.mkdir(parents=True)
            shutil.copy(args.output_dir / "workloads.json", session_dir / "workloads.json")
            python, env = engine_environment(engine)
            command = [str(python), str(THIS_FILE), "--session", engine, "--output-dir", str(session_dir),
                       "--repetitions", str(args.repetitions), "--warmup-repetitions", str(args.warmup_repetitions)]
            print(f"=== round {round_index}: {engine} (GPU {ready['start_temp_c']} C)", flush=True)
            telemetry = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu=timestamp,{','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits",
                 "-lms", "1000", "-f", str(session_dir / "gpu_telemetry.csv")])
            started = time.time()
            with (session_dir / "session.log").open("w") as log:
                code = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            telemetry.terminate()
            telemetry.wait()
            manifest["sessions"].append({"round": round_index, "engine": engine, "exit_code": code,
                                         "seconds": round(time.time() - started, 1), **ready})
            (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
            print(f"    exit {code} after {time.time() - started:.0f} s", flush=True)
            failed += code != 0
    wait_until_gpu_ready()
    return failed


# ----------------------------------------------------------------------------- report

def combine_rounds(entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pool the measured runs of one case across rounds, keeping the per-round p50s to show noise."""

    all_runs = [run for entry in entries for run in entry["runs"]]
    combined = base.summarize(all_runs)
    combined["e2e_ms"] = statistics.median(run["duration_ms"] for run in all_runs)
    combined["rounds"] = {key: [entry[key] for entry in entries] for key in ("ttft_p50_ms", "tpot_p50_ms", "tokens_per_s")}
    combined["valid"] = all(entry.get("valid", True) for entry in entries) and combined["valid"]
    return combined


def round_spread(values: Sequence[float | None]) -> float | None:
    """Largest gap between rounds as a share of their mean."""

    present = [v for v in values if v is not None]
    if len(present) < 2 or not statistics.mean(present):
        return None
    return (max(present) - min(present)) / statistics.mean(present)


def first_difference(a: Sequence[int], b: Sequence[int]) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def telemetry_summary(path: Path) -> dict[str, float | None]:
    """Busy samples only (GPU over 50% used) of one session's nvidia-smi log."""

    if not path.exists():
        return {}
    rows = []
    with path.open() as handle:
        for row in csv.reader(handle):
            if len(row) != len(GPU_FIELDS) + 1:
                continue
            try:
                rows.append({"temp": float(row[1]), "power": float(row[2]), "sm": float(row[3]), "mem_used": float(row[5]),
                             "util": float(row[6]), "power_cap": row[7].strip() == "Active"})
            except ValueError:
                continue
    busy = [r for r in rows if r["util"] > 50]
    if not busy:
        return {"samples": len(rows)}
    return {"samples": len(rows), "busy_samples": len(busy), "power_w": statistics.median(r["power"] for r in busy),
            "sm_mhz": statistics.median(r["sm"] for r in busy), "temp_max_c": max(r["temp"] for r in busy),
            "memory_peak_mib": max(r["mem_used"] for r in rows),
            "power_capped_share": sum(r["power_cap"] for r in busy) / len(busy)}


def load_sessions(output_dir: Path) -> dict[str, dict[int, dict[str, Any]]]:
    sessions: dict[str, dict[int, dict[str, Any]]] = {engine: {} for engine in ENGINES}
    for round_dir in sorted(output_dir.glob("round*")):
        round_index = int(round_dir.name.removeprefix("round"))
        for engine in ENGINES:
            path = round_dir / engine / f"{engine}.json"
            if path.exists():
                data = json.loads(path.read_text())
                data["telemetry"] = telemetry_summary(round_dir / engine / "gpu_telemetry.csv")
                sessions[engine][round_index] = data
    return sessions


def build_report(output_dir: Path) -> tuple[str, dict[str, Any]]:
    workloads = json.loads((output_dir / "workloads.json").read_text())
    sessions = load_sessions(output_dir)
    engines = [engine for engine in ENGINES if sessions[engine]]
    combined: dict[str, dict[str, dict[str, Any]]] = {}
    for engine in engines:
        combined[engine] = {}
        for work in workloads:
            entries = [data["cases"][work["name"]] for _, data in sorted(sessions[engine].items())
                       if work["name"] in data["cases"]]
            if entries:
                combined[engine][work["name"]] = combine_rounds(entries)

    def cell(engine, name, key, digits):
        entry = combined[engine].get(name)
        if entry is None or entry.get(key) is None:
            return "-"
        return f"{entry[key]:,.{digits}f}" + ("" if entry["valid"] else " (invalid)")

    output_tokens = workloads[0]["output_tokens"]
    runs = {engine: sum(len(e["runs"]) for data in sessions[engine].values() for e in list(data["cases"].values())[:1])
            for engine in engines}
    lines = ["# Hugging Face vs vLLM vs SGLang vs MiniLLM-L4", "",
             f"Every prompt asks for {output_tokens} output tokens. Lower is better for times; higher is better for tok/s.", ""]
    for batch in BATCH_SIZES:
        cases = [w for w in workloads if w["batch"] == batch]
        if not cases:
            continue
        lines += [f"## {batch} request{'s' if batch > 1 else ''} at once", ""]
        for title, key, digits in (("Time to first token, p50 (ms)", "ttft_p50_ms", 1),
                                   ("Time to first token, p95 (ms)", "ttft_p95_ms", 1),
                                   ("Time per output token, p50 (ms)", "tpot_p50_ms", 1),
                                   ("Worst pause between two tokens, median across runs (ms)", "worst_gap_ms", 1),
                                   ("Output tokens per second", "tokens_per_s", 1),
                                   ("Total time for the whole run (ms)", "e2e_ms", 0)):
            lines += [f"**{title}**", "", "| Prompt tokens | " + " | ".join(LABELS[e] for e in engines) + " |",
                      "|---|" + "---|" * len(engines)]
            lines += [f"| {w['prompt_tokens']:,} | " + " | ".join(cell(e, w["name"], key, digits) for e in engines) + " |"
                      for w in cases]
            lines.append("")

    lines += ["## How repeatable it was", "",
              "Largest gap between the rounds, as a share of the mean, over all cases. A gap between engines "
              "smaller than this is noise.", "",
              "| Engine | Rounds | Measured runs per case | TTFT | Time per token | tok/s |", "|---|---|---|---|---|---|"]
    for engine in engines:
        spreads = {key: [s for s in (round_spread(entry["rounds"][key]) for entry in combined[engine].values())
                         if s is not None] for key in ("ttft_p50_ms", "tpot_p50_ms", "tokens_per_s")}
        cells = [f"{max(spreads[key]):.1%}" if spreads[key] else "-" for key in ("ttft_p50_ms", "tpot_p50_ms", "tokens_per_s")]
        lines.append(f"| {LABELS[engine]} | {len(sessions[engine])} | {runs[engine]} | " + " | ".join(cells) + " |")

    lines += ["", "## GPU during each session", "",
              "Medians while the GPU was over 50% busy (L4 power cap: 72 W).", "",
              "| Round | Engine | Power (W) | SM clock (MHz) | Max temp (C) | Peak memory (MiB) | Time at power cap |",
              "|---|---|---|---|---|---|---|"]
    for engine in engines:
        for round_index, data in sorted(sessions[engine].items()):
            t = data["telemetry"]
            if "power_w" not in t:
                lines.append(f"| {round_index} | {LABELS[engine]} | - | - | - | - | - |")
                continue
            lines.append(f"| {round_index} | {LABELS[engine]} | {t['power_w']:.0f} | {t['sm_mhz']:.0f} | {t['temp_max_c']:.0f} | "
                         f"{t['memory_peak_mib']:,.0f} | {t['power_capped_share']:.0%} |")

    if "hf" in engines:
        lines += ["", "## Do the outputs match Hugging Face?", "",
                  "First measured run of round 1 only, using tokens the benchmark already produced. "
                  "`first difference` is the lowest output position that differs in any request.", "",
                  "| Engine | Requests identical to HF | Earliest first difference |", "|---|---|---|"]
        hf_first = sessions["hf"][min(sessions["hf"])]["cases"]
        for engine in engines:
            if engine == "hf":
                continue
            other = sessions[engine][min(sessions[engine])]["cases"]
            same = total = 0
            earliest: int | None = None
            for work in workloads:
                name = work["name"]
                if name not in hf_first or name not in other:
                    continue
                for a, b in zip(hf_first[name]["runs"][0]["requests"], other[name]["runs"][0]["requests"]):
                    total += 1
                    if a["token_ids"] == b["token_ids"]:
                        same += 1
                    else:
                        position = first_difference(a["token_ids"], b["token_ids"])
                        earliest = position if earliest is None else min(earliest, position)
            lines.append(f"| {LABELS[engine]} | {same}/{total} | {'-' if earliest is None else earliest} |")

    lines += ["", "## Settings each engine used", ""]
    for engine in engines:
        config = sessions[engine][min(sessions[engine])]["config"]
        lines.append(f"- **{LABELS[engine]}**: `{json.dumps(config)}`")
    lines.append("")
    invalid = [f"{engine} {name}: {data['cases'][name].get('invalid_reason')}" for engine in engines
               for data in sessions[engine].values() for name in data["cases"] if data["cases"][name].get("valid") is False]
    if invalid:
        lines += ["## Invalid cases", ""] + [f"- {item}" for item in invalid] + [""]
    return "\n".join(lines), {"combined": combined, "invalid": invalid}


def report(output_dir: Path) -> int:
    text, data = build_report(output_dir)
    (output_dir / "comparison.md").write_text(text)
    for engine_cases in data["combined"].values():
        for entry in engine_cases.values():
            entry.pop("runs", None)
    (output_dir / "comparison.json").write_text(json.dumps(data, indent=1) + "\n")
    print(text)
    return 1 if data["invalid"] else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session", dest="engine", choices=ENGINES, help="internal: run one engine in this process")
    parser.add_argument("--report", action="store_true", help="only rebuild comparison.md from an existing directory")
    parser.add_argument("--engines", nargs="+", choices=ENGINES, help="limit to these engines (for smoke runs)")
    parser.add_argument("--cases", nargs="+", help="limit to these case names, such as p512_b1 (for smoke runs)")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM only")
    args = parser.parse_args()
    if args.engine:
        run_session(args)
    elif args.report:
        raise SystemExit(report(args.output_dir))
    else:
        failed = run_all(args)
        code = report(args.output_dir)
        raise SystemExit(1 if failed or code else 0)


if __name__ == "__main__":
    main()
