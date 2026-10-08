"""MiniLLM-L4 vs vLLM on identical token-level workloads, one L4.

Three steps, each its own process (the engines live in different Python
environments and must not share the GPU):

  # 1. build the workloads and run MiniLLM (best configuration: mixed batching,
  #    adaptive chunks, CUDA Graph decode, fused kernels)
  .conda-env/bin/python -m minillm_l4.benchmarks.commands.run_vllm_compare \\
      --engine minillm --output-dir minillm_l4/results/vllm_compare_<date>
  # 2. run vLLM on the saved workloads (separate env with vLLM installed; the
  #    engine core is spawned, FlashInfer's sampler JIT needs nvcc+ninja, and
  #    greedy decoding never uses it, so the PyTorch sampler is selected)
  ENV=/home/ubuntu/miniconda3/envs/nemo35_vllm
  PATH=$ENV/bin:$PATH VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0 \\
  LD_LIBRARY_PATH=$ENV/lib/python3.12/site-packages/nvidia/cu13/lib \\
  $ENV/bin/python minillm_l4/benchmarks/commands/run_vllm_compare.py \\
      --engine vllm --output-dir minillm_l4/results/vllm_compare_<date>
  # 3. compare
  .conda-env/bin/python -m minillm_l4.benchmarks.commands.run_vllm_compare \\
      --report --output-dir minillm_l4/results/vllm_compare_<date>

Matched: model and revision, prompt token IDs, output lengths (EOS ignored),
greedy decoding, arrival times, at most 32 requests in flight, prefix caching
off, 1 warm-up + 3 measured runs per case.  Each engine keeps its own defaults
otherwise (vLLM: torch.compile + CUDA graphs, chunked prefill, its FP8 kernels).

Metrics use one definition for both: TTFT = first token - scheduled arrival;
TPOT = (last token - first token) / (tokens - 1); worst gap = the largest
inter-token gap of any request (median over runs); output tok/s = output
tokens / (run start -> last token).  Percentiles are interpolated.  MiniLLM
timestamps are taken when a token is read back on the host; vLLM's when the
streamed token reaches the caller (including its engine-core IPC).

Only the standard library is imported at module level so the vLLM step can run
this file directly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ("/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507-FP8/"
              "snapshots/8591804019c8b22094c3b5b4454e0edc05dffc98")
MAX_IN_FLIGHT = 32
MAX_MODEL_LEN = 8448


def percentile(values: Sequence[float], q: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Runs are {"duration_ms", "requests": [{ttft_ms, tpot_ms, max_gap_ms, tokens, token_ids}]}."""

    rows = [r for run in runs for r in run["requests"]]
    ttft = [r["ttft_ms"] for r in rows if r["ttft_ms"] is not None]
    tpot = [r["tpot_ms"] for r in rows if r["tpot_ms"] is not None]
    worst = [max((r["max_gap_ms"] for r in run["requests"] if r["max_gap_ms"] is not None), default=None)
             for run in runs]
    worst = [w for w in worst if w is not None]
    return {
        "ttft_p50_ms": percentile(ttft, 0.5), "ttft_p95_ms": percentile(ttft, 0.95),
        "tpot_p50_ms": percentile(tpot, 0.5), "tpot_p95_ms": percentile(tpot, 0.95),
        "worst_gap_ms": statistics.median(worst) if worst else None,
        "tokens_per_s": statistics.median(sum(r["tokens"] for r in run["requests"]) / (run["duration_ms"] / 1000)
                                          for run in runs),
        "completed": sum(r["tokens"] == r["expected_tokens"] for r in rows), "expected": len(rows),
    }


# ----------------------------------------------------------------------------- workloads

def build_workloads(tok) -> list[dict[str, Any]]:
    import random

    from minillm_l4.benchmarks.commands.run_showcase import prompt

    def case(section, name, label, requests):
        return {"section": section, "name": name, "label": label, "requests": [
            {"id": f"{name}-r{i}", "prompt_token_ids": list(p), "max_new_tokens": n, "arrival_ms": t}
            for i, (p, n, t) in enumerate(requests)]}

    cases = []
    for length in (128, 512, 2048, 8192):
        cases.append(case("prefill", f"prefill_{length}", f"{length:,}-token prompt",
                          [(prompt(tok, length, "prefill"), 1, 0.0)]))
    p512 = prompt(tok, 512, "decode")
    for batch in (1, 8, 32):
        cases.append(case("decode", f"decode_b{batch}", f"{batch} request{'s' if batch > 1 else ''}",
                          [(p512, 128, 0.0)] * batch))
    lengths = [128, 256, 512, 1024] * 4
    random.Random(17).shuffle(lengths)
    prompts = {n: prompt(tok, n, "serve") for n in set(lengths)}
    cases.append(case("serving", "serving_16", "16 requests, 128-1,024 tokens, every 150 ms",
                      [(prompts[n], 128, 150.0 * i) for i, n in enumerate(lengths)]))
    longs = {n: prompt(tok, n, "long") for n in (128, 6144)}
    cases.append(case("serving", "longmix", "6,144-token prompts among short ones, every 400 ms",
                      [(longs[128 if i % 2 == 0 else 6144], 64, 400.0 * i) for i in range(6)]))
    heavy = [128, 256, 512, 1024, 2048] * 13
    random.Random(29).shuffle(heavy)
    heavy_prompts = {n: prompt(tok, n, "heavy") for n in set(heavy)}
    cases.append(case("serving", "heavy_64", "64 requests, 128-2,048 tokens, every 50 ms",
                      [(heavy_prompts[n], 128, 50.0 * i) for i, n in enumerate(heavy[:64])]))
    return cases


# ----------------------------------------------------------------------------- MiniLLM

def run_minillm(args) -> None:
    import gc

    import torch

    from minillm_l4.benchmarks.core.harness import BenchmarkHarness
    from minillm_l4.benchmarks.core.schemas import HarnessConfig, RequestSpec, WorkloadSpec
    from minillm_l4.benchmarks.runners.chunked_prefill import ChunkedPrefillPagedRunner
    from minillm_l4.benchmarks.runners.huggingface_baseline import MODEL_ID, MODEL_REVISION, load_qwen_fp8
    from minillm_l4.engine.kernels.fused import install_fused_kernels

    bundle = load_qwen_fp8(fp8_kernel_path="sm89", local_files_only=True)
    model = bundle.model
    if not install_fused_kernels(model):
        raise RuntimeError("fused kernels unavailable")
    workloads = build_workloads(bundle.tokenizer)
    if args.sections:
        workloads = [w for w in workloads if w["section"] in args.sections]
    (args.output_dir / "workloads.json").write_text(json.dumps(workloads) + "\n")
    harness = BenchmarkHarness(HarnessConfig(
        repetitions=args.repetitions, warmup_repetitions=args.warmup_repetitions,
        respect_arrival_schedule=True, collect_gpu=True, collect_system_telemetry=False,
        sample_interval_seconds=0.25))
    results = {"engine": "minillm", "config": {
        "mixed_batch": True, "adaptive_chunking": True, "cuda_graphs": True, "fused_kernels": True,
        "prefix_cache": False, "max_batch_size": MAX_IN_FLIGHT, "block_size": 16}, "cases": {}}
    for work in workloads:
        requests = tuple(RequestSpec(r["id"], tuple(r["prompt_token_ids"]), r["max_new_tokens"],
                                     scheduled_arrival_ms=r["arrival_ms"]) for r in work["requests"])
        pages = sum(math.ceil((r.prompt_tokens + r.max_new_tokens) / 16) for r in requests) + len(requests)
        runner = ChunkedPrefillPagedRunner(
            model, block_size=16, num_blocks=pages, max_batch_size=MAX_IN_FLIGHT, max_prefill_tokens=256,
            device="cuda", decode_backend="auto", decode_sdpa_compat=True, enable_prefix=False,
            prefill_chunk_size=None, mixed_batch=True, adaptive_chunking=True, cuda_graphs=True)
        retries = torch.cuda.memory_stats().get("num_alloc_retries", 0)
        try:
            result = harness.run_trace(WorkloadSpec(
                name=work["name"], seed=17, requests=requests, model_id=MODEL_ID, model_revision=MODEL_REVISION,
                dtype="fp8", device="cuda:0", arrival_pattern="fixed_rate"), runner)
        finally:
            runner.close()
            del runner
            gc.collect()
            torch.cuda.empty_cache()
        runs = []
        for run in result.runs:
            rows = []
            for record in run["requests"]:
                m, outcome = record["metrics"], record["outcome"]
                rows.append({"id": record["request_id"], "ttft_ms": m["ttft_ms"], "tpot_ms": m["tpot_ms"],
                             "max_gap_ms": max(m["itl_ms"]) if m["itl_ms"] else None,
                             "tokens": len(outcome["generated_token_ids"]),
                             "expected_tokens": m["requested_output_tokens"],
                             "token_ids": outcome["generated_token_ids"]})
            runs.append({"duration_ms": run["duration_ms"], "requests": rows})
        entry = {**summarize(runs), "alloc_retries": torch.cuda.memory_stats().get("num_alloc_retries", 0) - retries,
                 "runs": runs}
        results["cases"][work["name"]] = entry
        print(f"[minillm] {work['name']}: TTFT p50 {entry['ttft_p50_ms']:.1f} ms, TPOT p50 "
              f"{entry['tpot_p50_ms'] or 0:.2f} ms, {entry['tokens_per_s']:.1f} tok/s, "
              f"{entry['completed']}/{entry['expected']} complete, retries {entry['alloc_retries']}", flush=True)
        (args.output_dir / "minillm.json").write_text(json.dumps(results) + "\n")


# ----------------------------------------------------------------------------- vLLM

async def _vllm_case(engine, work, label) -> dict[str, Any]:
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    start = time.perf_counter_ns()

    async def one(request):
        arrival = start + int(request["arrival_ms"] * 1e6)
        delay = (arrival - time.perf_counter_ns()) / 1e9
        if delay > 0:
            await asyncio.sleep(delay)
        params = SamplingParams(max_tokens=request["max_new_tokens"], min_tokens=request["max_new_tokens"],
                                temperature=0.0, ignore_eos=True, detokenize=False,
                                output_kind=RequestOutputKind.DELTA)
        tokens, times = [], []
        async for output in engine.generate(TokensPrompt(prompt_token_ids=request["prompt_token_ids"]), params,
                                            request_id=f"{label}-{request['id']}"):
            delta = list(output.outputs[0].token_ids)
            if delta:
                now = time.perf_counter_ns()
                tokens += delta
                times += [now] * len(delta)
        gaps = [(b - a) / 1e6 for a, b in zip(times, times[1:])]
        return {"id": request["id"], "ttft_ms": (times[0] - arrival) / 1e6,
                "tpot_ms": (times[-1] - times[0]) / 1e6 / (len(times) - 1) if len(times) > 1 else None,
                "max_gap_ms": max(gaps) if gaps else None, "tokens": len(tokens),
                "expected_tokens": request["max_new_tokens"], "token_ids": tokens, "last_ns": times[-1]}

    rows = await asyncio.gather(*(one(r) for r in work["requests"]))
    duration = (max(r.pop("last_ns") for r in rows) - start) / 1e6
    return {"duration_ms": duration, "requests": rows}


async def _run_vllm(args) -> None:
    import vllm
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    workloads = json.loads((args.output_dir / "workloads.json").read_text())
    if args.sections:
        workloads = [w for w in workloads if w["section"] in args.sections]
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=MODEL_PATH, max_model_len=MAX_MODEL_LEN, max_num_seqs=MAX_IN_FLIGHT,
        gpu_memory_utilization=args.gpu_memory_utilization, enable_prefix_caching=False, seed=0,
        disable_log_stats=True))
    config = engine.vllm_config
    results = {"engine": "vllm", "config": {
        "vllm_version": vllm.__version__, "max_num_seqs": MAX_IN_FLIGHT, "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": args.gpu_memory_utilization, "prefix_cache": False,
        "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
        "chunked_prefill": config.scheduler_config.enable_chunked_prefill,
        "cudagraph_mode": str(config.compilation_config.cudagraph_mode),
        "quantization": config.model_config.quantization}, "cases": {}}
    try:
        for work in workloads:
            for index in range(args.warmup_repetitions):
                await _vllm_case(engine, work, f"warm{index}")
            runs = [await _vllm_case(engine, work, f"run{index}") for index in range(args.repetitions)]
            entry = {**summarize(runs), "runs": runs}
            results["cases"][work["name"]] = entry
            print(f"[vllm] {work['name']}: TTFT p50 {entry['ttft_p50_ms']:.1f} ms, TPOT p50 "
                  f"{entry['tpot_p50_ms'] or 0:.2f} ms, {entry['tokens_per_s']:.1f} tok/s, "
                  f"{entry['completed']}/{entry['expected']} complete", flush=True)
            (args.output_dir / "vllm.json").write_text(json.dumps(results) + "\n")
    finally:
        engine.shutdown()


# ----------------------------------------------------------------------------- report

def report(args) -> None:
    mini = json.loads((args.output_dir / "minillm.json").read_text())
    vllm = json.loads((args.output_dir / "vllm.json").read_text())
    workloads = {w["name"]: w for w in json.loads((args.output_dir / "workloads.json").read_text())}

    def fmt(value, unit=""):
        return "-" if value is None else f"{value:,.1f}{unit}"

    def ratio(ours, theirs, lower_better=True):
        if not ours or not theirs:
            return "-"
        return f"{ours / theirs:.2f}x" if lower_better else f"{ours / theirs:.0%}"

    lines = ["# MiniLLM-L4 vs vLLM", "",
             f"vLLM {vllm['config']['vllm_version']} ({vllm['config']['cudagraph_mode']}, chunked prefill, "
             f"{vllm['config']['max_num_batched_tokens']} tokens per step). MiniLLM: "
             + ", ".join(k for k, v in mini["config"].items() if v is True) + ".", ""]
    sections = (("prefill", "Prefill (one request): time to first token",
                 (("TTFT", "ttft_p50_ms", " ms", True),)),
                ("decode", "Decode (512-token prompts, 128 outputs, all at once)",
                 (("Time per token", "tpot_p50_ms", " ms", True), ("Output tok/s", "tokens_per_s", "", False))),
                ("serving", "Serving (arrivals over time, 128 outputs; long mix 64)",
                 (("TTFT p50", "ttft_p50_ms", " ms", True), ("TTFT p95", "ttft_p95_ms", " ms", True),
                  ("Time per token", "tpot_p50_ms", " ms", True), ("Worst gap", "worst_gap_ms", " ms", True),
                  ("Output tok/s", "tokens_per_s", "", False))))
    agreement = []
    for section, title, columns in sections:
        names = [n for n, w in workloads.items() if w["section"] == section and n in mini["cases"] and n in vllm["cases"]]
        if not names:
            continue
        lines += [f"## {title}", "", "| Case | " + " | ".join(
            f"{c[0]} (MiniLLM / vLLM / ratio)" for c in columns) + " |", "|---|" + "---|" * len(columns)]
        for name in names:
            ours, theirs = mini["cases"][name], vllm["cases"][name]
            cells = [f"{fmt(ours[key], unit)} / {fmt(theirs[key], unit)} / {ratio(ours[key], theirs[key], low)}"
                     for _, key, unit, low in columns]
            lines.append(f"| {workloads[name]['label']} | " + " | ".join(cells) + " |")
            for a, b in zip(ours["runs"][0]["requests"], theirs["runs"][0]["requests"]):
                common = next((i for i, (x, y) in enumerate(zip(a["token_ids"], b["token_ids"])) if x != y),
                              min(len(a["token_ids"]), len(b["token_ids"])))
                agreement.append((common, len(a["token_ids"]), a["token_ids"] == b["token_ids"]))
        lines.append("")
    if agreement:
        identical = sum(same for *_, same in agreement)
        lines += ["## Output agreement", "",
                  f"{identical}/{len(agreement)} requests (first measured run of each case) produced identical "
                  "greedy tokens in both engines"
                  + ("." if identical == len(agreement) else
                     f"; on average the first {statistics.mean(c / n for c, n, _ in agreement):.0%} of each "
                     "output matches (different FP8 kernels can flip near-tied greedy choices)."), ""]
    lines += ["Ratios: time columns are MiniLLM / vLLM (above 1.00x means MiniLLM is slower); "
              "tok/s columns are MiniLLM as a share of vLLM.", ""]
    (args.output_dir / "comparison.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--engine", choices=("minillm", "vllm"))
    mode.add_argument("--report", action="store_true")
    parser.add_argument("--sections", nargs="+", choices=("prefill", "decode", "serving"))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.report:
        report(args)
    elif args.engine == "minillm":
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        if (args.output_dir / "minillm.json").exists():
            raise FileExistsError(f"{args.output_dir} already has MiniLLM results; choose a new directory")
        run_minillm(args)
    else:
        if (args.output_dir / "vllm.json").exists():
            raise FileExistsError(f"{args.output_dir} already has vLLM results")
        asyncio.run(_run_vllm(args))


if __name__ == "__main__":
    main()
