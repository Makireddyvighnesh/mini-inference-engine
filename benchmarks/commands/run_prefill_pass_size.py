"""How big should one prefill pass be on the L4?

Runs exactly one prefill forward of N tokens (N = 128 .. 65536) and measures
its time, its throughput (tokens/s), and the extra GPU memory it needs. Each N
is run two ways (whole-prompt packed prefill, so everything waiting goes into
one forward; fused kernels and merged projections on; no HF reference check):

* one prompt of N tokens       - attention cost grows with N squared;
* N/512 prompts of 512 tokens   - same pass size, cheap attention: isolates
                                  the effect of the pass size itself.

Time = first-token time of the requests (they all finish in the same forward);
1 warm-up + 3 measured runs. Memory = peak allocated during the run minus
allocated before it (weights and KV pool excluded). A size that runs out of
memory is recorded as such.

  .conda-env/bin/python -m minillm_l4.benchmarks.commands.run_prefill_pass_size \\
      --output-dir minillm_l4/results/prefill_pass_size_<date>
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch

from minillm_l4.benchmarks.commands.run_showcase import MODEL_ID, MODEL_REVISION, load_qwen_fp8, metrics, prompt
from minillm_l4.benchmarks.core.harness import BenchmarkHarness
from minillm_l4.benchmarks.core.schemas import HarnessConfig, RequestSpec, WorkloadSpec

SIZES = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
PACKED_PROMPT = 512
BLOCK = 16


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory is not empty: {args.output_dir}")
    return args


@torch.inference_mode()
def main(argv=None) -> int:
    args = parse_args(argv)
    from minillm_l4.benchmarks.runners.chunked_prefill import ChunkedPrefillPagedRunner
    from minillm_l4.engine.kernels.fused import install_fused_kernels
    from minillm_l4.engine.kernels.merged_projections import install_merged_projections

    bundle = load_qwen_fp8(fp8_kernel_path="sm89", local_files_only=True)
    model, tok = bundle.model, bundle.tokenizer
    if not install_fused_kernels(model):
        raise RuntimeError("fused kernels unavailable")
    install_merged_projections(model)
    harness = BenchmarkHarness(HarnessConfig(
        repetitions=args.repetitions, warmup_repetitions=1, respect_arrival_schedule=True,
        collect_gpu=False, collect_system_telemetry=False))
    free, total = torch.cuda.mem_get_info()
    result = {"created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": MODEL_ID,
              "revision": MODEL_REVISION, "gpu": torch.cuda.get_device_name(0), "hf_reference_check": False,
              "free_bytes_after_load": free, "total_bytes": total, "packed_prompt_tokens": PACKED_PROMPT,
              "cases": []}
    packed_prompts = [prompt(tok, PACKED_PROMPT, f"pass-r{i}") for i in range(max(args.sizes) // PACKED_PROMPT)]

    for size in args.sizes:
        for mode in ("one_prompt", "packed_512"):
            if mode == "packed_512" and size < PACKED_PROMPT:
                continue
            if mode == "one_prompt":
                reqs = (RequestSpec(f"one-{size}", prompt(tok, size, "pass-one"), 1, scheduled_arrival_ms=0.0),)
            else:
                reqs = tuple(RequestSpec(f"packed-{size}-{i}", packed_prompts[i], 1, scheduled_arrival_ms=0.0)
                             for i in range(size // PACKED_PROMPT))
            blocks = sum(math.ceil((r.prompt_tokens + 1) / BLOCK) for r in reqs) + len(reqs)
            case = {"tokens": size, "mode": mode, "requests": len(reqs)}
            runner = None
            try:
                runner = ChunkedPrefillPagedRunner(
                    model, block_size=BLOCK, num_blocks=blocks, max_batch_size=len(reqs), device="cuda",
                    decode_backend="auto", decode_sdpa_compat=True, enable_prefix=False, prefill_chunk_size=None,
                    max_prefill_tokens=size)
                gc.collect()
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                run = harness.run_trace(WorkloadSpec(
                    name=f"pass_{mode}_{size}", seed=17, requests=reqs, model_id=MODEL_ID,
                    model_revision=MODEL_REVISION, dtype="fp8", device="cuda:0", arrival_pattern="fixed_rate"), runner)
                peak = torch.cuda.max_memory_allocated()
                forwards = runner.last_summary.get("prefill_forwards") or len(runner.last_summary.get("prefill_records", []))
                rows = [r["metrics"] for rr in run.runs for r in rr["requests"]]
                # All requests of a run get their first token in the same forward: take the max per run.
                per_run = [max(r["metrics"]["ttft_ms"] for r in rr["requests"]) for rr in run.runs]
                pass_ms = sorted(per_run)[len(per_run) // 2]
                case.update({"pass_ms": pass_ms, "tokens_per_s": size / (pass_ms / 1000),
                             "extra_memory_bytes": peak - before, "prefill_forwards_last_run": forwards,
                             "failed_requests": sum(1 for r in rows if r.get("status") != "completed"),
                             "status": "ok"})
            except torch.OutOfMemoryError:
                case.update({"status": "oom"})
            finally:
                if runner is not None:
                    runner.close()
                del runner
                gc.collect()
                torch.cuda.empty_cache()
            result["cases"].append(case)
            if case["status"] == "ok":
                print(f"[pass] {size:6d} tokens {mode:10s}: {case['pass_ms']:9.1f} ms  {case['tokens_per_s']:7.0f} tok/s  "
                      f"extra memory {case['extra_memory_bytes'] / 2**30:5.2f} GiB  forwards {forwards}", flush=True)
            else:
                print(f"[pass] {size:6d} tokens {mode:10s}: out of memory", flush=True)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "prefill_pass_size.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {args.output_dir / 'prefill_pass_size.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
