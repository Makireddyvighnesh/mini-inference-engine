"""Unfused PyTorch ops vs the fused Triton kernels, per op and end to end.

Sections (one session, pinned Qwen3-4B FP8, no HF reference check):

* ops     - each fusion at prefill (2048 tokens) and decode (8 rows) shapes:
            time of the unfused PyTorch op sequence vs the fused kernel
            (CUDA events, median of 100), estimated bytes moved, bandwidth;
* prefill - one request, 128..8192 prompt tokens, time to first token;
* decode  - identical 512-token prompts, batch 1/8/32, eager and CUDA Graph;
* serving - 16 requests (128-1024 tokens) every 150 ms: continuous + graphs,
            and mixed batching + adaptive chunks + graphs.

Every engine case runs unfused and fused on the same loaded model; outputs are
token-identical (scripts/check_policy_equivalence.py). 1 warm-up + 3 measured
runs per engine case; each runner is freed before the next.

  .conda-env/bin/python -m minillm_l4.benchmarks.commands.run_fused_kernels \\
      --output-dir minillm_l4/results/fused_kernels_<date>
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import time
from pathlib import Path

import torch

from minillm_l4.benchmarks.commands.run_showcase import MODEL_ID, MODEL_REVISION, load_qwen_fp8, metrics, paged, prompt
from minillm_l4.benchmarks.commands.run_vllm_compare import case_validity, exit_if_invalid, print_invalid_case
from minillm_l4.benchmarks.core.harness import BenchmarkHarness
from minillm_l4.benchmarks.core.schemas import HarnessConfig, RequestSpec, WorkloadSpec

SECTIONS = ("ops", "prefill", "decode", "serving")


def install_fused_kernels(model):
    from minillm_l4.engine.kernels.fused import install_fused_kernels as install
    return install(model)


def uninstall_fused_kernels(model):
    from minillm_l4.engine.kernels.fused import uninstall_fused_kernels as uninstall
    return uninstall(model)


def _time_ms(fn, iterations=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def op_benchmarks(model):
    """Per-fusion timings with real Qwen3 weights; byte counts are per element estimates."""

    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb as torch_rope
    from minillm_l4.engine.kernels.fused import apply_rotary_pos_emb, rms_norm, silu_mul

    layer = model.model.layers[10]
    norm, q_norm = layer.post_attention_layernorm, layer.self_attn.q_norm
    rotary = model.model.rotary_emb
    rows = []
    for label, tokens in (("prefill 2048 tokens", 2048), ("decode 8 rows", 8)):
        torch.manual_seed(0)
        hidden = (torch.randn(tokens, 2560, device="cuda") * 2).to(torch.bfloat16)
        residual = (torch.randn(tokens, 2560, device="cuda") * 2).to(torch.bfloat16)
        heads = (torch.randn(tokens, 32, 128, device="cuda") * 2).to(torch.bfloat16)
        gate = (torch.randn(tokens, 9728, device="cuda") * 3).to(torch.bfloat16)
        up = (torch.randn(tokens, 9728, device="cuda") * 3).to(torch.bfloat16)
        q = heads.transpose(0, 1).unsqueeze(0)
        k = heads[:, :8].transpose(0, 1).unsqueeze(0)
        cos, sin = rotary(q, torch.arange(tokens, device="cuda")[None])
        # (name, elements, unfused bytes/elem, fused bytes/elem, unfused fn, fused fn)
        cases = (
            ("RMSNorm (hidden)", hidden.numel(), 36, 4,
             lambda: norm(hidden), lambda: rms_norm(hidden, norm.weight, norm.variance_epsilon)),
            ("residual add + RMSNorm", hidden.numel(), 42, 8,
             lambda: norm(residual + hidden),
             lambda: rms_norm(hidden, norm.weight, norm.variance_epsilon, residual=residual)),
            ("RMSNorm (q heads)", heads.numel(), 36, 4,
             lambda: q_norm(heads), lambda: rms_norm(heads, q_norm.weight, q_norm.variance_epsilon)),
            ("SiLU x up", gate.numel(), 10, 6,
             lambda: torch.nn.functional.silu(gate) * up, lambda: silu_mul(gate, up)),
            ("RoPE (q + k)", q.numel() + k.numel(), 20, 4,
             lambda: torch_rope(q, k, cos, sin), lambda: apply_rotary_pos_emb(q, k, cos, sin)),
        )
        with torch.inference_mode():
            for name, elements, unfused_b, fused_b, unfused, fused in cases:
                t_unfused, t_fused = _time_ms(unfused), _time_ms(fused)
                rows.append({
                    "shape": label, "op": name, "unfused_ms": t_unfused, "fused_ms": t_fused,
                    "speedup": t_unfused / t_fused,
                    "unfused_mb": elements * unfused_b / 1e6, "fused_mb": elements * fused_b / 1e6,
                    "fused_gb_per_s": elements * fused_b / (t_fused / 1000) / 1e9,
                })
                print(f"[ops] {label} {name}: {t_unfused:.3f} -> {t_fused:.3f} ms "
                      f"({rows[-1]['speedup']:.1f}x), ~{rows[-1]['unfused_mb']:.0f} -> {rows[-1]['fused_mb']:.0f} MB", flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sections", nargs="+", choices=SECTIONS, default=list(SECTIONS))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"{args.output_dir} is not empty; choose a new dated directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_qwen_fp8(fp8_kernel_path="sm89", local_files_only=True)
    model, tok = bundle.model, bundle.tokenizer
    harness = BenchmarkHarness(HarnessConfig(
        repetitions=args.repetitions, warmup_repetitions=args.warmup_repetitions,
        respect_arrival_schedule=True, collect_gpu=True, collect_system_telemetry=False,
        sample_interval_seconds=0.25))
    summary = {"created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": MODEL_ID,
               "revision": MODEL_REVISION, "gpu": torch.cuda.get_device_name(0),
               "repetitions": args.repetitions, "warmup_repetitions": args.warmup_repetitions,
               "hf_reference_check": False, "sections": {}}

    def save():
        (args.output_dir / "fused_kernels.json").write_text(json.dumps(summary, indent=1) + "\n")

    def run(section, name, requests, *, fused, batch_size=None, extra=None, **options):
        if fused:
            install_fused_kernels(model)
        else:
            uninstall_fused_kernels(model)
        workload = WorkloadSpec(name=f"{section}_{name}", seed=17, requests=tuple(requests), model_id=MODEL_ID,
                                model_revision=MODEL_REVISION, dtype="fp8", device="cuda:0",
                                arrival_pattern="closed_loop" if batch_size else "fixed_rate",
                                metadata={"section": section, "fused": fused})
        runner = paged(model, requests, **options)
        retries_before = torch.cuda.memory_stats().get("num_alloc_retries", 0)
        try:
            result = harness.run_batched(workload, batch_size, runner) if batch_size else harness.run_trace(workload, runner)
        finally:
            runner.close()
            del runner
            gc.collect()
            torch.cuda.empty_cache()
        entry = {"case": name, "fused": fused, **(extra or {}),
                 **metrics(result, requests, expected_runs=args.repetitions),
                 "alloc_retries": torch.cuda.memory_stats().get("num_alloc_retries", 0) - retries_before}
        entry.update(case_validity(entry))
        summary["sections"].setdefault(section, []).append(entry)
        raw = result.to_dict()
        raw["case_metrics"] = entry
        (args.output_dir / f"{section}_{name}_{'fused' if fused else 'unfused'}.json").write_text(
            json.dumps(raw, default=str) + "\n")
        save()
        print_invalid_case(f"[{section}] {name} {'fused' if fused else 'unfused'}", entry)
        print(f"[{section}] {name} {'fused' if fused else 'unfused'}: TTFT p50 {entry['ttft_p50_ms'] or 0:,.1f} ms, "
              f"TPOT p50 {entry['tpot_p50_ms'] or 0:.2f} ms, {entry['tokens_per_s']:.1f} tok/s, "
              f"retries {entry['alloc_retries']}", flush=True)

    if "ops" in args.sections:
        uninstall_fused_kernels(model)
        install_fused_kernels(model)  # compiles all kernel variants
        uninstall_fused_kernels(model)
        summary["sections"]["ops"] = op_benchmarks(model)
        save()

    if "prefill" in args.sections:
        for length in (128, 256, 512, 1024, 2048, 4096, 8192):
            reqs = [RequestSpec(f"prefill-{length}", prompt(tok, length, "prefill"), 1)]
            for fused in (False, True):
                run("prefill", f"p{length}", reqs, fused=fused, prefill_chunk_size=None,
                    extra={"prompt_tokens": length})

    if "decode" in args.sections:
        p512 = prompt(tok, 512, "decode")
        for batch in (1, 8, 32):
            reqs = [RequestSpec(f"decode-b{batch}-r{i}", p512, 128) for i in range(batch)]
            for graphs in (False, True):
                for fused in (False, True):
                    run("decode", f"b{batch}_{'graph' if graphs else 'eager'}", reqs, fused=fused,
                        prefill_chunk_size=None, cuda_graphs=graphs,
                        extra={"batch": batch, "graphs": graphs})

    if "serving" in args.sections:
        lengths = [128, 256, 512, 1024] * 4
        random.Random(17).shuffle(lengths)
        prompts = {n: prompt(tok, n, "serve") for n in set(lengths)}
        reqs = [RequestSpec(f"serve-r{i}", prompts[n], 128, scheduled_arrival_ms=150.0 * i) for i, n in enumerate(lengths)]
        for policy, options in (("continuous_graph", dict(prefill_chunk_size=None, cuda_graphs=True)),
                                ("mixed_adaptive_graph", dict(prefill_chunk_size=None, mixed_batch=True,
                                                              adaptive_chunking=True, cuda_graphs=True))):
            for fused in (False, True):
                run("serving", policy, reqs, fused=fused, extra={"policy": policy}, **options)

    uninstall_fused_kernels(model)
    print(f"Summary: {args.output_dir / 'fused_kernels.json'}")
    exit_if_invalid(entry for section, entries in summary["sections"].items() if section != "ops" for entry in entries)


if __name__ == "__main__":
    main()
