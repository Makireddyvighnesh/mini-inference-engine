"""Split-KV (FlashDecoding) vs standard decode attention, kernel and end to end.

Standard decode (the engine default, ``decode_sdpa_compat=True``): one program
per (request, query head) walks the whole KV cache in 128-token tiles, which
reproduces SDPA's numerics bit for bit. Split-KV (``decode_sdpa_compat=False``):
the cache is cut into 2/4/8 pieces handled by separate programs at the same
time, then a reduce kernel merges the partial softmax results. More programs
fill the L4's 58 SMs when the batch is small; the different summation order
means outputs are not bitwise equal to SDPA.

1. kernel - one attention layer (Qwen3-4B shape: 32 query heads, 8 KV heads,
   head dim 128, 16-token pages, bf16) for batch 1-32 x context 128-2048.
   Methods: standard (SDPA-compatible, 128-token tiles), split 1/2/4/8 with
   64-token tiles, and ``auto`` (the older ``select_decode_split_count``). Each method is captured in a CUDA graph (20 calls
   per graph) and the graphs replay in turn each round (median of 25 rounds),
   so clock drift hits all methods equally. Error = max |split - standard|.
2. engine - real decode TPOT with CUDA Graphs (whole-prompt prefill, fused
   kernels, merged projections), standard vs ``decode_split_kv=True`` (split
   only where ``select_split_kv_decode`` says it wins, 64-token tiles), for
   batch 1-32 x prompt 128-2048 tokens and 128 new tokens; cases where it keeps
   the standard kernel are recorded, not timed. One runner per case and mode,
   warm-up then measured runs, mode order alternating per case. Also counts
   requests whose greedy tokens differ between the two modes.

  .conda-env/bin/python -m minillm_l4.benchmarks.commands.run_split_kv_decode \\
      --output-dir minillm_l4/results/split_kv_decode_<date>
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path

import torch

HEADS, KV_HEADS, DIM, PAGE = 32, 8, 128, 16
SCALE = DIM ** -0.5
SPLIT_TILE = 64
BATCHES = (1, 2, 4, 8, 16, 32)
CONTEXTS = (128, 256, 512, 1024, 2048)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=list(BATCHES))
    parser.add_argument("--contexts", nargs="+", type=int, default=list(CONTEXTS))
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--sections", nargs="+", choices=("kernel", "engine"), default=["kernel", "engine"])
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory is not empty: {args.output_dir}")
    return args


def graph_of(fn, calls=20):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            out = fn()
    return graph, out


def interleaved_graphs(graphs: dict, calls=20, rounds=25) -> dict:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for g in graphs.values():
        g.replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in graphs}
    for _ in range(rounds):
        for name, g in graphs.items():
            start.record()
            g.replay()
            end.record()
            torch.cuda.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / calls)  # microseconds per call
    return {name: statistics.median(xs) for name, xs in samples.items()}


def kernel_case(batch, context):
    from minillm_l4.engine.kv_cache.triton_paged_attention import (
        select_decode_split_count, triton_paged_decode_attention)

    dev, dt = "cuda", torch.bfloat16
    pages_per = math.ceil(context / PAGE)
    kb = torch.randn(batch * pages_per, KV_HEADS, PAGE, DIM, device=dev, dtype=dt)
    vb = torch.randn_like(kb)
    perm = torch.randperm(batch * pages_per, device=dev).to(torch.int32)  # scattered pages, like a real pool
    tables = perm.view(batch, pages_per).contiguous()
    lengths = torch.full((batch,), context, device=dev, dtype=torch.int32)
    q = torch.randn(batch, HEADS, 1, DIM, device=dev, dtype=dt)
    auto = select_decode_split_count(context, batch_size=batch)

    def make(split):
        common = dict(scale=SCALE, use_gqa_reuse=False, _validate_metadata=False)
        if split == 0:
            return lambda: triton_paged_decode_attention(q, kb, vb, tables, lengths, sdpa_compat=True,
                                                         block_tokens=128, **common)
        # Non-compat kernels use 64-token tiles (16-token tiles are 2-3x slower on the L4).
        return lambda: triton_paged_decode_attention(q, kb, vb, tables, lengths, split_count=split,
                                                     max_sequence_length=context, block_tokens=SPLIT_TILE, **common)

    methods = {"standard": 0, "split1": 1, "split2": 2, "split4": 4, "split8": 8, "auto": auto}
    graphs, outs = {}, {}
    for name, split in methods.items():
        graphs[name], outs[name] = graph_of(make(split))
    times = interleaved_graphs(graphs)
    ref = outs["standard"].float()
    kv_bytes = 2 * batch * context * KV_HEADS * DIM * 2
    rows = []
    for name, split in methods.items():
        rows.append({"batch": batch, "context": context, "method": name, "split_count": max(split, 1),
                     "us": times[name], "speedup_vs_standard": times["standard"] / times[name],
                     "kv_gbps": kv_bytes / (times[name] * 1e-6) / 1e9,
                     "max_abs_diff_vs_standard": (outs[name].float() - ref).abs().max().item()})
    del graphs, outs, kb, vb
    return rows


def engine_runs(model, tok, batch, context, new_tokens, compat, repetitions):
    from minillm_l4.benchmarks.commands.run_showcase import prompt
    from minillm_l4.benchmarks.core.harness import RequestEventRecorder
    from minillm_l4.benchmarks.core.schemas import RequestSpec

    requests = [RequestSpec(f"b{batch}-c{context}-r{i}", prompt(tok, context, f"splitkv-r{i}"), new_tokens)
                for i in range(batch)]
    from minillm_l4.benchmarks.commands.run_showcase import pages
    from minillm_l4.engine.serving.chunked_prefill import ChunkedPrefillPagedRunner

    runner = ChunkedPrefillPagedRunner(
        model, block_size=16, num_blocks=pages(requests), max_batch_size=len(requests), device="cuda",
        decode_backend="auto", decode_split_kv=not compat, enable_prefix=False, prefill_chunk_size=None,
        cuda_graphs=True, packed_prefill_token_limit=16384, max_prefill_tokens=16384)
    tpots, tokens = [], None
    try:
        for rep in range(repetitions + 1):
            start = time.perf_counter_ns()
            recorders = tuple(RequestEventRecorder(r.request_id, run_started_ns=start) for r in requests)
            outcomes = runner(requests, recorders)
            if any(o.status != "completed" for o in outcomes):
                raise RuntimeError(f"incomplete: {[o.status for o in outcomes]}")
            if rep == 0:
                tokens = [o.generated_token_ids for o in outcomes]
                summary = runner.last_summary
                continue
            # All requests decode together: per-step time = gap between consecutive token events of row 0,
            # after the last request's first token (all rows decoding).
            times = [sorted(e.timestamp_ns / 1e6 for e in rec.events if e.event == "token_ready") for rec in recorders]
            first_all = max(t[0] for t in times)
            steps = [b - a for a, b in zip(times[0], times[0][1:]) if a >= first_all]
            tpots.append(statistics.median(steps))
    finally:
        runner.close()
        del runner
        gc.collect()
        torch.cuda.empty_cache()
    return statistics.median(tpots), tokens, summary.get("decode_mode_used"), summary.get("graph_replays")


@torch.inference_mode()
def main(argv=None) -> int:
    args = parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    result = {"created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "hf_reference_check": False,
              "new_tokens": args.new_tokens, "kernel": [], "engine": []}

    def save():
        (out / "split_kv_decode.json").write_text(json.dumps(result, indent=1) + "\n")

    torch.manual_seed(0)
    if "kernel" in args.sections:
        for batch in args.batches:
            for context in args.contexts:
                rows = kernel_case(batch, context)
                result["kernel"].extend(rows)
                save()
                std = rows[0]["us"]
                print(f"[kernel] b={batch:2d} ctx={context:5d} standard {std:7.1f} us  "
                      + "  ".join(f"{r['method']} {r['us']:7.1f} ({r['speedup_vs_standard']:.2f}x)" for r in rows[1:]),
                      flush=True)
                gc.collect()
                torch.cuda.empty_cache()

    if "engine" in args.sections:
        from minillm_l4.benchmarks.commands.run_showcase import load_qwen_fp8
        from minillm_l4.engine.kernels.fused import install_fused_kernels
        from minillm_l4.engine.kernels.merged_projections import install_merged_projections
        from minillm_l4.engine.kv_cache.triton_paged_attention import select_split_kv_decode

        bundle = load_qwen_fp8(fp8_kernel_path="sm89", local_files_only=True)
        model, tok = bundle.model, bundle.tokenizer
        if not install_fused_kernels(model):
            raise RuntimeError("fused kernels unavailable")
        install_merged_projections(model)
        index = 0
        for batch in args.batches:
            for context in args.contexts:
                split_count = select_split_kv_decode(context + args.new_tokens - 1, _bucket(batch))
                if split_count == 1:
                    # decode_split_kv keeps the standard kernel here: identical code, nothing to time.
                    result["engine"].append({"batch": batch, "prompt_tokens": context, "split_count": 1,
                                             "same_kernel_as_standard": True})
                    continue
                order = (True, False) if index % 2 == 0 else (False, True)
                index += 1
                res = {}
                for compat in order:
                    res[compat] = engine_runs(model, tok, batch, context, args.new_tokens, compat, args.repetitions)
                std, split = res[True], res[False]
                differ = sum(a != b for a, b in zip(std[1], split[1]))
                row = {"batch": batch, "prompt_tokens": context,
                       "split_count": split_count,
                       "standard_tpot_ms": std[0], "split_tpot_ms": split[0],
                       "speedup": std[0] / split[0], "requests_with_different_tokens": differ,
                       "decode_mode": [std[2], split[2]], "graph_replays": [std[3], split[3]]}
                result["engine"].append(row)
                save()
                print(f"[engine] b={batch:2d} prompt={context:5d} split={row['split_count']}  standard "
                      f"{std[0]:6.2f} ms  split-KV {split[0]:6.2f} ms  ({row['speedup']:.3f}x)  "
                      f"tokens differ {differ}/{batch}  modes {row['decode_mode']}", flush=True)
        result["num_alloc_retries"] = torch.cuda.memory_stats().get("num_alloc_retries", 0)
    save()
    print(f"wrote {out / 'split_kv_decode.json'}")
    return 0


def _bucket(batch):
    from minillm_l4.engine.serving.decode_graph import DEFAULT_GRAPH_BATCH_SIZES, select_graph_bucket

    return select_graph_bucket(batch, DEFAULT_GRAPH_BATCH_SIZES) or batch


if __name__ == "__main__":
    raise SystemExit(main())
