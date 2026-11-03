"""Kernel track: our kernels vs SGLang's on one L4, judged the way the Open Kernel Challenge is.

Each problem runs one computation at several loads on the same inputs:

  * Both kernels live in one process, so this file runs in the SGLang env:

      PATH=$PWD/.sglang-env/lib/python3.12/site-packages/nvidia/cu13/bin:$PWD/.sglang-env/bin:$PATH \\
      PYTHONPATH=$PWD/.kernel-judge-triton371 \\
      .sglang-env/bin/python minillm_l4/benchmarks/commands/run_kernel_track.py \\
          --output-dir minillm_l4/results/kernel_track_<date>

    The SGLang env ships Triton 3.8, but our engine runs on 3.7.1 and the version matters:
    the same kernel of ours took 2,136 us on 3.8 and 1,735 us on 3.7 (gate_up, 2,048 rows,
    separate runs), because 3.8 slows `tl.range`. So the judge runs Triton 3.7.1:
      uv pip install --python .sglang-env/bin/python --target .kernel-judge-triton371 --no-deps triton==3.7.1
    main() refuses any other version unless --any-triton is passed.

  * Timing: each kernel is one CUDA graph. They replay in turn on the same inputs
    (the order flips each round), with L2 flushed between replays because a real
    step reads far more than L2 holds. A kernel's time is its median replay, and a
    load's speedup is the median of the per-round ratios, so clock drift under the
    72 W cap cancels.
  * Score: geometric mean of the speedups over a problem's loads. Above 1.00x beats SGLang.
  * Correctness: the output against a dense double-precision reference. A kernel passes
    within max(FLOOR, 1.25x SGLang's own error). `bitwise` says whether the kernel also
    matches our exact engine path; only such kernels may replace it, the rest are opt-in.

Problems (Qwen3-4B FP8, one decoder layer's real weights):

  proj              FP8 projections, activation quantization included: qkv, o, gate_up, down
  attention_decode  one step of attention over a BF16 KV cache (kernel plus its cache layout)
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ("/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507-FP8/snapshots/"
              "8591804019c8b22094c3b5b4454e0edc05dffc98/model.safetensors")
ENGINE_TRITON = "3.7.1"
BASELINE = "sglang"
EXACT = "ours"  # the kernel the engine runs today; `bitwise` compares against it
LAYER = 17  # a middle layer; every layer has the same shapes
HEADS, KV_HEADS, HEAD_DIM, OUR_PAGE = 32, 8, 128, 16
BLOCK_SIZE = [128, 128]
FLUSH_BYTES = 512 * 1024 * 1024  # several times the L4's L2
PROJ_ROWS = (1, 8, 32, 512, 2048)
ATTENTION_LOADS = tuple((batch, context) for batch in (1, 8, 32) for context in (512, 1024, 2048))
ERROR_FLOOR = {"proj": 0.01, "attention_decode": 0.015}
BASELINE_ERROR_MARGIN = 1.25


# ----------------------------------------------------------------------------- scoring (no GPU)

def geomean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("geomean needs at least one value")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def paired_speedup(baseline_us: Sequence[float], kernel_us: Sequence[float]) -> float:
    """Median of per-round ratios: both kernels in a round share the same clock and temperature."""

    if len(baseline_us) != len(kernel_us) or not baseline_us:
        raise ValueError("speedup needs the same non-zero number of rounds for both kernels")
    return statistics.median(b / k for b, k in zip(baseline_us, kernel_us))


def passes(error: float, baseline_error: float, floor: float) -> bool:
    return error <= max(floor, BASELINE_ERROR_MARGIN * baseline_error)


def round_order(names: Sequence[str], round_index: int) -> list[str]:
    """Flip the order each round so no kernel always follows the same neighbour."""

    return list(names) if round_index % 2 == 0 else list(reversed(names))


def summarize(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per kernel in each problem: geometric-mean speedup and the worst correctness result."""

    summary: dict[str, dict[str, Any]] = {}
    for row in results:
        problem = summary.setdefault(row["problem"], {})
        for name, kernel in row["kernels"].items():
            entry = problem.setdefault(name, {"speedups": [], "passed": True, "bitwise": True, "max_error": 0.0})
            entry["speedups"].append(kernel["speedup"])
            entry["passed"] = entry["passed"] and kernel["passed"]
            entry["bitwise"] = entry["bitwise"] and bool(kernel.get("bitwise"))
            entry["max_error"] = max(entry["max_error"], kernel["error"])
    for problem in summary.values():
        for entry in problem.values():
            entry["score"] = geomean(entry["speedups"])
            entry["worst_load"], entry["best_load"] = min(entry["speedups"]), max(entry["speedups"])
    return summary


def write_report(results: list[dict[str, Any]], environment: dict[str, Any]) -> str:
    summary = summarize(results)
    lines = ["# Kernel track", "", "Score: geometric-mean speedup over SGLang's kernel across a problem's loads. "
             "Above 1.00x beats SGLang.", "", "## Scores", "",
             "| Problem | Kernel | Score | Worst load | Best load | Correct | Bitwise with engine | Max error |",
             "|---|---|---:|---:|---:|---|---|---:|"]
    for problem, kernels in summary.items():
        for name, entry in kernels.items():
            lines.append(f"| {problem} | {name} | {entry['score']:.2f}x | {entry['worst_load']:.2f}x | "
                         f"{entry['best_load']:.2f}x | {'yes' if entry['passed'] else '**NO**'} | "
                         f"{'yes' if entry['bitwise'] else 'no'} | {entry['max_error']:.2%} |")
    lines += ["", "## Every load", "",
              "| Problem | Load | Kernel | Median replay (us) | Speedup | Error |", "|---|---|---|---:|---:|---:|"]
    for row in results:
        for name, kernel in row["kernels"].items():
            lines.append(f"| {row['problem']} | {row['load']} | {name} | {kernel['median_us']:,.1f} | "
                         f"{kernel['speedup']:.2f}x | {kernel['error']:.2%} |")
    lines += ["", "## Environment", "", f"`{json.dumps(environment)}`", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- the judge

def relative_error(output, reference) -> float:
    """Largest per-row relative L2 error, so one bad row cannot hide in an average."""

    import torch

    out = output.detach().double().reshape(reference.shape)
    denominator = torch.linalg.vector_norm(reference, dim=-1).clamp_min(1e-12)
    return (torch.linalg.vector_norm(out - reference, dim=-1) / denominator).max().item()


def judge_load(problem: str, load: str, kernels: dict[str, Callable[[], Any]], reference, rounds: int) -> dict[str, Any]:
    import torch

    flush_a = torch.zeros(FLUSH_BYTES, dtype=torch.uint8, device="cuda")
    flush_b = torch.ones(FLUSH_BYTES, dtype=torch.uint8, device="cuda")

    def flush_l2() -> None:
        flush_a.copy_(flush_b)
        torch.cuda.synchronize()

    graphs, outputs = {}, {}
    with torch.inference_mode():
        for name, fn in kernels.items():
            fn()  # warm up: Triton compiles and tunes outside the capture
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs[name] = fn()
            graphs[name] = graph
            graph.replay()
        torch.cuda.synchronize()

        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        samples: dict[str, list[float]] = {name: [] for name in kernels}
        for index in range(rounds):
            for name in round_order(list(kernels), index):
                flush_l2()
                start.record()
                graphs[name].replay()
                end.record()
                torch.cuda.synchronize()
                samples[name].append(start.elapsed_time(end) * 1000.0)

        # Read outputs after timing: replays above overwrote them with the same inputs.
        errors = {name: relative_error(outputs[name], reference) for name in kernels}
        exact = outputs[EXACT].detach().reshape(reference.shape) if EXACT in outputs else None

    baseline_error = errors[BASELINE]
    row: dict[str, Any] = {"problem": problem, "load": load, "kernels": {}}
    for name in kernels:
        output = outputs[name].detach().reshape(reference.shape)
        row["kernels"][name] = {
            "median_us": statistics.median(samples[name]), "samples_us": samples[name],
            "speedup": paired_speedup(samples[BASELINE], samples[name]), "error": errors[name],
            "passed": passes(errors[name], baseline_error, ERROR_FLOOR[problem]),
            "bitwise": exact is not None and bool(torch.equal(output, exact))}
    del graphs, outputs, flush_a, flush_b
    torch.cuda.empty_cache()
    return row


# ----------------------------------------------------------------------------- problems

def load_layer_weights(layer: int) -> dict[str, tuple[Any, Any]]:
    """Real weights and block scales of one decoder layer, with q/k/v and gate/up merged as both engines run them."""

    import torch
    from safetensors import safe_open

    prefix = f"model.layers.{layer}."
    parts = {"qkv": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), "o": ("self_attn.o_proj",),
             "gate_up": ("mlp.gate_proj", "mlp.up_proj"), "down": ("mlp.down_proj",)}
    weights = {}
    with safe_open(MODEL_PATH, framework="pt", device="cuda") as checkpoint:
        for name, modules in parts.items():
            weights[name] = (torch.cat([checkpoint.get_tensor(f"{prefix}{m}.weight") for m in modules], dim=0),
                             # The file stores FP16 scales; both engines widen them to FP32 at load.
                             torch.cat([checkpoint.get_tensor(f"{prefix}{m}.weight_scale_inv") for m in modules], dim=0).float())
    return weights


def dense_fp8_reference(x, weight, scales):
    """x @ (W * block scale).T in double precision, with no activation quantization."""

    import torch

    expanded = scales.double().repeat_interleave(BLOCK_SIZE[0], dim=0).repeat_interleave(BLOCK_SIZE[1], dim=1)
    dequantized = weight.double() * expanded[:weight.shape[0], :weight.shape[1]]
    return x.double() @ dequantized.T


def proj_problem(rounds: int, seed: int) -> list[dict[str, Any]]:
    import torch
    from minillm_l4.engine.kernels.sm89_fp8 import pretune_sm89_fp8_shapes, sm89_fp8_linear
    from sglang.srt.layers.quantization.fp8_utils import triton_w8a8_block_fp8_linear

    weights = load_layer_weights(LAYER)
    pretune_sm89_fp8_shapes([(weight, scales, BLOCK_SIZE) for weight, scales in weights.values()], torch.bfloat16,
                            max_tokens=max(PROJ_ROWS))
    results = []
    for name, (weight, scales) in weights.items():
        for rows in PROJ_ROWS:
            generator = torch.Generator(device="cuda").manual_seed(seed + rows)
            x = torch.randn(rows, weight.shape[1], device="cuda", dtype=torch.float32, generator=generator).to(torch.bfloat16)
            reference = dense_fp8_reference(x, weight, scales)
            kernels = {
                BASELINE: lambda: triton_w8a8_block_fp8_linear(x, weight, BLOCK_SIZE, scales),
                EXACT: lambda: sm89_fp8_linear(x, weight, scales, block_size=BLOCK_SIZE, output_dtype=torch.bfloat16),
            }
            row = judge_load("proj", f"{name} m={rows}", kernels, reference, rounds)
            results.append(row)
            print_row(row)
    del weights
    torch.cuda.empty_cache()
    return results


def attention_reference(q, k, v):
    """softmax(q @ K.T / sqrt(d)) @ V per request and query head, in double precision. q: [B, H, D]; k, v: [B, C, Hk, D]."""

    import torch

    per_kv = q.shape[1] // k.shape[2]
    keys = k.double().repeat_interleave(per_kv, dim=2)
    values = v.double().repeat_interleave(per_kv, dim=2)
    scores = torch.einsum("bhd,bchd->bch", q.double(), keys) / math.sqrt(q.shape[-1])
    return torch.einsum("bch,bchd->bhd", scores.softmax(dim=1), values).flatten(1)


def flashinfer_decode(q, k, v, *, workspace):
    """SGLang's decode kernel on its own cache layout: page size 1, FP-graph planning as in its backend."""

    import torch
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

    batch, context, kv_heads, dim = k.shape
    heads = q.shape[1]
    permutation = torch.randperm(batch * context, device="cuda").to(torch.int32)
    k_cache = torch.empty(batch * context, 1, kv_heads, dim, device="cuda", dtype=k.dtype)
    v_cache = torch.empty_like(k_cache)
    k_cache[permutation.long(), 0] = k.flatten(0, 1)
    v_cache[permutation.long(), 0] = v.flatten(0, 1)
    indptr = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * context
    last_page = torch.ones(batch, device="cuda", dtype=torch.int32)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2", use_cuda_graph=True, use_tensor_cores=True,
        paged_kv_indptr_buffer=indptr.clone(), paged_kv_indices_buffer=permutation.clone(),
        paged_kv_last_page_len_buffer=last_page.clone())
    wrapper.plan(indptr, permutation, last_page, heads, kv_heads, dim, 1, q_data_type=q.dtype,
                 kv_data_type=k.dtype, sm_scale=dim ** -0.5)
    return lambda: wrapper.forward(q, (k_cache, v_cache), sm_scale=dim ** -0.5)


def minillm_decode(q, k, v):
    """Our kernels on our cache layout: 16-token pages at scattered positions."""

    import torch
    from minillm_l4.engine.kv_cache import select_tensor_core_decode
    from minillm_l4.engine.kv_cache.triton_paged_attention import triton_paged_decode_attention

    batch, context, kv_heads, dim = k.shape
    split, tile = select_tensor_core_decode(context, batch)
    pages_per_request = math.ceil(context / OUR_PAGE)
    total_pages = batch * pages_per_request
    table = torch.randperm(total_pages, device="cuda").view(batch, pages_per_request)
    k_blocks = torch.zeros(total_pages, kv_heads, OUR_PAGE, dim, device="cuda", dtype=k.dtype)
    v_blocks = torch.zeros_like(k_blocks)
    positions = torch.arange(context, device="cuda")
    page, slot = positions // OUR_PAGE, positions % OUR_PAGE
    for request in range(batch):
        k_blocks[table[request, page], :, slot] = k[request]
        v_blocks[table[request, page], :, slot] = v[request]
    table32 = table.to(torch.int32)
    lengths = torch.full((batch,), context, device="cuda", dtype=torch.int32)
    query = q.unsqueeze(2)  # [B, H, 1, D]
    common = dict(scale=dim ** -0.5, use_gqa_reuse=False, _validate_metadata=False)
    return {
        EXACT: lambda: triton_paged_decode_attention(query, k_blocks, v_blocks, table32, lengths, sdpa_compat=True,
                                                     block_tokens=128, **common),
        "ours_tensor_core": lambda: triton_paged_decode_attention(
            query, k_blocks, v_blocks, table32, lengths, split_count=split, max_sequence_length=context,
            block_tokens=tile, **{**common, "use_gqa_reuse": True}),
    }


def attention_problem(rounds: int, seed: int) -> list[dict[str, Any]]:
    import torch

    workspace = torch.zeros(1024 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    results = []
    for batch, context in ATTENTION_LOADS:
        generator = torch.Generator(device="cuda").manual_seed(seed + batch * 10000 + context)
        shape = (batch, context, KV_HEADS, HEAD_DIM)
        k = torch.randn(shape, device="cuda", generator=generator).to(torch.bfloat16)
        v = torch.randn(shape, device="cuda", generator=generator).to(torch.bfloat16)
        q = torch.randn(batch, HEADS, HEAD_DIM, device="cuda", generator=generator).to(torch.bfloat16)
        reference = attention_reference(q, k, v)
        kernels = {BASELINE: flashinfer_decode(q, k, v, workspace=workspace), **minillm_decode(q, k, v)}
        row = judge_load("attention_decode", f"batch {batch} x context {context:,}", kernels, reference, rounds)
        results.append(row)
        print_row(row)
        del kernels, k, v, q, reference
        torch.cuda.empty_cache()
    del workspace
    torch.cuda.empty_cache()
    return results


PROBLEMS = {"proj": proj_problem, "attention_decode": attention_problem}


def print_row(row: dict[str, Any]) -> None:
    cells = ", ".join(f"{name} {k['median_us']:,.1f} us {k['speedup']:.2f}x err {k['error']:.2%}"
                      + ("" if k["passed"] else " FAIL") for name, k in row["kernels"].items())
    print(f"[{row['problem']}] {row['load']}: {cells}", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--problems", nargs="+", choices=tuple(PROBLEMS), default=list(PROBLEMS))
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--any-triton", action="store_true", help="allow a Triton version other than the engine's")
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"{args.output_dir} is not empty; choose a new dated directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import flashinfer
    import sglang
    import torch
    import triton

    if triton.__version__ != ENGINE_TRITON and not args.any_triton:
        raise SystemExit(f"Triton {triton.__version__} is not the engine's {ENGINE_TRITON}: results would not "
                         "describe the engine's kernels. See the docstring, or pass --any-triton.")
    environment = {"created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "torch": torch.__version__,
                   "triton": triton.__version__, "sglang": sglang.__version__, "flashinfer": flashinfer.__version__,
                   "device": torch.cuda.get_device_name(0), "rounds": args.rounds, "layer": LAYER, "seed": args.seed}
    results: list[dict[str, Any]] = []
    for problem in args.problems:
        results += PROBLEMS[problem](args.rounds, args.seed)
        (args.output_dir / "kernel_track.json").write_text(json.dumps({"environment": environment, "results": results}) + "\n")
    report = write_report(results, environment)
    (args.output_dir / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
