"""Q6: why is FlashAttention faster when the math is the same?

Times one attention layer at Qwen3-4B shapes (32 query heads, 8 KV heads,
head_dim 128, bf16) for context lengths 512 .. 32k, with no model loaded.

Prefill (one prompt of N tokens, causal):
* naive         - Transformers "eager": copy K/V to 32 heads, Q@K^T, causal
                  mask, fp32 softmax, @V. Writes the full N x N score matrix;
* sdpa_math     - PyTorch SDPA math backend (also materializes N x N);
* sdpa_efficient- PyTorch SDPA memory-efficient backend (xFormers style; it has
                  no GQA support, so it gets K/V pre-copied to 32 heads);
* sdpa_flash    - PyTorch SDPA FlashAttention-2 backend (the engine's prefill);
* triton_packed - our Triton packed-prefill kernel over paged K/V: online
                  softmax (no N x N matrix) but one program per query row, so
                  it re-reads K/V for every row (no query tiling).

Decode (batch 8, one new token per request, attending N cached tokens):
* naive, sdpa_math, sdpa_flash, triton_paged (engine decode kernel
  with the engine's settings: SDPA-compatible mode, one KV split), and
  triton_split8 (the same kernel family with 8 KV splits for long contexts).

Per case: median time (CUDA events), extra GPU memory (peak minus before),
achieved TFLOPS, and max abs error against an FP32 reference. A case that runs
out of memory or has no kernel is recorded as such.

  .conda-env/bin/python -m minillm_l4.benchmarks.commands.run_attention_compare \\
      --output-dir minillm_l4/results/attention_compare_<date>
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import warnings

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

warnings.filterwarnings("ignore", message=".*(kernel not used|runtime disabled|same num_heads).*")

HEADS, KV_HEADS, DIM, BLOCK = 32, 8, 128, 16
GROUP = HEADS // KV_HEADS
SCALE = 1.0 / math.sqrt(DIM)
SIZES = (512, 1024, 2048, 4096, 8192, 16384, 32768)
DECODE_BATCH = 8


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory is not empty: {args.output_dir}")
    return args


def timed(fn, decode: bool) -> float:
    """Median milliseconds of fn() after one warm-up; slow cases run fewer times."""

    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    first = start.elapsed_time(end)
    reps = 3 if first > 1000 else 7
    inner = max(1, min(50, int(5 / max(first, 1e-3)))) if decode else 1
    samples = []
    for _ in range(reps):
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / inner)
    return sorted(samples)[len(samples) // 2]


def extra_memory(fn) -> int:
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del out
    return peak - before


def repeat_kv(x):
    b, h, n, d = x.shape
    return x[:, :, None].expand(b, h, GROUP, n, d).reshape(b, h * GROUP, n, d)


def to_pages(k):
    """[batch, kv_heads, N, D] -> pages [batch*N/16, kv_heads, 16, D] and block tables."""

    b, h, n, d = k.shape
    pages = k.view(b, h, n // BLOCK, BLOCK, d).permute(0, 2, 1, 3, 4).reshape(b * n // BLOCK, h, BLOCK, d)
    tables = torch.arange(b * n // BLOCK, device=k.device, dtype=torch.int32).view(b, n // BLOCK)
    return pages.contiguous(), tables


# ---- prefill ----------------------------------------------------------------

def prefill_naive(q, k, v):
    n = q.shape[2]
    k, v = repeat_kv(k), repeat_kv(v)
    scores = torch.matmul(q, k.transpose(2, 3)) * SCALE
    mask = torch.full((n, n), float("-inf"), device=q.device, dtype=q.dtype).triu_(1)
    probs = F.softmax(scores + mask, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(probs, v)


def prefill_sdpa(backend):
    def run(q, k, v):
        with sdpa_kernel([backend]):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=SCALE, enable_gqa=True)
    return run


def prefill_efficient(q, k, v):
    # This backend has no GQA support: give it K/V already copied to 32 heads (copy not timed).
    k, v = repeat_kv(k).contiguous(), repeat_kv(v).contiguous()

    def run():
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=SCALE)
    return run


def prefill_triton(q, k, v):
    from minillm_l4.engine.kv_cache.triton_packed_attention import triton_packed_prefill_attention

    n = q.shape[2]
    kp, tables = to_pages(k)
    vp, _ = to_pages(v)
    seq = torch.zeros(n, device=q.device, dtype=torch.int32)
    pos = torch.arange(n, device=q.device, dtype=torch.int32)
    flat_q = q[0].transpose(0, 1).contiguous()
    return lambda: triton_packed_prefill_attention(flat_q, kp, vp, seq, pos, tables, scale=SCALE).transpose(0, 1)[None]


def prefill_reference(q, k, v, rows=512):
    n = q.shape[2]
    k32, v32 = repeat_kv(k).float(), repeat_kv(v).float()
    out = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    keys = torch.arange(n, device=q.device)
    for s in range(0, n, rows):
        e = min(n, s + rows)
        scores = torch.matmul(q[:, :, s:e].float(), k32[:, :, :e].transpose(2, 3)) * SCALE
        scores.masked_fill_(keys[None, :e] > torch.arange(s, e, device=q.device)[:, None], float("-inf"))
        out[:, :, s:e] = torch.matmul(torch.softmax(scores, -1), v32[:, :, :e])
    return out


# ---- decode -----------------------------------------------------------------

def decode_naive(q, k, v):
    k, v = repeat_kv(k), repeat_kv(v)
    probs = F.softmax(torch.matmul(q, k.transpose(2, 3)) * SCALE, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(probs, v)


def decode_sdpa(backend):
    def run(q, k, v):
        with sdpa_kernel([backend]):
            return F.scaled_dot_product_attention(q, k, v, scale=SCALE, enable_gqa=True)
    return run


def decode_triton(split):
    def build(q, k, v):
        from minillm_l4.engine.kv_cache.triton_paged_attention import triton_paged_decode_attention

        b, _, n, _ = k.shape
        kp, tables = to_pages(k)
        vp, _ = to_pages(v)
        lengths = torch.full((b,), n, device=q.device, dtype=torch.int32)
        # The engine's settings: no GQA-reuse kernel, no host-side metadata checks (as in graph replay).
        if split == 1:
            return lambda: triton_paged_decode_attention(q, kp, vp, tables, lengths, scale=SCALE, use_gqa_reuse=False,
                                                         sdpa_compat=True, _validate_metadata=False).transpose(1, 2)
        return lambda: triton_paged_decode_attention(q, kp, vp, tables, lengths, scale=SCALE, use_gqa_reuse=False,
                                                     split_count=split, max_sequence_length=n,
                                                     _validate_metadata=False).transpose(1, 2)
    return build


def decode_reference(q, k, v):
    probs = torch.softmax(torch.matmul(q.float(), repeat_kv(k).float().transpose(2, 3)) * SCALE, -1)
    return torch.matmul(probs, repeat_kv(v).float())


PREFILL = (("naive", prefill_naive, False), ("sdpa_math", prefill_sdpa(SDPBackend.MATH), False),
           ("sdpa_efficient", prefill_efficient, True),
           ("sdpa_flash", prefill_sdpa(SDPBackend.FLASH_ATTENTION), False), ("triton_packed", prefill_triton, True))
DECODE = (("naive", decode_naive, False), ("sdpa_math", decode_sdpa(SDPBackend.MATH), False),
          ("sdpa_flash", decode_sdpa(SDPBackend.FLASH_ATTENTION), False),
          ("triton_paged", decode_triton(1), True), ("triton_split8", decode_triton(8), True))


def run_case(stage, name, impl, built, q, k, v, reference, flops):
    case = {"stage": stage, "backend": name, "tokens": int(k.shape[2])}
    fn = None
    try:
        fn = impl(q, k, v) if built else (lambda: impl(q, k, v))
        out = fn()
        torch.cuda.synchronize()
        error = (out.float() - reference).abs().max().item()
        del out
        memory = extra_memory(fn)
        ms = timed(fn, decode=stage == "decode")
        case.update({"status": "ok", "ms": ms, "tflops": flops / (ms / 1000) / 1e12, "extra_memory_bytes": memory,
                     "max_abs_error": error})
    except torch.OutOfMemoryError:
        case["status"] = "oom"
    except RuntimeError as exc:  # e.g. "No available kernel" for a backend/shape
        case.update({"status": "unsupported", "error": str(exc).splitlines()[0][:200]})
    finally:
        del fn
        gc.collect()
        torch.cuda.empty_cache()
    return case


@torch.inference_mode()
def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(0)
    result = {"created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gpu": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "heads": HEADS, "kv_heads": KV_HEADS, "head_dim": DIM, "dtype": "bf16",
              "decode_batch": DECODE_BATCH, "cases": []}
    dev, dt = "cuda", torch.bfloat16

    def save(case):
        result["cases"].append(case)
        if case["status"] == "ok":
            print(f"[{case['stage']:7s}] {case['tokens']:6d} {case['backend']:15s} {case['ms']:10.3f} ms "
                  f"{case['tflops']:6.1f} TF  +{case['extra_memory_bytes'] / 2**20:8.0f} MiB  "
                  f"err {case['max_abs_error']:.2e}", flush=True)
        else:
            print(f"[{case['stage']:7s}] {case['tokens']:6d} {case['backend']:15s} {case['status']}", flush=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "attention_compare.json").write_text(json.dumps(result, indent=1) + "\n")

    for n in args.sizes:
        q = torch.randn(1, HEADS, n, DIM, device=dev, dtype=dt)
        k = torch.randn(1, KV_HEADS, n, DIM, device=dev, dtype=dt)
        v = torch.randn(1, KV_HEADS, n, DIM, device=dev, dtype=dt)
        reference = prefill_reference(q, k, v)
        flops = 2 * HEADS * DIM * n * n  # QK^T and PV over the causal half: 2 * (2*N*N*D/2) per head
        for name, impl, built in PREFILL:
            save(run_case("prefill", name, impl, built, q, k, v, reference, flops))
        del q, k, v, reference

        q = torch.randn(DECODE_BATCH, HEADS, 1, DIM, device=dev, dtype=dt)
        k = torch.randn(DECODE_BATCH, KV_HEADS, n, DIM, device=dev, dtype=dt)
        v = torch.randn(DECODE_BATCH, KV_HEADS, n, DIM, device=dev, dtype=dt)
        reference = decode_reference(q, k, v)
        flops = 4 * DECODE_BATCH * HEADS * DIM * n
        for name, impl, built in DECODE:
            save(run_case("decode", name, impl, built, q, k, v, reference, flops))
        del q, k, v, reference
        gc.collect()
        torch.cuda.empty_cache()
    print(f"wrote {args.output_dir / 'attention_compare.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
