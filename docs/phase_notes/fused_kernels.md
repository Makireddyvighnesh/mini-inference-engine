# Fused Triton kernels for norms, SiLU, and RoPE

Status (2026-10-08): complete. Four fused kernels in `engine/kernels/fused.py` replace
Qwen3's memory-bound elementwise ops. Every kernel is bit-identical to the unfused
PyTorch ops, so greedy tokens do not change; prefill is 10-23% faster and eager decode
20% faster.

## Why these kernels (profile)

`torch.profiler` on a 2048-token prefill (sm89 FP8 path, 388 ms of GPU kernels): FP8
matmuls 68%; the rest was dominated by small memory-bound ops that each read and write
a full activation tensor:

| Unfused work (36 layers) | GPU time |
| --- | --- |
| fp32/bf16 casts and copies | 51.4 ms |
| multiplies (SiLU gate, norm scaling, RoPE) | 17.3 ms + 8.3 ms + 6.7 ms + ... |
| SiLU | 10.0 ms |
| RMSNorm `pow` / `mean` | 11.5 ms / 3.8 ms |
| residual adds, RoPE add/cat/neg | ~17 ms |

Grouped by fusion: SiLU x up ~27 ms, q/k per-head RMSNorm ~28 ms, RoPE ~20 ms, residual
add + hidden RMSNorm ~22 ms: about 95 ms of a ~410 ms prefill. In decode the same ops
are ~1,000 tiny kernel launches per step.

## Kernels

- `rms_norm(x, w, eps)`: one pass per row (Qwen3RMSNorm: fp32 cast, square, mean, rsqrt,
  scale, bf16 cast, weight multiply).
- `rms_norm(x, w, eps, residual=r)`: the post-attention residual add fused into the
  following RMSNorm; returns `(norm(x + r), x + r)`.
- `silu_mul(gate, up)`: `silu(gate) * up` without materializing the SiLU output.
- `apply_rotary_pos_emb(q, k, cos, sin)`: drop-in RoPE; the half rotation happens in
  registers instead of slicing, negating, and concatenating tensors.

`install_fused_kernels(model)` overrides the norm, MLP, and decoder-layer `forward`
methods on that model and patches the RoPE function in Transformers' Qwen3 module and
in `engine/kv_cache/qwen3_paged.py` (reference counted); `uninstall_fused_kernels`
restores the PyTorch path. Install pre-compiles every kernel specialization so no
Triton compile lands in a timed request (Triton specializes integer arguments
equal to 1 or divisible by 16, so RoPE is warmed for decode, aligned, and odd
token counts with both head counts, and `silu_mul` for both block sizes;
`tests/test_fused_kernels.py` checks that serving shapes compile nothing new).
Rows narrower than 128 or not a multiple of 4 fall back to PyTorch.

## Exactness

Each kernel reproduces PyTorch's arithmetic, including where it rounds:

- Intermediate bf16 roundings use integer round-to-nearest-even. A first RoPE
  prototype that rounded with `.to(bf16).to(fp32)` mismatched 14% of elements:
  Triton folded the round trip and fused multiply-adds, skipping roundings PyTorch does.
- RMSNorm's row sum follows `ATen/native/cuda/Reduce.cuh` exactly: the row is read as
  4-wide vectors over `bw` threads, each thread keeps four sequential accumulators
  combined `((a0 + a1) + a2) + a3`, then threads fold pairwise with halving offsets.
  PyTorch sets `bw` from the row count (512 threads for 1 row, 256 for 2-3, 128 for 4-7,
  64 for 8-15, 32 for 16+), and so does the kernel. A plain `tl.sum` matched most rows
  but differed on about 1 row in 1,000.
- Squares and the mean's `1/n` scaling use explicitly rounded multiplies so the
  compiler cannot fuse them into FMAs with the following adds; `exp`/`rsqrt`/division
  use the same libdevice functions as PyTorch.

Validation on the L4:

| Check | Result |
| --- | --- |
| `tests/test_fused_kernels.py` (RMSNorm rows 1-4096 x widths 128/2560, add+norm, SiLU, RoPE in prefill/packed/decode layouts, model install/uninstall) | 26/26 bitwise |
| `scripts/check_policy_equivalence.py` with fused variants (eager, chunked, adaptive, CUDA Graph, graph + adaptive) | **112/112 request outputs identical** to unfused eager |
| Full suite with `MINILLM_RUN_MODEL_TESTS=1` | 380 passed |
| Allocator retries across all benchmark cases | 0 |

## Results

`benchmarks/commands/run_fused_kernels.py`, one session, 1 warm-up + 3 measured runs
(`results/fused_kernels_20261008/`). Unfused and fused cases share the loaded model.

### Per operation

CUDA-event medians of 100 calls with real layer weights. "Bytes touched" counts each
op's reads and writes assuming nothing is cached; the L4's 48 MB L2 serves part of it,
which is why some fused kernels exceed the ~300 GB/s DRAM bandwidth.

| Shape | Fusion | Unfused | Fused | Speedup | Bytes touched |
| --- | --- | --- | --- | --- | --- |
| 2048-token prefill | RMSNorm (hidden) | 0.210 ms | 0.067 ms | 3.2x | ~189 -> 21 MB |
| 2048-token prefill | residual add + RMSNorm | 0.318 ms | 0.088 ms | 3.6x | ~220 -> 42 MB |
| 2048-token prefill | RMSNorm (q heads) | 0.783 ms | 0.098 ms | 8.0x | ~302 -> 34 MB |
| 2048-token prefill | SiLU x up | 0.863 ms | 0.530 ms | 1.6x | ~199 -> 120 MB |
| 2048-token prefill | RoPE (q + k) | 0.538 ms | 0.248 ms | 2.2x | ~210 -> 42 MB |

At decode size (8 rows) the per-op timings mostly measure Python launch overhead (each
Triton launch costs tens of microseconds on the host): norms and RoPE are still
1.6-1.7x faster (one launch instead of six or seven), while `silu_mul` is slower than
PyTorch's two launches (0.043 vs 0.029 ms). Under CUDA Graph replay host launch cost
disappears.

### End to end

| Prompt tokens | Unfused TTFT | Fused TTFT | Saved |
| --- | --- | --- | --- |
| 128 | 65.9 ms | 52.4 ms | 20% |
| 256 | 65.1 ms | 52.3 ms | 20% |
| 512 | 91.3 ms | 82.0 ms | 10% |
| 1,024 | 181.1 ms | 159.3 ms | 12% |
| 2,048 | 396.8 ms | 334.0 ms | 16% |
| 4,096 | 921.8 ms | 712.3 ms | 23% |
| 8,192 | 2,104.5 ms | 1,618.2 ms | 23% |

| Decode (512-token prompts) | Unfused | Fused | Change |
| --- | --- | --- | --- |
| 1 request, eager | 62.5 ms/token | 50.2 ms/token | -20% |
| 1 request, CUDA Graph | 22.6 ms/token | 20.7 ms/token | -8% |
| 8 requests, eager | 64.6 ms/token | 51.7 ms/token | -20% |
| 8 requests, CUDA Graph | 25.9 ms/token | 24.0 ms/token | -7% |
| 32 requests, eager | 66.7 ms/token | 54.5 ms/token | -18% |
| 32 requests, CUDA Graph | 37.9 ms/token | 35.7 ms/token | -6% |

| Serving (16 requests every 150 ms, CUDA Graphs on) | TTFT p50 unfused / fused | TPOT p50 unfused / fused | tok/s unfused / fused |
| --- | --- | --- | --- |
| Continuous | 157 / 97 ms | 34.3 / 32.3 ms | 339.7 / 357.9 |
| Mixed + adaptive | 133 / 128 ms | 34.2 / 31.7 ms | 340.1 / 356.0 |

- Prefill saves 10-23%, most at long prompts where activations are large; the small
  prompts' 20% comes from fewer launches.
- Eager decode is 20% faster: removing ~1,000 launches per step outweighs the higher
  host cost of each Triton launch.
- With CUDA Graphs the host cost is already gone, so decode gains only the GPU-side
  saving (6-8%); serving gains 5% throughput and lower TTFT from faster prefill.

## Reproduce

```bash
.conda-env/bin/python -m pytest -q minillm_l4/tests/test_fused_kernels.py
.conda-env/bin/python minillm_l4/scripts/check_policy_equivalence.py
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_fused_kernels --output-dir minillm_l4/results/fused_kernels_<date>
cd minillm_l4/scripts && ../../.conda-env/bin/python make_fused_charts.py ../results/fused_kernels_<date>/fused_kernels.json ../docs/assets/fused_kernels
```

## Limits

- The decoder layer's final residual add is not fused into the next layer's input norm
  (that needs the residual carried across layers).
- `silu_mul` is slower than PyTorch at decode size in eager mode; it is a net loss of
  about 0.5 ms per eager step, outweighed by the norm and RoPE savings.
- The RMSNorm reduction emulation targets PyTorch 2.13's `Reduce.cuh` and an fp32
  last-dim `mean`; a PyTorch upgrade must re-run the bitwise tests.
