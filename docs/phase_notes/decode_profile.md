# Decode-step profile: where the ~71 ms goes

Date: 2026-09-27

## Question

Phases 1–5 all measured about 71 ms per decode step, unchanged from batch 1
to 4 and from 128 to 2,048 context tokens, with the GPU 34–49% busy.
GPU-side optimizations (a preallocated KV cache, trimming padding) gave no
speedup, and recomputing the whole prompt was free at 128 tokens/batch 1.
This profile checks where the step time actually goes.

## Method

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_profile_decode \
  --output-dir minillm_l4/results/profile_decode_20260927_auto
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_profile_decode \
  --fp8-kernel-path sm89 \
  --output-dir minillm_l4/results/profile_decode_20260927_sm89
```

For each shape the command prefills a random-token prompt, runs 5 warm-up
decode steps, times 20 steps without the profiler, and then records 5 steps
with `torch.profiler` (CPU and CUDA activity). The decode loop mirrors the
manual backend: explicit attention mask, `DynamicCache`, and one host read of
the next tokens per step. Because each step ends with that host read, all of
a step's GPU work finishes inside the step's CPU window; GPU-busy time is the
union of kernel and memcpy intervals in that window. The first profiled step
is dropped. Step times in the tables are the unprofiled P50; the profiler
itself adds 15–25 ms of host time per step.

Each output directory holds `summary.json`, a Chrome/Perfetto trace per shape
(`trace_p*_b*.json`, open at ui.perfetto.dev), and the top CPU operators and
GPU kernels (`cpu_ops_*.txt`, `gpu_kernels_*.txt`).

## Results

Weights on the GPU are 4.11 GiB, so reading them once at the L4's 300 GB/s
takes at least **14.7 ms** per step.

| FP8 path | Prompt | Batch | Step P50 | GPU busy P50 | GPU busy | GPU ops/step |
|---|---:|---:|---:|---:|---:|---:|
| Transformers `auto` | 128 | 1 | 71.10 ms | 22.79 ms | 32% | 2,081 |
| Transformers `auto` | 2,048 | 1 | 78.76 ms | 25.47 ms | 32% | 2,081 |
| Transformers `auto` | 2,048 | 4 | 79.18 ms | 33.35 ms | 42% | 2,081 |
| project `sm89` | 128 | 1 | 53.32 ms | 22.69 ms | 43% | 2,081 |
| project `sm89` | 2,048 | 1 | 62.20 ms | 25.38 ms | 41% | 2,081 |
| project `sm89` | 2,048 | 4 | 62.51 ms | 33.22 ms | 53% | 2,081 |

GPU time per step on the `auto` path (128 tokens, batch 1):

| Work | Per step |
|---|---:|
| FP8 block-scaled matmul kernels (252 calls, ~64 µs each) | ~16 ms |
| `lm_head` GEMV (BF16, ~0.8 GB of weights) | ~3 ms |
| Element-wise and reduction kernels | ~3 ms |

On the host, each call into the Transformers FP8 matmul op costs about
115 µs of CPU time, nearly twice its 64 µs kernel, or about 29 ms per step
across 252 calls. `cudaLaunchKernel` adds about 11 ms per step (roughly 6 µs
per launch); the remainder is Python and Transformers module overhead.

## Interpretation

Decode through Transformers is **host-bound (launch-bound)**. Each token
issues about 2,000 small GPU operations; for many of them the CPU work to
prepare and launch the operation takes longer than the GPU work itself, so
the launch queue runs dry and the GPU idles for roughly two-thirds of the
step.

The `sm89` comparison is the causal check. Swapping only the FP8 matmul
path leaves GPU work (22.7 ms) and the operation count (2,081) unchanged,
yet the step drops from 71.1 ms to 53.3 ms (−25%) because each call costs
less CPU time. A step that responds to host-only changes and not to
GPU-only changes is host-bound.

This explains the Phase 1–5 observations:

- Step time is flat across batch 1–4: batch 4 adds about 8 ms of GPU work,
  which fits into GPU idle time, and the launch count does not change. That
  is why throughput scaled almost linearly with batch size.
- Removing KV-cache copies or padding saves GPU time the GPU was not short
  of, so the step does not get shorter.
- Recomputing the prompt is free while its extra GPU work fits in idle time
  (128 tokens, batch 1) and costs up to 6.2× once it no longer does
  (512 tokens, batch 4).

With all host overhead removed, a step would approach its GPU time, about
23 ms at 128 tokens and batch 1, roughly 3× faster than 71 ms. The remaining
gap to the 14.7 ms floor is GPU-side: the FP8 kernels run below peak
bandwidth, and the `lm_head` and small kernels add their own traffic. CUDA
Graph replay (one launch for the whole step) and fused kernels are the
direct fixes; Phase 6's fixed-shape graph path reached about 30 ms per step
on its own benchmark shape.

## Measurement pitfall

`record_function` step markers are mirrored onto the GPU timeline as
annotations that span the whole step. Counting them as GPU work reported the
GPU as ~100% busy, which contradicted `nvidia-smi`. Excluding the markers
gave 32%, which agrees with the 34% `nvidia-smi` utilization measured in
Phase 1. The command filters them out, and
`tests/test_profile_decode.py` covers the interval merging and per-step
attribution.

## Open question

On the `auto` path, 2,048-token steps take about 79 ms here, while Phase 1
and Phase 3 measured 70–72 ms TPOT at 2,048/128. GPU-busy time rises only
2.7 ms from 128 to 2,048 tokens, so the extra ~8 ms is host time; its source
is not identified yet.

## Provenance

Commit `73bdf69` plus this command, uncommitted at run time (the worktree
also held unrelated uncommitted Phase 6/7 edits that this decode path does
not import). Model
`Qwen/Qwen3-4B-Instruct-2507-FP8` at snapshot
`8591804019c8b22094c3b5b4454e0edc05dffc98`, PyTorch 2.13.0+cu130, one
NVIDIA L4. Prompts are uniform random token IDs, so these runs have no
exact-token check; they measure time, not outputs.
