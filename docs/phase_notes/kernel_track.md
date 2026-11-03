# Kernel track: our kernels vs SGLang's (2026-11-03)

We judged our kernels the way Meta's Open Kernel Challenge judges entries, on our own GPU and model. This found one real gap, and fixing it made decode up to 7% faster per token.

Commands: `run_kernel_track` (the judge) and `run_tensor_core_decode` (the real engine). Raw data: `results/kernel_track_t371_20261103/`, `results/kernel_track_tc_20261103/`, `results/tensor_core_decode_20261103/`.

## Short answer

- **FP8 projections were already fine:** 1.02x SGLang overall, faster at decode sizes and about even at prefill sizes.
- **Decode attention was the real gap:** 0.54-0.63x at 1 request, 0.89-0.90x at 8 requests, about even at 32.
- **New kernel, `decode_tensor_core=True`:** 1.04-1.15x FlashInfer at every load, and 1.07x overall.
- **In the real engine:** 0.5-7.0% less time per token, faster in all 18 cases, with an average of 3.2%.
- **Tokens:** 187 of 189 requests were identical to the exact path. It stays opt-in, and the default path is unchanged.

## How the judge works

Both kernels run in one process, each as a CUDA graph, on the same inputs. They take turns, the order flips each round, and L2 is flushed between replays. A load's speedup is the median of the per-round ratios, and a problem's score is the geometric mean over its loads. Outputs must match a double-precision reference within a tolerance.

SGLang's real kernels are the baseline: Triton `w8a8_block_fp8_matmul_triton` for FP8 and FlashInfer `fa2` for decode.

## Decode attention

Scores are against FlashInfer. Above 1.00x beats it.

| Requests x context | Exact kernel (us, score) | Tensor-core kernel (us, score) |
|---:|---:|---:|
| 1 x 512 | 48.1, 0.58x | 24.6, 1.13x |
| 1 x 1,024 | 72.7, 0.57x | 35.8, 1.15x |
| 1 x 2,048 | 115.7, 0.54x | 57.3, 1.07x |
| 8 x 512 | 0.89x | 99.3, 1.06x |
| 8 x 1,024 | 0.89x | 187.5, 1.05x |
| 8 x 2,048 | 0.90x | 362.5, 1.04x |
| 32 x 512 | 1.02x | 365.6, 1.04x |
| 32 x 1,024 | 1.04x | 661.5, 1.04x |
| 32 x 2,048 | 1.03x | 1,175.5, 1.04x |

Largest error against the reference: 0.23% for the new kernel, 0.30% for FlashInfer.

### Why it is faster

- **One K/V read serves 4 query heads.** Qwen3 has 4 query heads per KV head, and the exact kernel reads the same K/V once per head.
- **The dot products run on tensor cores.** An older grouped kernel kept a 4x128x128 FP32 block in registers instead, which is why grouped decode once measured far slower. That result was about that arithmetic, not about grouping.
- **Small batches split each KV head into 4 programs** to fill the L4's 58 SMs. From batch 4 the merge step costs more than this gains, so those batches use one split.

## In the real engine

Recommended config (mixed batching, adaptive chunks, CUDA Graph decode, fused kernels, merged projections), 128 new tokens. TPOT is the median gap while every request is decoding. Each mode ran in two blocks with the order flipped.

| Batch | Prompt tokens | Exact TPOT (ms) | Tensor-core TPOT (ms) | Gain | Identical |
|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 20.42 | 20.07 | 1.7% | 1/1 |
| 1 | 1,024 | 21.17 | 20.47 | 3.3% | 1/1 |
| 1 | 2,048 | 22.63 | 21.13 | 6.6% | 1/1 |
| 2 | 512 | 21.13 | 20.79 | 1.6% | 2/2 |
| 2 | 1,024 | 22.04 | 21.40 | 2.9% | 2/2 |
| 2 | 2,048 | 23.94 | 22.56 | 5.8% | 2/2 |
| 4 | 512 | 21.89 | 21.55 | 1.5% | 4/4 |
| 4 | 1,024 | 23.23 | 22.77 | 2.0% | 4/4 |
| 4 | 2,048 | 26.04 | 25.05 | 3.8% | 4/4 |
| 8 | 512 | 24.04 | 23.23 | 3.4% | 8/8 |
| 8 | 1,024 | 26.80 | 25.54 | 4.7% | 8/8 |
| 8 | 2,048 | 32.44 | 30.17 | 7.0% | 8/8 |
| 16 | 512 | 27.35 | 27.02 | 1.2% | 15/16 |
| 16 | 1,024 | 32.24 | 32.07 | 0.5% | 16/16 |
| 16 | 2,048 | 41.92 | 41.63 | 0.7% | 16/16 |
| 32 | 512 | 34.55 | 33.49 | 3.1% | 31/32 |
| 32 | 1,024 | 44.30 | 42.85 | 3.3% | 32/32 |
| 32 | 2,048 | 64.23 | 61.48 | 4.3% | 32/32 |

- **The kernel is about 2x faster at batch 1, but a step gains only 2-7%.** Reading the weights dominates a decode step, and attention is a small part of it.
- **The gain grows with context,** because attention's share of the step grows with it.
- **Batch 16 gained least.** This is not explained yet; the judge shows 1.04x FlashInfer at the same size.
- **Two requests differ,** both at output token 105 on 512-token prompts (batch 16 and batch 32). They are probably the same near-tied token on the same prompt, but the run does not save tokens, so this is not confirmed. No HF reference was run.
- **The default path is unchanged:** the policy gate gave 160/160 identical outputs across 20 policies after this change. The gate does not cover `decode_tensor_core`.

## FP8 projections

Scores are against SGLang's FP8 path, including activation quantization.

| Rows | qkv | o | gate_up | down |
|---:|---:|---:|---:|---:|
| 1 | 1.02x | 1.07x | 1.03x | 1.05x |
| 8 | 1.02x | 1.07x | 1.03x | 1.05x |
| 32 | 1.02x | 1.07x | 1.03x | 1.07x |
| 512 | 0.98x | 0.99x | 0.99x | 1.06x |
| 2,048 | 0.96x | 0.98x | 0.99x | 0.96x |

Not done yet: for 4,096+ rows a plain-`range` loop with 64x128 tiles is 4-11% faster than the engine's split GEMM, in the engine's own env, and bit-identical on the tested inputs. It does not help at 2,048 rows or below, so this was left out.

## The Triton version matters

The first judge run used the SGLang env's Triton 3.8, while the engine uses 3.7.1. It made our prefill projections look 4-12% behind SGLang, which was mostly false:

| Our kernel | Triton 3.8 | Triton 3.7.1 |
|---|---:|---:|
| gate_up FP8, 2,048 rows, score | 0.88x | 0.99x |
| Attention, 1 x 2,048, score | 0.66x | 0.54x |

Triton 3.8 made our FP8 kernel slower and our attention faster, so the first run misjudged both. A bisect at the same tile settings on 3.8 traced the FP8 loss to `tl.range(num_stages=N)` instead of plain `range`: up to 28% slower. 64-bit addressing cost nothing. The judge now refuses any Triton other than the engine's unless `--any-triton` is passed. Upgrading the engine to 3.8 would need these numbers rechecked.

## Limits

- The judge uses random inputs and one layer's real weights, and it runs our kernels on torch 2.14 rather than the engine's 2.13. Only the engine table above uses real prompts.
- Judge cases use requests of equal length. Real traffic mixes lengths.
- These runs used a dirty working tree (uncommitted scheduler changes are mixed in).
- Tensor-core decode has not yet been run through `run_engine_compare`, so there is no new four-engine number, and TPOT here is defined differently from that report (only steps where every request decodes).
