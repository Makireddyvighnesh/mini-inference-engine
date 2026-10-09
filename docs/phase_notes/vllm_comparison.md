# MiniLLM-L4 vs vLLM (2026-10-08)

Same L4, same pinned Qwen3-4B FP8 checkpoint, identical prompt token IDs, output lengths
(EOS ignored), greedy decoding, and arrival times; at most 32 requests in flight; prefix
caching off for both; 1 warm-up + 3 measured runs per case. MiniLLM ran its best
configuration (mixed batching, adaptive chunks, CUDA Graph decode, fused kernels); vLLM
0.26.0 ran its defaults (torch.compile + full/piecewise CUDA graphs, chunked prefill with
2,048 tokens per step). Command: `benchmarks/commands/run_vllm_compare.py` (its docstring
has the three-step run, including the vLLM environment settings).

Caveats: vLLM logged that it has no tuned W8A8 block-FP8 kernel config for the L4 and used
its default; vLLM token times are taken when the streamed token reaches the caller (after
its engine-core IPC), MiniLLM's when the token is read back on the host.

## Prefill (one request): time to first token

| Case | TTFT (MiniLLM / vLLM / ratio) |
|---|---|
| 128-token prompt | 49.5 ms / 32.1 ms / 1.54x |
| 512-token prompt | 77.5 ms / 69.5 ms / 1.11x |
| 2,048-token prompt | 354.2 ms / 259.0 ms / 1.37x |
| 8,192-token prompt | 1,680.4 ms / 1,314.4 ms / 1.28x |

## Decode (512-token prompts, 128 outputs, all at once)

| Case | Time per token (MiniLLM / vLLM / ratio) | Output tok/s (MiniLLM / vLLM / ratio) |
|---|---|---|
| 1 request | 20.7 ms / 22.8 ms / 0.91x | 47.3 / 43.3 / 109% |
| 8 requests | 25.9 ms / 26.0 ms / 1.00x | 268.6 / 277.1 / 97% |
| 32 requests | 44.6 ms / 40.0 ms / 1.11x | 535.5 / 643.7 / 83% |

## Serving (arrivals over time, 128 outputs; long mix 64)

| Case | TTFT p50 (MiniLLM / vLLM / ratio) | TTFT p95 (MiniLLM / vLLM / ratio) | Time per token (MiniLLM / vLLM / ratio) | Worst gap (MiniLLM / vLLM / ratio) | Output tok/s (MiniLLM / vLLM / ratio) |
|---|---|---|---|---|---|
| 16 requests, 128-1,024 tokens, every 150 ms | 121.7 ms / 99.8 ms / 1.22x | 302.1 ms / 186.1 ms / 1.62x | 31.4 ms / 29.7 ms / 1.06x | 146.3 ms / 135.2 ms / 1.08x | 359.1 / 358.4 / 100% |
| 6,144-token prompts among short ones, every 400 ms | 968.6 ms / 1,025.7 ms / 0.94x | 2,939.3 ms / 1,441.2 ms / 2.04x | 70.8 ms / 52.9 ms / 1.34x | 160.4 ms / 368.8 ms / 0.43x | 54.8 / 69.8 / 78% |
| 64 requests, 128-2,048 tokens, every 50 ms | 6,304.8 ms / 4,505.5 ms / 1.40x | 11,086.8 ms / 8,451.0 ms / 1.31x | 70.8 ms / 58.3 ms / 1.21x | 174.3 ms / 299.0 ms / 0.58x | 406.0 / 489.3 / 83% |

## Output agreement

131/131 requests (first measured run of each case) produced identical greedy tokens in both engines.

Ratios: time columns are MiniLLM / vLLM (above 1.00x means MiniLLM is slower); tok/s columns are MiniLLM as a share of vLLM.

## Interpretation

- Decode at 1-8 requests and light serving are at parity (batch 1 is 9% faster).
- Prefill is 1.1-1.5x slower (vLLM compiles and fuses more of the forward; at 128 tokens
  fixed per-forward overhead dominates).
- Under load, MiniLLM's worst decode pause is ~2x shorter (adaptive chunks keep busy steps
  near 150 ms; vLLM's 2,048-token chunks reach ~370 ms over a 6K prefix), but time per token
  is 20-34% higher and throughput 78-83% of vLLM. Likely causes, not yet profiled: MiniLLM's
  mixed prompt+decode steps run eagerly (vLLM graphs them piecewise), one attention call per
  prompt chunk, and the per-layer prefix gather for continuation chunks.
- All 393 measured requests produced identical greedy tokens in both engines.

## After the split FP8 GEMM (2026-10-09)

Profiling showed the FP8 GEMM was 81% of prefill GPU time (GPU ~98% busy, so CUDA Graphs
for mixed steps would gain little). The fused kernel re-quantized activations inside the
GEMM loop for every output tile; the split path quantizes once and uses 128-row tiles.
Per-layer projections at 2,048 tokens: 7,163 -> 4,973 us (vLLM ~4,864 us). Outputs are
bitwise identical to the fused kernel (468 exactness cases, every autotune config; policy
gate 112/112). Same workloads; vLLM numbers from the run above:

| Case | MiniLLM before | MiniLLM now | vLLM |
|---|---|---|---|
| Prefill 512 / 2,048 / 8,192 tokens | 77.5 / 354 / 1,680 ms | 63.4 / 287 / 1,379 ms | 69.5 / 259 / 1,314 ms |
| Decode, 32 requests | 536 tok/s | 593 tok/s | 644 tok/s |
| Serving 16 requests, TTFT p50 | 122 ms | 97 ms | 100 ms |
| 6,144-token mix, output tok/s | 55 | 63 | 70 |
| 64 requests every 50 ms, output tok/s | 406 | 461 | 489 |

Heavy-load throughput is now 90-94% of vLLM (was 78-83%), with ~2x shorter worst decode
pauses. Remaining: 128-token prefill is fixed per-forward overhead (50 vs 32 ms); q/k/v
and gate/up still quantize the same input separately (merged projections would save ~6%).
