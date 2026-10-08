# Phase 6 — Paged KV and CUDA Graph decode

Final saved validation: 2026-09-30. One NVIDIA L4; pinned Qwen3-4B FP8 checkpoint; greedy exact-length output. One warm-up and three measured repetitions per point. Loading, tokenization, and warm-up are excluded. Latencies are milliseconds; TPS is aggregate completed output tokens/s unless explicitly labeled per request.

Block size 16; 32,768 physical token slots; `sm89` FP8 projections and BF16 activations. Default numerical policies; dense prefill for the main graph matrix. All 27 main points and all 4 packed controls passed exact-reference and repeat-stability checks: **31/31 points**.

| Prompt / output | Batch | Contiguous TTFT P50 | Contiguous TPOT P50 | Hybrid TTFT P50 | Hybrid TPOT P50 | Graph TTFT P50 | Graph TPOT P50 | Graph TTFT P95 | Exact tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 / 32 | 1 | 57.60 | 56.03 | 61.77 | 60.50 | 59.52 | 21.95 | 60.64 | pass |
| 128 / 32 | 2 | 58.63 | 57.22 | 67.72 | 60.62 | 59.98 | 22.38 | 61.42 | pass |
| 128 / 32 | 4 | 86.93 | 56.77 | 101.60 | 60.84 | 92.49 | 22.87 | 93.01 | pass |
| 512 / 64 | 1 | 88.70 | 56.60 | 102.47 | 60.48 | 94.88 | 22.78 | 96.02 | pass |
| 512 / 64 | 2 | 182.32 | 57.00 | 220.24 | 61.07 | 192.47 | 23.24 | 192.74 | pass |
| 512 / 64 | 4 | 391.70 | 56.44 | 479.34 | 60.75 | 406.59 | 24.03 | 408.68 | pass |
| 2,048 / 128 | 1 | 407.47 | 56.79 | 496.66 | 60.36 | 423.42 | 24.71 | 424.02 | pass |
| 2,048 / 128 | 2 | 904.65 | 56.95 | 1,094.32 | 60.77 | 926.91 | 25.90 | 934.28 | pass |
| 2,048 / 128 | 4 | 1,885.65 | 56.24 | 2,281.71 | 60.68 | 1,910.85 | 28.28 | 1,911.00 | pass |

Graph runs measure one stable request group (1/2/4 requests), while contiguous/hybrid runs execute all four workload requests. These are request-level latency comparisons; whole-run throughput is not a matched comparison. Graph capture is excluded. GPU/system sampling was disabled, so GPU utilization and peak VRAM are unavailable for this matrix.

## Packed-prefill controls

Prompt 128 / output 32, one warm-up and three measured repetitions.

| Mode | Batch | TTFT P50 | TPOT P50 | Exact tokens |
| --- | --- | --- | --- | --- |
| paged_packed | 1 | 57.80 | 58.45 | pass |
| paged_graph | 1 | 57.65 | 22.45 | pass |
| paged_packed | 4 | 124.66 | 59.13 | pass |
| paged_graph | 4 | 127.47 | 23.32 | pass |

## Allocator capacity

Saved 2026-09-27; CPU allocator experiment, 32,768-slot budget, seed 17. These are allocation results, not model throughput.

| Policy | Requests admitted | Internal waste (%) | Budget holding real tokens (%) |
| --- | --- | --- | --- |
| Contiguous final-length reservation | 25 | 0.00 | 94.8 |
| Contiguous longest-length reservation | 15 | 44.33 | 55.4 |
| Paged block 8 | 25 | 0.08 | 94.8 |
| Paged block 16 | 25 | 0.08 | 94.8 |
| Paged block 32 | 25 | 0.08 | 94.8 |
| Paged block 64 | 25 | 0.69 | 94.8 |

Churn: 600 FIFO requests growing by one token per decode step.

| Policy | Decode steps | Mean active requests | Budget holding real tokens (%) | Steps blocked by fragmentation |
| --- | --- | --- | --- | --- |
| Contiguous final-length reservation | 2176 | 20.59 | 86.9 | 1280 |
| Contiguous longest-length reservation | 3040 | 14.74 | 62.2 | 0 |
| Paged block 8 | 2080 | 21.54 | 90.9 | 0 |
| Paged block 16 | 2080 | 21.54 | 90.9 | 0 |
| Paged block 32 | 2080 | 21.54 | 90.9 | 0 |
| Paged block 64 | 2080 | 21.54 | 90.9 | 0 |

Default-policy validation is limited to the saved pinned-model corpus. The historical `accurate` numerical policy can still differ from BF16 SDPA greedy outputs. The final validation supersedes older failing hybrid runs; those raw runs remain local for diagnosis. Notes: [paged KV](../docs/phase_notes/paged_kv.md).

Allocator source: `results/kv_capacity_20260927/kv_capacity.json`.

## Saved sources

Paths are relative to the local `results/` directory. Raw JSON/JSONL stays local. The commit is the recorded parent revision; a dirty flag means the run included local changes.

| Result JSON | Captured (UTC) | Parent commit | Dirty |
| --- | --- | --- | --- |
| `phase6_fixes_20260930/validation/contiguous_short_b1.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/validation/paged_hybrid_s16_short_b1.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/validation/paged_graph_s16_short_b1.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/validation/contiguous_short_b2.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/validation/paged_hybrid_s16_short_b2.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/validation/paged_graph_s16_short_b2.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/validation/contiguous_short_b4.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/validation/paged_hybrid_s16_short_b4.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/validation/paged_graph_s16_short_b4.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/validation/contiguous_medium_b1.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/validation/paged_hybrid_s16_medium_b1.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/validation/paged_graph_s16_medium_b1.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/validation/contiguous_medium_b2.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/validation/paged_hybrid_s16_medium_b2.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/validation/paged_graph_s16_medium_b2.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/validation/contiguous_medium_b4.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/validation/paged_hybrid_s16_medium_b4.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/validation/paged_graph_s16_medium_b4.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/validation/contiguous_long_b1.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/validation/paged_hybrid_s16_long_b1.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/validation/paged_graph_s16_long_b1.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/validation/contiguous_long_b2.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/validation/paged_hybrid_s16_long_b2.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/validation/paged_graph_s16_long_b2.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/validation/contiguous_long_b4.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/validation/paged_hybrid_s16_long_b4.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/validation/paged_graph_s16_long_b4.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/packed_validation/paged_packed_s16_short_b1.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/packed_validation/paged_graph_s16_short_b1.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/packed_validation/paged_packed_s16_short_b4.json` | 2026-09-30 | `e3ff495` | true |
| `phase6_fixes_20260930/packed_validation/paged_graph_s16_short_b4.json` | 2026-09-30 | `e3ff495` | true |
