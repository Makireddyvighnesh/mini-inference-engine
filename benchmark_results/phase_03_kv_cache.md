# Phase 3 — Explicit KV cache

Latest matrix: 2026-09-26. One NVIDIA L4; pinned Qwen3-4B FP8 checkpoint; greedy exact-length output. One warm-up and three measured repetitions per point. Loading, tokenization, and warm-up are excluded. Latencies are milliseconds; TPS is aggregate completed output tokens/s unless explicitly labeled per request.

All 15 measured points passed exact-reference checks: 9 contiguous-cache points and 6 recomputation controls. The long-context recomputation control was not remeasured in this matrix.

| Prompt / output | Batch | Cached TPOT P50 | Recompute TPOT P50 | TPOT ratio | Cached aggregate TPS | Recompute aggregate TPS | Exact tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 128 / 32 | 1 | 72.43 | 71.27 | 0.98× | 13.78 | 14.02 | pass |
| 128 / 32 | 2 | 71.35 | 82.15 | 1.15× | 27.98 | 24.47 | pass |
| 128 / 32 | 4 | 71.38 | 119.41 | 1.67× | 55.43 | 33.68 | pass |
| 512 / 64 | 1 | 72.26 | 117.78 | 1.63× | 13.77 | 8.52 | pass |
| 512 / 64 | 2 | 72.29 | 221.11 | 3.06× | 27.06 | 9.06 | pass |
| 512 / 64 | 4 | 72.73 | 451.08 | 6.20× | 51.50 | 8.88 | pass |
| 2,048 / 128 | 1 | 72.24 | not remeasured | — | 13.39 | — | pass |
| 2,048 / 128 | 2 | 72.60 | not remeasured | — | 25.26 | — | pass |
| 2,048 / 128 | 4 | 72.63 | not remeasured | — | 45.69 | — | pass |

## Earlier long-context checkpoint

Prompt 2,048 / output 128; batch 1. This earlier independent comparison is shown separately from the 2026-09-26 matrix.

| Mode | TTFT P50 | TPOT P50 | Aggregate TPS P50 | Measured repetitions | Exact tokens |
| --- | --- | --- | --- | --- | --- |
| contiguous | 400.16 | 69.99 | 13.78 | 3 | pass |
| recompute | 418.93 | 443.58 | 2.26 | 3 | pass |

KV storage is 147,456 bytes per sequence token (36 layers, 8 KV heads, head dimension 128, BF16 K and V).

Sources are the individual result files: the shared `kv_cache_manifest.json` was overwritten by the final recomputation-only invocation and does not index all cached points. Notes: [KV cache](../docs/phase_notes/kv_cache.md).

## Saved sources

Paths are relative to the local `results/` directory. Raw JSON/JSONL stays local. The commit is the recorded parent revision; a dirty flag means the run included local changes.

| Result JSON | Captured (UTC) | Parent commit | Dirty |
| --- | --- | --- | --- |
| `kv_cache_20260926/contiguous_short_b1.json` | 2026-09-26 | `7967cdf` | true |
| `kv_cache_20260926/recompute_short_b1.json` | 2026-09-26 | `7967cdf` | true |
| `kv_cache_20260926/contiguous_short_b2.json` | 2026-09-26 | `7967cdf` | true |
| `kv_cache_20260926/recompute_short_b2.json` | 2026-09-26 | `7967cdf` | true |
| `kv_cache_20260926/contiguous_short_b4.json` | 2026-09-26 | `7967cdf` | true |
| `kv_cache_20260926/recompute_short_b4.json` | 2026-09-26 | `7967cdf` | true |
| `kv_cache_20260926/contiguous_medium_b1.json` | 2026-09-26 | `7967cdf` | true |
| `kv_cache_20260926/recompute_medium_b1.json` | 2026-09-26 | `7967cdf` | true |
| `kv_cache_20260926/contiguous_medium_b2.json` | 2026-09-26 | `7967cdf` | true |
| `kv_cache_20260926/recompute_medium_b2.json` | 2026-09-26 | `7967cdf` | true |
| `kv_cache_20260926/contiguous_medium_b4.json` | 2026-09-26 | `7967cdf` | true |
| `kv_cache_20260926/recompute_medium_b4.json` | 2026-09-26 | `7967cdf` | true |
| `kv_cache_20260926/contiguous_long_b1.json` | 2026-09-26 | `7967cdf` | true |
| `kv_cache_20260926/contiguous_long_b2.json` | 2026-09-26 | `7967cdf` | true |
| `kv_cache_20260926/contiguous_long_b4.json` | 2026-09-26 | `7967cdf` | true |
| `kv_cache_long_repeated/contiguous_long_b1.json` | 2026-08-29 | not recorded | not recorded |
| `kv_cache_long_repeated/recompute_long_b1.json` | 2026-08-29 | not recorded | not recorded |
