# Phase 7 — Prefix caching

Saved paired measurements: 2026-09-29. One NVIDIA L4, pinned Qwen3-4B FP8 model, block size 16, greedy output.

## Sequential cold/warm pairs

Eight output tokens; one warm-up pair and three measured pairs; 256 blocks; SDPA suffix prefill. Every cold and warm output exactly matches the Phase 1 corpus.

| Prompt tokens | Reused tokens | Cold TTFT P50 (ms) | Warm TTFT P50 (ms) | TTFT speedup | Cold E2E P50 (ms) | Warm E2E P50 (ms) | Exact tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 112 | 62.34 | 66.03 | 0.94× | 478.67 | 482.25 | pass |
| 512 | 496 | 101.49 | 65.34 | 1.55× | 509.74 | 474.40 | pass |
| 2048 | 2032 | 492.79 | 65.84 | 7.49× | 917.22 | 492.41 | pass |

## Continuous trace

Eight requests 20 ms apart; 512-token prefix with a 16/32-token suffix; 4/8 output tokens; max decode batch 4; 512 blocks; `sm89` projections; eager dynamic decode. One warm-up and three measured traces per mode. Values use uncached → cached order. Cache counters are identical across measured repetitions.

| Target reuse | Hit rate | Reused tokens | Computed prefill tokens | Peak cached KV blocks | TTFT P50 (ms) | TTFT P95 (ms) | Aggregate TPS P50 | Cached = uncached | Dense reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0% | 0.0% | 0 | 4288 | 271 | 645.23 → 655.20 | 1,297.45 → 1,311.82 | 27.77 → 27.46 | pass | fail: prefix-002 |
| 50% | 37.5% | 1536 | 2752 | 175 | 651.54 → 502.53 | 1,311.03 → 1,151.55 | 27.49 → 30.34 | pass | pass |
| 100% | 87.5% | 3584 | 704 | 47 | 653.94 → 481.23 | 1,310.35 → 962.18 | 27.36 → 34.39 | pass | pass |

At 100% target reuse, seven of eight requests hit; the first is cold. Cached and uncached outputs match at every reuse rate. **The strict dense-reference gate fails at 0% reuse on `prefix-002`**, where the historical dense BF16 reference has an exact logit tie. The saved command therefore exits nonzero for that trace; paired cache correctness must not be reported as a full dense-reference pass. These are 2026-09-29 results, before the final Phase 6 numerical-policy validation, and have not been rerun here. No L4 evictions occurred.

Notes: [prefix caching](../docs/phase_notes/prefix_cache.md). Continuous cache counters and gate status: `results/prefix_trace_20260929/manifest.json`.

Sequential sources: `results/prefix_sequential_20260929/short_b16_o8.json`, `results/prefix_sequential_20260929/medium_b16_o8.json`, `results/prefix_sequential_20260929/long_b16_o8.json`.

## Saved sources

Paths are relative to the local `results/` directory. Raw JSON/JSONL stays local. The commit is the recorded parent revision; a dirty flag means the run included local changes.

| Result JSON | Captured (UTC) | Parent commit | Dirty |
| --- | --- | --- | --- |
| `prefix_trace_20260929/reuse_000_uncached.json` | 2026-09-29 | `16ef083` | true |
| `prefix_trace_20260929/reuse_000_cached.json` | 2026-09-29 | `16ef083` | true |
| `prefix_trace_20260929/reuse_050_uncached.json` | 2026-09-29 | `16ef083` | true |
| `prefix_trace_20260929/reuse_050_cached.json` | 2026-09-29 | `16ef083` | true |
| `prefix_trace_20260929/reuse_100_uncached.json` | 2026-09-29 | `16ef083` | true |
| `prefix_trace_20260929/reuse_100_cached.json` | 2026-09-29 | `16ef083` | true |
