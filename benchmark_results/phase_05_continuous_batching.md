# Phase 5 — Continuous batching

Saved corrected matrix: 2026-09-26. One NVIDIA L4; pinned Qwen3-4B FP8 checkpoint; greedy exact-length output. One warm-up and three measured repetitions per point. Loading, tokenization, and warm-up are excluded. Latencies are milliseconds; TPS is aggregate completed output tokens/s unless explicitly labeled per request.

Fixed-rate arrivals 25 ms apart; prefill budget 4,096 tokens; initial batch wait 2 ms. All six points passed exact-reference checks.

| Trace | Requests/run | Max batch | TTFT P50 | TTFT P95 | TPOT P50 | E2E P95 | Aggregate TPS P50 | Exact tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uniform | 4 | 1 | 3,391.95 | 6,719.42 | 69.97 | 8,889.66 | 14.28 | pass |
| uniform | 4 | 2 | 1,313.91 | 2,565.85 | 75.81 | 4,844.74 | 26.01 | pass |
| uniform | 4 | 4 | 149.34 | 187.19 | 73.90 | 2,477.45 | 51.08 | pass |
| mixed | 6 | 1 | 11,593.69 | 23,137.97 | 70.68 | 32,510.07 | 13.73 | pass |
| mixed | 6 | 2 | 4,164.66 | 13,518.85 | 81.50 | 23,221.78 | 19.20 | pass |
| mixed | 6 | 4 | 971.34 | 4,904.40 | 107.00 | 15,870.54 | 28.01 | pass |

The uniform and mixed traces have different populations from the three-request Phase 4 mixed trace; the two tables are not a matched speedup comparison. Capacity probes are recorded separately in [capacity stress](capacity_stress.md). Notes: [continuous batching](../docs/phase_notes/continuous_batching.md).

## Saved sources

Paths are relative to the local `results/` directory. Raw JSON/JSONL stays local. The commit is the recorded parent revision; a dirty flag means the run included local changes.

| Result JSON | Captured (UTC) | Parent commit | Dirty |
| --- | --- | --- | --- |
| `continuous_20260926_mod/continuous_uniform_b1.json` | 2026-09-26 | `ef0013b` | true |
| `continuous_20260926_mod/continuous_uniform_b2.json` | 2026-09-26 | `ef0013b` | true |
| `continuous_20260926_mod/continuous_uniform_b4.json` | 2026-09-26 | `ef0013b` | true |
| `continuous_20260926_mod/continuous_mixed_b1.json` | 2026-09-26 | `ef0013b` | true |
| `continuous_20260926_mod/continuous_mixed_b2.json` | 2026-09-26 | `ef0013b` | true |
| `continuous_20260926_final/continuous_mixed_b4.json` | 2026-09-26 | `ef0013b` | true |
