# Phase 2 — Manual decode

Saved full matrix: 2026-09-26. One NVIDIA L4; pinned Qwen3-4B FP8 checkpoint; greedy exact-length output. One warm-up and three measured repetitions per point. Loading, tokenization, and warm-up are excluded. Latencies are milliseconds; TPS is aggregate completed output tokens/s unless explicitly labeled per request.

Four requests per run; 9/9 points and 108/108 measured outputs passed exact-reference checks.

| Prompt / output | Batch | HF TTFT P50 | Manual TTFT P50 | HF TPOT P50 | Manual TPOT P50 | HF aggregate TPS | Manual aggregate TPS | TPS change | Exact tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 / 32 | 1 | 72.86 | 73.25 | 71.47 | 71.51 | 14.00 | 13.96 | -0.3% | pass |
| 128 / 32 | 2 | 74.46 | 74.45 | 71.86 | 72.09 | 27.81 | 27.70 | -0.4% | pass |
| 128 / 32 | 4 | 96.67 | 94.44 | 72.20 | 70.44 | 54.77 | 56.17 | +2.6% | pass |
| 512 / 64 | 1 | 97.21 | 95.41 | 71.24 | 71.16 | 13.97 | 13.98 | +0.1% | pass |
| 512 / 64 | 2 | 181.91 | 179.59 | 72.06 | 71.16 | 27.10 | 27.41 | +1.1% | pass |
| 512 / 64 | 4 | 391.39 | 386.27 | 71.83 | 71.51 | 52.04 | 52.32 | +0.5% | pass |
| 2,048 / 128 | 1 | 406.22 | 403.29 | 70.12 | 71.52 | 13.74 | 13.49 | -1.8% | pass |
| 2,048 / 128 | 2 | 918.17 | 898.65 | 70.03 | 72.36 | 26.08 | 25.37 | -2.7% | pass |
| 2,048 / 128 | 4 | 1,965.16 | 1,933.54 | 71.71 | 72.19 | 46.23 | 46.11 | -0.3% | pass |

Throughput differs by at most 2.7% from `model.generate()` on this matrix. Notes: [manual decode](../docs/phase_notes/manual_decode.md).

## Saved sources

Paths are relative to the local `results/` directory. Raw JSON/JSONL stays local. The commit is the recorded parent revision; a dirty flag means the run included local changes.

| Result JSON | Captured (UTC) | Parent commit | Dirty |
| --- | --- | --- | --- |
| `manual_decode_20260926/manual_short_b1.json` | 2026-09-26 | `07fe935` | true |
| `manual_decode_20260926/manual_short_b2.json` | 2026-09-26 | `07fe935` | true |
| `manual_decode_20260926/manual_short_b4.json` | 2026-09-26 | `07fe935` | true |
| `manual_decode_20260926/manual_medium_b1.json` | 2026-09-26 | `07fe935` | true |
| `manual_decode_20260926/manual_medium_b2.json` | 2026-09-26 | `07fe935` | true |
| `manual_decode_20260926/manual_medium_b4.json` | 2026-09-26 | `07fe935` | true |
| `manual_decode_20260926/manual_long_b1.json` | 2026-09-26 | `07fe935` | true |
| `manual_decode_20260926/manual_long_b2.json` | 2026-09-26 | `07fe935` | true |
| `manual_decode_20260926/manual_long_b4.json` | 2026-09-26 | `07fe935` | true |
