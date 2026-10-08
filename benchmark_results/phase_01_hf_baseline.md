# Phase 1 — Hugging Face baseline

Saved full matrix: 2026-09-26, after the harness timing correction. One NVIDIA L4; pinned Qwen3-4B FP8 checkpoint; greedy exact-length output. One warm-up and three measured repetitions per point. Loading, tokenization, and warm-up are excluded. Latencies are milliseconds; TPS is aggregate completed output tokens/s unless explicitly labeled per request.

Four requests per run. All 9 points passed: 108/108 measured request outputs exactly matched the existing Phase 1 reference corpus.

| Prompt / output | Batch | TTFT P50 | TTFT P95 | TPOT P50 | E2E P50 | Aggregate TPS P50 | Peak allocated GiB | GPU P50 (%) | Exact tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 / 32 | 1 | 72.86 | 76.81 | 71.47 | 2,289.78 | 14.00 | 4.15 | 34 | pass |
| 128 / 32 | 2 | 74.46 | 76.97 | 71.86 | 2,303.43 | 27.81 | 4.18 | 34 | pass |
| 128 / 32 | 4 | 96.67 | 97.56 | 72.20 | 2,336.29 | 54.77 | 4.23 | 34 | pass |
| 512 / 64 | 1 | 97.21 | 97.85 | 71.24 | 4,585.40 | 13.97 | 4.25 | 34 | pass |
| 512 / 64 | 2 | 181.91 | 182.81 | 72.06 | 4,722.95 | 27.10 | 4.38 | 35 | pass |
| 512 / 64 | 4 | 391.39 | 392.40 | 71.83 | 4,918.12 | 52.04 | 4.56 | 38 | pass |
| 2,048 / 128 | 1 | 406.22 | 407.39 | 70.12 | 9,311.86 | 13.74 | 4.56 | 38 | pass |
| 2,048 / 128 | 2 | 918.17 | 921.19 | 70.03 | 9,814.02 | 26.08 | 4.99 | 43 | pass |
| 2,048 / 128 | 4 | 1,965.16 | 1,966.53 | 71.71 | 11,073.91 | 46.23 | 5.85 | 49 | pass |

Decode TPOT stays near 70–72 ms across these shapes. Notes: [HF baseline](../docs/phase_notes/hf_baseline.md).

## Saved sources

Paths are relative to the local `results/` directory. Raw JSON/JSONL stays local. The commit is the recorded parent revision; a dirty flag means the run included local changes.

| Result JSON | Captured (UTC) | Parent commit | Dirty |
| --- | --- | --- | --- |
| `phase1_20260926/baseline_short_b1.json` | 2026-09-26 | `251b6a2` | true |
| `phase1_20260926/baseline_short_b2.json` | 2026-09-26 | `251b6a2` | true |
| `phase1_20260926/baseline_short_b4.json` | 2026-09-26 | `251b6a2` | true |
| `phase1_20260926/baseline_medium_b1.json` | 2026-09-26 | `251b6a2` | true |
| `phase1_20260926/baseline_medium_b2.json` | 2026-09-26 | `251b6a2` | true |
| `phase1_20260926/baseline_medium_b4.json` | 2026-09-26 | `251b6a2` | true |
| `phase1_20260926/baseline_long_b1.json` | 2026-09-26 | `251b6a2` | true |
| `phase1_20260926/baseline_long_b2.json` | 2026-09-26 | `251b6a2` | true |
| `phase1_20260926/baseline_long_b4.json` | 2026-09-26 | `251b6a2` | true |
