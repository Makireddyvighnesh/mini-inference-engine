# Phase 8 — Chunked prefill measurements

Captured 2026-10-05T17:06:38.327054+00:00. Model `Qwen/Qwen3-4B-Instruct-2507-FP8` at `8591804019c8b22094c3b5b4454e0edc05dffc98`.

One warm-up/measurement configuration across every case: 1 warm-up runs and 3 measured runs. Same prompts, outputs, arrivals, FP8 path, decode policy, page pool, and active limit. Prefix reuse enabled: False. Loading and warm-up are excluded.

Chunk 0 is the unchunked control, which permits an oversized prompt only when decode is idle. Positive chunk sizes respect the per-iteration token budget. Cases run in the recorded order, with the control first. Raw JSON retains environment/Git identity, request events, all percentiles, repeat variance, and scheduler/chunk records. `source_snapshot/` and `run_provenance.json` preserve the actual runtime source, input dataset, reference corpora, resolved configuration, and file hashes.

| Workload | Chunk tokens | TTFT P50 / P95 / P99 (ms) | ITL P50 / P95 / P99 (ms) | TPOT P50 (ms) | E2E P95 (ms) | Output TPS P50 | Exact HF / control |
| --- | --- | --- | --- | --- | --- | --- | --- |
| uniform | 0 | 210.31 / 357.08 / 357.24 | 61.52 / 65.39 / 122.61 | 64.54 | 2,273.48 | 54.50 | pass / pass |
| uniform | 64 | 558.78 / 765.27 / 765.70 | 61.64 / 119.82 / 121.19 | 65.43 | 2,679.11 | 46.47 | pass / pass |
| uniform | 128 | 210.74 / 358.08 / 358.12 | 61.75 / 63.72 / 123.33 | 64.82 | 2,278.12 | 54.46 | pass / pass |
| uniform | 256 | 209.64 / 355.46 / 355.60 | 61.38 / 63.03 / 122.34 | 64.40 | 2,262.31 | 54.77 | pass / pass |
| mixed | 0 | 6,539.21 / 19,076.21 / 19,086.62 | 62.57 / 63.99 / 64.85 | 62.81 | 27,046.35 | 16.49 | pass / pass |
| mixed | 64 | 4,164.53 / 10,083.87 / 10,095.68 | 64.00 / 124.09 / 124.72 | 107.25 | 18,116.43 | 24.56 | pass / pass |
| mixed | 128 | 1,800.00 / 5,787.38 / 5,787.95 | 63.78 / 127.87 / 129.09 | 88.79 | 13,859.88 | 32.04 | pass / pass |
| mixed | 256 | 1,006.45 / 4,148.49 / 4,154.43 | 63.71 / 135.57 / 149.87 | 79.05 | 12,209.14 | 36.35 | pass / pass |

A failed correctness gate invalidates a performance claim; its measurements are retained for diagnosis. Three repetitions of this fixed corpus do not establish production tail latency.

## Capacity and telemetry

| Workload | Chunk tokens | Requests/s P50 | GPU utilization P50 (%) | Peak allocated / reserved GiB | Maximum chunk tokens | Chunks during decode |
| --- | --- | --- | --- | --- | --- | --- |
| uniform | 0 | 1.703 | 39.0 | 8.65 / 9.75 | 128 | 3 |
| uniform | 64 | 1.452 | 40.0 | 8.65 / 9.76 | 64 | 6 |
| uniform | 128 | 1.702 | 39.0 | 8.65 / 9.76 | 128 | 3 |
| uniform | 256 | 1.711 | 40.0 | 8.65 / 9.76 | 128 | 3 |
| mixed | 0 | 0.221 | 41.0 | 9.06 / 9.79 | 2048 | 1 |
| mixed | 64 | 0.329 | 44.0 | 9.03 / 9.87 | 64 | 82 |
| mixed | 128 | 0.429 | 43.0 | 8.97 / 9.87 | 128 | 41 |
| mixed | 256 | 0.487 | 43.0 | 8.98 / 9.87 | 256 | 21 |

Raw sources: `/home/ubuntu/STT/LLMPerfLab/minillm_l4/results/phase8_aligned_prefill_ab_20261005/after`; configuration and case index: `chunked_prefill_manifest.json`.
