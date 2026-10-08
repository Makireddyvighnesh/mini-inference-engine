# Phase 8 — Chunked prefill measurements

Captured 2026-10-03T03:22:36.085583+00:00. Model `Qwen/Qwen3-4B-Instruct-2507-FP8` at `8591804019c8b22094c3b5b4454e0edc05dffc98`.

One warm-up/measurement configuration across every case: 1 warm-up runs and 1 measured runs. Same prompts, outputs, arrivals, FP8 path, decode policy, page pool, and active limit. Prefix reuse enabled: False. Loading and warm-up are excluded.

Chunk 0 is the unchunked control, which permits an oversized prompt only when decode is idle. Positive chunk sizes respect the per-iteration token budget. Cases run in the recorded order, with the control first. Raw JSON retains environment/Git identity, request events, all percentiles, repeat variance, and scheduler/chunk records. `source_snapshot/` and `run_provenance.json` preserve the actual runtime source, input dataset, reference corpora, resolved configuration, and file hashes.

| Workload | Chunk tokens | TTFT P50 / P95 / P99 (ms) | ITL P50 / P95 / P99 (ms) | TPOT P50 (ms) | E2E P95 (ms) | Output TPS P50 | Exact HF / control |
| --- | --- | --- | --- | --- | --- | --- | --- |
| uniform | 0 | 206.34 / 334.14 / 345.49 | 60.24 / 61.65 / 120.04 | 63.14 | 2,211.12 | 55.85 | pass / pass |
| uniform | 64 | 545.46 / 730.07 / 741.30 | 60.22 / 117.03 / 118.71 | 63.93 | 2,604.76 | 47.67 | FAIL / FAIL |
| uniform | 128 | 202.60 / 331.30 / 342.74 | 60.43 / 61.75 / 120.49 | 63.50 | 2,218.35 | 55.68 | pass / pass |
| uniform | 256 | 208.07 / 340.53 / 352.30 | 62.09 / 62.62 / 123.30 | 65.01 | 2,272.86 | 54.39 | pass / pass |
| mixed | 0 | 6,431.85 / 17,337.81 / 18,187.49 | 60.10 / 62.02 / 62.17 | 60.63 | 23,987.16 | 17.14 | pass / pass |
| mixed | 64 | 4,082.40 / 9,106.67 / 9,690.90 | 60.89 / 121.14 / 122.78 | 104.49 | 17,170.16 | 25.38 | FAIL / FAIL |
| mixed | 128 | 1,756.29 / 5,122.67 / 5,525.53 | 61.18 / 124.70 / 125.59 | 85.97 | 12,945.55 | 33.15 | pass / pass |
| mixed | 256 | 960.86 / 3,668.82 / 3,898.05 | 61.06 / 130.90 / 135.19 | 75.76 | 11,221.83 | 37.98 | pass / pass |

A failed correctness gate invalidates a performance claim; its measurements are retained for diagnosis. Three repetitions of this fixed corpus do not establish production tail latency.

## Capacity and telemetry

| Workload | Chunk tokens | Requests/s P50 | GPU utilization P50 (%) | Peak allocated / reserved GiB | Maximum chunk tokens | Chunks during decode |
| --- | --- | --- | --- | --- | --- | --- |
| uniform | 0 | 1.745 | — | — / — | 128 | 3 |
| uniform | 64 | 1.490 | — | — / — | 64 | 6 |
| uniform | 128 | 1.740 | — | — / — | 128 | 3 |
| uniform | 256 | 1.700 | — | — / — | 128 | 3 |
| mixed | 0 | 0.230 | — | — / — | 2048 | 1 |
| mixed | 64 | 0.340 | — | — / — | 64 | 82 |
| mixed | 128 | 0.444 | — | — / — | 128 | 41 |
| mixed | 256 | 0.509 | — | — / — | 256 | 21 |

Raw sources: `/home/ubuntu/STT/LLMPerfLab/minillm_l4/results/phase8_chunked_prefill_policy_fix_smoke_20261003`; configuration and case index: `chunked_prefill_manifest.json`.
