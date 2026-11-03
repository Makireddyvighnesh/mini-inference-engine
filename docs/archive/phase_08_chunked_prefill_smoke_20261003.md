# Phase 8 — Chunked prefill measurements

Captured 2026-10-03T03:15:54.949900+00:00. Model `Qwen/Qwen3-4B-Instruct-2507-FP8` at `8591804019c8b22094c3b5b4454e0edc05dffc98`.

One warm-up/measurement configuration across every case: 1 warm-up runs and 1 measured runs. Same prompts, outputs, arrivals, FP8 path, decode policy, page pool, and active limit. Prefix reuse enabled: False. Loading and warm-up are excluded.

Chunk 0 is the unchunked control, which permits an oversized prompt only when decode is idle. Positive chunk sizes respect the per-iteration token budget. Cases run in the recorded order, with the control first. Raw JSON retains environment/Git identity, request events, all percentiles, repeat variance, and scheduler/chunk records. `source_snapshot/` and `run_provenance.json` preserve the actual runtime source, input dataset, reference corpora, resolved configuration, and file hashes.

| Workload | Chunk tokens | TTFT P50 / P95 / P99 (ms) | ITL P50 / P95 / P99 (ms) | TPOT P50 (ms) | E2E P95 (ms) | Output TPS P50 | Exact HF / control |
| --- | --- | --- | --- | --- | --- | --- | --- |
| uniform | 0 | 211.17 / 345.13 / 357.05 | 62.42 / 62.70 / 124.42 | 65.41 | 2,288.90 | 54.01 | pass / pass |
| uniform | 64 | 569.86 / 764.36 / 776.24 | 62.00 / 121.80 / 124.12 | 66.05 | 2,697.66 | 46.07 | FAIL / FAIL |
| uniform | 128 | 213.91 / 349.40 / 361.47 | 62.96 / 63.32 / 126.19 | 65.92 | 2,306.72 | 53.61 | pass / pass |
| uniform | 256 | 214.20 / 350.11 / 362.19 | 62.64 / 63.08 / 125.78 | 65.59 | 2,298.24 | 53.80 | pass / pass |

A failed correctness gate invalidates a performance claim; its measurements are retained for diagnosis. Three repetitions of this fixed corpus do not establish production tail latency.

## Capacity and telemetry

| Workload | Chunk tokens | Requests/s P50 | GPU utilization P50 (%) | Peak allocated / reserved GiB | Maximum chunk tokens | Chunks during decode |
| --- | --- | --- | --- | --- | --- | --- |
| uniform | 0 | 1.688 | — | — / — | 128 | 3 |
| uniform | 64 | 1.440 | — | — / — | 64 | 6 |
| uniform | 128 | 1.675 | — | — / — | 128 | 3 |
| uniform | 256 | 1.681 | — | — / — | 128 | 3 |

Raw sources: `/home/ubuntu/STT/LLMPerfLab/minillm_l4/results/phase8_chunked_prefill_smoke_20261003`; configuration and case index: `chunked_prefill_manifest.json`.
