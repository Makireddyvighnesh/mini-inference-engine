# Phase 8 — Chunked prefill measurements

Captured 2026-10-03T03:19:12.279473+00:00. Model `Qwen/Qwen3-4B-Instruct-2507-FP8` at `8591804019c8b22094c3b5b4454e0edc05dffc98`.

One warm-up/measurement configuration across every case: 1 warm-up runs and 1 measured runs. Same prompts, outputs, arrivals, FP8 path, decode policy, page pool, and active limit. Prefix reuse enabled: False. Loading and warm-up are excluded.

Chunk 0 is the unchunked control, which permits an oversized prompt only when decode is idle. Positive chunk sizes respect the per-iteration token budget. Cases run in the recorded order, with the control first. Raw JSON retains environment/Git identity, request events, all percentiles, repeat variance, and scheduler/chunk records. `source_snapshot/` and `run_provenance.json` preserve the actual runtime source, input dataset, reference corpora, resolved configuration, and file hashes.

| Workload | Chunk tokens | TTFT P50 / P95 / P99 (ms) | ITL P50 / P95 / P99 (ms) | TPOT P50 (ms) | E2E P95 (ms) | Output TPS P50 | Exact HF / control |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mixed | 0 | 2,027.58 / 5,933.51 / 6,280.71 | 61.35 / 61.77 / 61.96 | 61.23 | 5,704.00 | 0.00 | FAIL / FAIL |
| mixed | 128 | 764.64 / 2,394.35 / 2,603.09 | 126.04 / 126.54 / 126.56 | 122.10 | — | 0.00 | FAIL / FAIL |
| mixed | 256 | 399.52 / 1,313.72 / 1,428.40 | 132.06 / 133.06 / 133.06 | 123.34 | — | 0.00 | FAIL / FAIL |

A failed correctness gate invalidates a performance claim; its measurements are retained for diagnosis. Three repetitions of this fixed corpus do not establish production tail latency.

## Capacity and telemetry

| Workload | Chunk tokens | Requests/s P50 | GPU utilization P50 (%) | Peak allocated / reserved GiB | Maximum chunk tokens | Chunks during decode |
| --- | --- | --- | --- | --- | --- | --- |
| mixed | 0 | 0.000 | — | — / — | 2048 | 0 |
| mixed | 128 | 0.000 | — | — / — | 128 | 21 |
| mixed | 256 | 0.000 | — | — / — | 256 | 11 |

Raw sources: `/home/ubuntu/STT/LLMPerfLab/minillm_l4/results/phase8_chunked_prefill_mixed_smoke_20261003`; configuration and case index: `chunked_prefill_manifest.json`.
