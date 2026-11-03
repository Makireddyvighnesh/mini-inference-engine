# Phase 8 — Chunked prefill measurements

Captured 2026-10-03T03:32:07.616551+00:00. Model `Qwen/Qwen3-4B-Instruct-2507-FP8` at `8591804019c8b22094c3b5b4454e0edc05dffc98`.

One warm-up/measurement configuration across every case: 1 warm-up runs and 3 measured runs. Same prompts, outputs, arrivals, FP8 path, decode policy, page pool, and active limit. Prefix reuse enabled: False. Loading and warm-up are excluded.

Chunk 0 is the unchunked control, which permits an oversized prompt only when decode is idle. Positive chunk sizes respect the per-iteration token budget. Cases run in the recorded order, with the control first. Raw JSON retains environment/Git identity, request events, all percentiles, repeat variance, and scheduler/chunk records. `source_snapshot/` and `run_provenance.json` preserve the actual runtime source, input dataset, reference corpora, resolved configuration, and file hashes.

| Workload | Chunk tokens | TTFT P50 / P95 / P99 (ms) | ITL P50 / P95 / P99 (ms) | TPOT P50 (ms) | E2E P95 (ms) | Output TPS P50 | Exact HF / control |
| --- | --- | --- | --- | --- | --- | --- | --- |
| uniform | 0 | 206.08 / 350.07 / 353.00 | 60.73 / 63.06 / 121.56 | 63.83 | 2,241.61 | 55.77 | pass / pass |
| uniform | 128 | 203.59 / 346.28 / 346.63 | 60.23 / 62.85 / 120.19 | 63.34 | 2,223.73 | 55.71 | pass / pass |
| uniform | 256 | 206.37 / 350.60 / 353.97 | 60.55 / 63.06 / 122.03 | 63.60 | 2,235.32 | 55.53 | pass / pass |
| mixed | 0 | 6,354.41 / 18,467.22 / 18,483.79 | 60.68 / 62.58 / 63.23 | 60.70 | 26,200.91 | 17.02 | pass / pass |
| mixed | 128 | 1,749.10 / 5,619.18 / 5,671.66 | 61.93 / 125.17 / 126.78 | 86.51 | 13,420.80 | 33.10 | pass / pass |
| mixed | 256 | 968.78 / 4,019.17 / 4,031.33 | 62.19 / 132.57 / 137.17 | 76.95 | 11,891.72 | 37.33 | pass / pass |

A failed correctness gate invalidates a performance claim; its measurements are retained for diagnosis. Three repetitions of this fixed corpus do not establish production tail latency.

## Capacity and telemetry

| Workload | Chunk tokens | Requests/s P50 | GPU utilization P50 (%) | Peak allocated / reserved GiB | Maximum chunk tokens | Chunks during decode |
| --- | --- | --- | --- | --- | --- | --- |
| uniform | 0 | 1.743 | 40.0 | 8.65 / 9.75 | 128 | 3 |
| uniform | 128 | 1.741 | 41.0 | 8.65 / 9.75 | 128 | 3 |
| uniform | 256 | 1.735 | 39.0 | 8.65 / 9.75 | 128 | 3 |
| mixed | 0 | 0.228 | 42.0 | 9.06 / 9.79 | 2048 | 1 |
| mixed | 128 | 0.443 | 44.0 | 8.94 / 9.86 | 128 | 41 |
| mixed | 256 | 0.500 | 44.0 | 8.96 / 9.86 | 256 | 21 |

Raw sources: `/home/ubuntu/STT/LLMPerfLab/minillm_l4/results/phase8_chunked_prefill_final_20261003`; configuration and case index: `chunked_prefill_manifest.json`.
