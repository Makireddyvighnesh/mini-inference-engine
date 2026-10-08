# Phase 8 — Chunked prefill measurements

Captured 2026-10-04T16:10:26.539263+00:00. Model `Qwen/Qwen3-4B-Instruct-2507-FP8` at `8591804019c8b22094c3b5b4454e0edc05dffc98`.

One warm-up/measurement configuration across every case: 1 warm-up runs and 3 measured runs. Same prompts, outputs, arrivals, FP8 path, decode policy, page pool, and active limit. Prefix reuse enabled: False. Loading and warm-up are excluded.

This focused Phase 8 confirmation completed successfully on one NVIDIA L4. The six cases use continuous paged batching with four active slots, a 256-token prefill budget, `sm89` FP8 projections, compatible eager decode, and a 32,768-token page pool. Requests arrive 25 ms apart. The uniform trace has four requests, each with 128 prompt tokens and 32 output tokens. The mixed trace has six requests: two each with 128/32, 512/64, and 2048/128 prompt/output tokens. The extended prompt-length and generation-length sweep remains deferred; its saved results are separate from this checkpoint.

Chunk 0 is the unchunked control, which permits an oversized prompt only when decode is idle. Positive chunk sizes respect the per-iteration token budget. Cases run in the recorded order, with the control first. Raw JSON retains environment/Git identity, request events, all percentiles, repeat variance, and scheduler/chunk records. `source_snapshot/` and `run_provenance.json` preserve the actual runtime source, input dataset, reference corpora, resolved configuration, and file hashes.

| Workload | Chunk tokens | TTFT P50 / P95 / P99 (ms) | ITL P50 / P95 / P99 (ms) | TPOT P50 (ms) | E2E P95 (ms) | Output TPS P50 | Exact HF / control |
| --- | --- | --- | --- | --- | --- | --- | --- |
| uniform | 0 | 215.24 / 367.19 / 367.47 | 62.95 / 65.04 / 125.99 | 66.14 | 2,322.68 | 53.47 | pass / pass |
| uniform | 128 | 209.82 / 364.17 / 364.76 | 62.67 / 64.68 / 125.01 | 65.64 | 2,305.00 | 54.05 | pass / pass |
| uniform | 256 | 217.42 / 372.04 / 373.18 | 63.74 / 66.14 / 127.42 | 66.91 | 2,357.59 | 52.64 | pass / pass |
| mixed | 0 | 6,646.30 / 19,192.56 / 19,269.02 | 63.04 / 64.98 / 65.97 | 63.37 | 27,189.19 | 16.41 | pass / pass |
| mixed | 128 | 1,824.75 / 5,874.63 / 5,877.20 | 64.18 / 130.30 / 130.97 | 89.93 | 13,980.00 | 31.77 | pass / pass |
| mixed | 256 | 1,006.33 / 4,133.96 / 4,138.44 | 64.28 / 136.20 / 139.22 | 79.29 | 12,305.24 | 36.05 | pass / pass |

A failed correctness gate invalidates a performance claim; its measurements are retained for diagnosis. Three repetitions of this fixed corpus do not establish production tail latency.

## Capacity and telemetry

| Workload | Chunk tokens | Requests/s P50 | GPU utilization P50 (%) | Peak allocated / reserved GiB | Maximum chunk tokens | Chunks during decode |
| --- | --- | --- | --- | --- | --- | --- |
| uniform | 0 | 1.671 | 39.0 | 8.65 / 9.74 | 128 | 3 |
| uniform | 128 | 1.689 | 39.0 | 8.65 / 9.74 | 128 | 3 |
| uniform | 256 | 1.645 | 38.0 | 8.65 / 9.74 | 128 | 3 |
| mixed | 0 | 0.220 | 40.0 | 9.06 / 9.79 | 2048 | 1 |
| mixed | 128 | 0.425 | 43.0 | 8.94 / 9.85 | 128 | 41 |
| mixed | 256 | 0.483 | 42.0 | 8.95 / 9.85 | 256 | 21 |

Raw sources: `/home/ubuntu/STT/LLMPerfLab/minillm_l4/results/phase8_focused_20261004`; configuration and case index: `chunked_prefill_manifest.json`.

## Validation and interpretation

All **6/6 cases** passed, with **90/90 measured request outputs** exactly matching the existing Hugging Face corpus and their whole-prompt controls. The process exited with status 0 and released the GPU. An independent check of the saved records confirmed JSON/JSONL agreement, event-derived latency percentiles and aggregate throughput, contiguous prompt coverage, no generated token before the final prefill chunk, decode-first ordering, chunk/budget limits, the four-request active limit, and zero outstanding request pages after every run. All 96 source/input snapshot hashes match, and the trusted reference files remain unchanged. The check is saved as `independent_validation.json` alongside the raw results.

On the mixed trace, 256-token chunks increase throughput by about 120% and reduce median TTFT by about 85%. P95 end-to-end latency falls from 27.19 to 12.31 seconds. P95 token gaps rise from 64.98 to 136.20 ms because prompt chunks execute between decode steps. Median token gaps stay near 64 ms. Short prompts already fit in one chunk; their throughput stays close to the control.

These results confirm the saved 128-/256-token Phase 8 presets on this corpus. The previously failing 64-token setting remains experimental. Longer prompts and 1048-token generation are outside this focused confirmation. The [2026-10-03 measurements](phase_08_chunked_prefill_20261003.md) retain the earlier checkpoint and exploratory failures.
