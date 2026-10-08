# Phase 4 — Concurrent requests and static batching

Corrected trace: 2026-09-26. One NVIDIA L4; pinned Qwen3-4B FP8 checkpoint; greedy exact-length output. One warm-up and three measured repetitions per point. Loading, tokenization, and warm-up are excluded. Latencies are milliseconds; TPS is aggregate completed output tokens/s unless explicitly labeled per request.

Three requests (128/32, 512/64, 2,048/128), arriving 25 ms apart. All three batch limits passed exact-reference checks.

| Max batch | TTFT P50 | TTFT P95 | TPOT P50 | E2E P95 | Aggregate TPS P50 | Requests/s P50 | Padding (%) | Exact tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2,319.43 | 7,105.72 | 70.02 | 16,004.27 | 13.96 | 0.187 | 0.0 | pass |
| 2 | 3,265.84 | 3,294.05 | 73.60 | 12,627.08 | 17.68 | 0.237 | 35.5 | pass |
| 4 | 3,301.10 | 3,332.17 | 74.11 | 12,717.51 | 17.54 | 0.235 | 35.5 | pass |

On this trace static batching increases throughput but raises median TTFT. The earlier median-TTFT improvement was a telemetry timing artifact.

## Earlier high-utilization stress

Twelve mixed requests arriving at time zero; one warm-up and three measured traces. The maximum batch limit 16 executes 12 requests together. These measurements predate the timing fix: about 25 ms is included in early TTFT and about 50 ms in run duration. Batch grouping is unaffected because every request arrives at time zero.

| Max batch | Aggregate TPS P50 | TTFT P50 | TTFT P95 | E2E P95 | TPOT P50 | Peak reserved GiB | GPU P50 (%) | Exact tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 22.81 | 15,348.11 | 28,502.29 | 39,266.33 | 84.15 | 6.26 | 98 | pass |
| 8 | 25.06 | 4,723.87 | 24,937.17 | 35,735.49 | 138.76 | 8.64 | 98 | pass |
| 16 | 27.91 | 7,044.40 | 7,095.99 | 32,106.87 | 194.48 | 10.79 | 99 | pass |

Notes: [concurrent requests](../docs/phase_notes/concurrent_requests.md).

## Saved sources

Paths are relative to the local `results/` directory. Raw JSON/JSONL stays local. The commit is the recorded parent revision; a dirty flag means the run included local changes.

| Result JSON | Captured (UTC) | Parent commit | Dirty |
| --- | --- | --- | --- |
| `concurrent_repeated_20260926/static_mixed_b1.json` | 2026-09-26 | `e77cbcf` | true |
| `concurrent_repeated_20260926/static_mixed_b2.json` | 2026-09-26 | `e77cbcf` | true |
| `concurrent_repeated_20260926/static_mixed_b4.json` | 2026-09-26 | `e77cbcf` | true |
| `stress_batch_final/static_mixed_b4.json` | 2026-09-07 | not recorded | not recorded |
| `stress_batch_final/static_mixed_b8.json` | 2026-09-07 | not recorded | not recorded |
| `stress_batch_final/static_mixed_b16.json` | 2026-09-07 | not recorded | not recorded |
