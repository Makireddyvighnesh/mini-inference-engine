# Phase 0 — Benchmark harness

Latest corrected capture: 2026-09-26. These are deterministic CPU fixture measurements, not model inference performance. One warm-up and three measured repetitions.

| Workload | Requests/run | TTFT P50 (ms) | TPOT P50 (ms) | Aggregate TPS P50 | Run duration P50 (ms) |
| --- | --- | --- | --- | --- | --- |
| short | 4 | 0.345 | 0.129 | 6,833.8 | 18.73 |
| medium | 4 | 0.344 | 0.134 | 6,941.8 | 36.88 |
| long | 4 | 0.340 | 0.134 | 7,197.2 | 71.14 |
| mixed | 12 | 0.329 | 0.133 | 7,290.7 | 131.67 |

The corrected harness excludes blocking GPU snapshots from the timed run.

## Initial fixture archive

The original fixture validates the first harness implementation; its execution environment differs from the later run.

| Workload | TTFT P50 (ms) | TPOT P50 (ms) | Aggregate TPS P50 | Run duration P50 (ms) |
| --- | --- | --- | --- | --- |
| short | 0.352 | 0.140 | 1,889.6 | 67.74 |
| medium | 0.338 | 0.140 | 2,969.2 | 86.22 |
| long | 0.337 | 0.134 | 4,263.0 | 120.10 |
| mixed | 0.334 | 0.136 | 5,179.4 | 185.35 |

Notes: [benchmark harness](../docs/phase_notes/benchmark_harness.md).

## Saved sources

Paths are relative to the local `results/` directory. Raw JSON/JSONL stays local. The commit is the recorded parent revision; a dirty flag means the run included local changes.

| Result JSON | Captured (UTC) | Parent commit | Dirty |
| --- | --- | --- | --- |
| `phase0_20260926_fixed/harness_short.json` | 2026-09-26 | `a5b2aca` | true |
| `phase0_20260926_fixed/harness_medium.json` | 2026-09-26 | `a5b2aca` | true |
| `phase0_20260926_fixed/harness_long.json` | 2026-09-26 | `a5b2aca` | true |
| `phase0_20260926_fixed/harness_mixed.json` | 2026-09-26 | `a5b2aca` | true |
| `phase0/harness_short.json` | 2026-08-27 | not recorded | not recorded |
| `phase0/harness_medium.json` | 2026-08-27 | not recorded | not recorded |
| `phase0/harness_long.json` | 2026-08-27 | not recorded | not recorded |
| `phase0/harness_mixed.json` | 2026-08-27 | not recorded | not recorded |
