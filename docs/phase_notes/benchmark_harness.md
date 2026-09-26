# MiniLLM-L4 Phase 0 — Benchmark harness

Date completed: 2026-08-26

## Outcome

Phase 0 is complete. MiniLLM-L4 now has a standalone benchmark package with
deterministic request/workload schemas, fixed and mixed-length workload
generators, request-event recording, synchronized timing helpers, best-effort
CUDA/NVML sampling, percentile summaries, repeated-run support, and raw JSON
and JSONL artifact writers.

The implementation is intentionally runtime-agnostic. Phase 1 will replace
the deterministic CPU fixture runner with the pinned Hugging Face model
runner, while preserving the same workload and result schema.

## What was built

- `minillm_l4/benchmarks/core/schemas.py`
  - `RequestSpec`, `WorkloadSpec`, `RequestOutcome`, `EventRecord`, and
    `HarnessConfig`.
  - Prompt token IDs and SHA-256 digests are retained for reproducibility.
- `minillm_l4/benchmarks/core/workloads.py`
  - Short, medium, and long buckets: 128/32, 512/64, and 2048/128 prompt /
    output tokens.
  - Fixed-shape and deterministic mixed-length workloads.
  - Closed-loop, fixed-rate, and Poisson arrival schedules.
- `minillm_l4/benchmarks/core/harness.py`
  - Sequential measured repetitions with excluded warm-ups.
  - Canonical request events and request-level metric derivation.
  - Failure capture without discarding the rest of a run.
  - Machine-readable result and raw-event JSONL output.
- `minillm_l4/benchmarks/core/timing.py`
  - Synchronized wall timing, optional CUDA-event timing, and timer-overhead
    measurement.
- `minillm_l4/benchmarks/core/hardware.py`
  - CUDA allocator/peak-memory snapshots and best-effort `nvidia-smi`
    utilization/VRAM telemetry.
  - Python, PyTorch, package, platform, and Git identity metadata.
- `minillm_l4/benchmarks/runners/simulated.py`
  - Deterministic CPU fixture runner for harness validation only.
- `minillm_l4/benchmarks/commands/run_harness.py`
  - One-command Phase 0 CLI.
- `minillm_l4/configs/workloads/minillm_l4_harness.yaml`
  - Pinned Phase 0 fixture configuration, seed 17, one warm-up, and three
    measured repetitions.
- `minillm_l4/docs/metric_definitions.md`
  - Shared definitions for TTFT, ITL, TPOT, E2E latency, throughput, memory,
    and percentile calculation.

## Before / after checkpoint

| Capability | Before Phase 0 | After Phase 0 |
|---|---|---|
| Request schema | Existing experiment-specific inputs | Typed deterministic `RequestSpec` |
| Workload shapes | Existing synthetic datasets, no MiniLLM-L4 contract | Short, medium, long, and mixed fixtures |
| Request timing | Model-specific phase timings | Canonical arrival/admission/prefill/token/completion events |
| Repetitions | Separate script conventions | Warm-up exclusion and repeated-run summaries |
| Raw evidence | Existing LLMPerfLab result formats | Phase 0 JSON result plus streamable events JSONL |
| GPU/VRAM | Existing low-level utilities | Harness-integrated allocator and best-effort NVML snapshots |
| Percentiles | Existing utilities | Shared P50/P95/P99 definition and raw observations |

## Reproduction command

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_harness \
  --config minillm_l4/configs/workloads/minillm_l4_harness.yaml \
  --output-dir minillm_l4/results/phase0
```

The command writes:

- `harness_manifest.json`
- `workload_{short,medium,long,mixed}.json`
- `harness_{short,medium,long,mixed}.json`
- `harness_{short,medium,long,mixed}_events.jsonl`

## Canonical fixture result

The command used one warm-up and three measured repetitions. The values below
are for the deterministic CPU fixture runner; they validate the harness and
are not neural-network performance claims.

| Workload | Prompt/output shape | Requests/run | TTFT P50 (ms) | TPOT P50 (ms) | Aggregate TPS P50 | Run duration P50 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| short | 128 / 32 | 4 | 0.370 | 0.143 | 2,986.6 | 42.859 |
| medium | 512 / 64 | 4 | 0.322 | 0.137 | 4,304.7 | 59.470 |
| long | 2048 / 128 | 4 | 0.326 | 0.126 | 5,693.7 | 89.924 |
| mixed | mixed | 12 | 0.333 | 0.137 | 5,937.5 | 161.685 |

Timer-pair overhead in the mixed workload result was 52.2 ns mean, 50.0 ns
median, and 60.0 ns at P95 over 1,000 samples.

## Validation

```text
65 passed, 1 skipped in 4.89s
```

The one skipped test is the pre-existing CUDA timer test because PyTorch in the
current execution namespace cannot initialize CUDA/NVML. The harness records
this state explicitly: CUDA allocator memory and GPU utilization are marked
unavailable rather than reported as zero. `nvidia-smi` is queried on a
best-effort basis, and the same command will populate hardware fields when the
runner has a working CUDA/NVML namespace.

## 2026-09-26 corrections

Two harness defects were fixed after Phase 0 was first closed.

**Telemetry inside the timed window.** `run_started_ns` was taken before the
synchronous `before` GPU snapshot, and the run duration was taken after the
`after` snapshot and the sampler join. Each snapshot shells out to
`nvidia-smi`, which costs about 25 ms on the L4 (median of 20 calls). The
original Phase 0 run could not see the GPU, so the cost was hidden then; on the
L4 it added about 50 ms to every run. Because scheduled arrivals are offsets
from `run_started_ns`, arrival-scheduled runs (Phase 4 concurrent, Phase 5
continuous, and capacity stress) also charged the `before` snapshot to the
queue delay, TTFT, and E2E latency of the first requests. Both snapshots now
sit outside the timed window. `test_blocking_gpu_snapshots_are_outside_the_timed_run`
reproduces the defect with a 60 ms snapshot (120.7 ms run duration before the
fix).

**Git provenance.** `git rev-parse HEAD` ran in the caller's working
directory. Commands run from the LLMPerfLab root, which is not a repository,
so every result recorded `git_commit_sha: null`. The lookup now targets the
`minillm_l4` repository and also records `git_worktree_dirty`.

The background sampler itself is not a measurable source of error: a
launch-bound 252-kernel GPU loop had the same P50 and P95 step latency with the
250 ms sampler on and off (differences below 0.5%, within noise).

Fixture rerun on the L4 (`results/phase0_20260926` before the timing fix,
`results/phase0_20260926_fixed` after; one warm-up, three repetitions each):

| Workload | TTFT P50 before → after (ms) | TPOT P50 before → after (ms) | Run duration P50 before → after (ms) | Aggregate TPS P50 before → after |
|---|---:|---:|---:|---:|
| short | 0.365 → 0.345 | 0.137 → 0.129 | 67.5 → 18.7 | 1,895 → 6,834 |
| medium | 0.381 → 0.344 | 0.142 → 0.134 | 85.5 → 36.9 | 2,993 → 6,942 |
| long | 0.334 → 0.340 | 0.144 → 0.134 | 123.9 → 71.1 | 4,133 → 7,197 |
| mixed | 0.338 → 0.329 | 0.139 → 0.133 | 187.2 → 131.7 | 5,127 → 7,291 |

Per-token latencies are unchanged; run-level throughput was under-reported by
1.4–3.6× on this millisecond-scale fixture. For multi-second model runs the
same ~50 ms is a 0.2–2% throughput error, but the TTFT effect on
arrival-scheduled runs is up to ~25 ms per early request. Results produced
before this date by Phases 1–5 are re-measured in their own notes.

## Known limitations

- The Phase 0 fixture runner is not a model benchmark.
- The harness currently executes requests sequentially; concurrency and
  continuous batching belong to later phases.
- GPU utilization is `nvidia-smi`'s sampled "kernel active" percentage, not a
  profiler trace or SM occupancy. It cannot distinguish a busy GPU from one
  running many tiny launch-bound kernels.

## Entry conditions for Phase 1

- Preserve the Phase 0 result schema and metric definitions.
- Add the pinned model ID and exact revision to the Hugging Face baseline
  configuration.
- Implement a Hugging Face tokenizer/model loader and greedy baseline.
- Reuse the Phase 0 harness to measure sequential generation.
- Add token-ID correctness checks before comparing performance.
