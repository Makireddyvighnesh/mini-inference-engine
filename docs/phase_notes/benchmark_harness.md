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

## Known limitations

- The Phase 0 fixture runner is not a model benchmark.
- The harness currently executes requests sequentially; concurrency and
  continuous batching belong to later phases.
- GPU utilization is coarse `nvidia-smi` telemetry, not a profiler trace.
- The current workspace has no valid Git repository metadata, so result files
  record the commit SHA as unavailable. Phase 1 results should be run from a
  tracked checkout or include an explicit source revision.

## Entry conditions for Phase 1

- Preserve the Phase 0 result schema and metric definitions.
- Add the pinned model ID and exact revision to the Hugging Face baseline
  configuration.
- Implement a Hugging Face tokenizer/model loader and greedy baseline.
- Reuse the Phase 0 harness to measure sequential generation.
- Add token-ID correctness checks before comparing performance.
