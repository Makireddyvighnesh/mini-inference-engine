# Concurrent request lifecycle and static batching

## Outcome

MiniLLM-L4 now accepts an entire staggered request trace, moves every request
through an explicit lifecycle, and dispatches arrived work through a FIFO
padded static-batch scheduler. Uniform and mixed prompt/output lengths are
supported. Each batch owns an isolated contiguous KV cache and releases it in
a `finally` block.

This phase intentionally does not admit requests during an active model call.
That limitation makes head-of-line blocking measurable and gives continuous
batching a clear static baseline.

## What was built

- `engine/request.py`: request state machine, cancellation, and resource state.
- `engine/scheduler.py`: deterministic pending/ready FIFO queues and admission.
- `benchmarks/core/harness.py`: trace-level execution with scheduled arrivals.
- `engine/generation/manual.py`: per-row output limits and decode position IDs.
- `benchmarks/runners/concurrent_requests.py`: padded mixed-shape execution,
  lifecycle events, KV cleanup, padding accounting, and reference validation.
- `benchmarks/commands/run_concurrent_requests.py`: reproducible uniform/mixed
  workloads and maximum-batch-size sweeps.
- `configs/workloads/qwen3_fp8_concurrent.yaml`: pinned model, workload,
  scheduler, repetition, and telemetry settings.

## Architecture

```text
scheduled requests
       |
       v
 pending arrival queue --cancel--> cancelled + resources released
       |
       | arrival time reached
       v
   FIFO ready queue
       |
       | select up to max_batch_size
       v
 waiting -> prefill -> decoding -> finished
                         |
                         v
                shared batch KV cache
                         |
                         v
                       release
```

A selected batch is left-padded to its maximum prompt length and decoded to
its maximum requested output length. The attention mask hides prompt padding.
Position IDs are derived from that mask, preserving each row's logical token
positions. Per-row output limits stop token events and completion latency at
the requested length even if longer rows continue decoding.

Padding waste is:

```text
(prompt padding slots + output padding slots)
------------------------------------------------
(all padded prompt slots + all padded output slots)
```

## Correctness and stress gates

Tests cover legal/illegal state transitions, deterministic FIFO admission,
mixed lengths, position IDs, output limits, cancellation, cleanup, and padding
accounting. A 60-request trace runs twice with mixed shapes and cancellations;
every request reaches a terminal state without starvation.

```text
31 passed in 3.52s
```

The real Qwen validation swept uniform and mixed workloads at maximum batch
sizes 1, 2, and 4. All six configurations exactly matched the Phase 1 token
references. The repeated mixed benchmark below also passed every reference
check across three measured traces.

## Repeated benchmark configuration

- Model: `Qwen/Qwen3-4B-Instruct-2507-FP8`
- Revision: `8591804019c8b22094c3b5b4454e0edc05dffc98`
- Hardware: one NVIDIA L4, 24 GB
- Workload: one short, one medium, and one long request
- Shapes: 128/32, 512/64, and 2,048/128 prompt/output tokens
- Arrival pattern: fixed rate, 25 ms between arrivals
- Maximum static batch sizes: 1, 2, and 4
- Sampling: greedy argmax; EOS disabled
- Warm-up: one trace per configuration
- Measurement: three traces per configuration
- GPU and CUDA allocator telemetry: enabled

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_concurrent_requests \
  --config minillm_l4/configs/workloads/qwen3_fp8_concurrent.yaml \
  --workload mixed \
  --max-batch-sizes 1 2 4 \
  --count-per-bucket 1 \
  --repetitions 3 \
  --warmup-repetitions 1 \
  --reference-dir minillm_l4/results/phase1/references_baseline \
  --output-dir minillm_l4/results/concurrent_repeated
```

## Results

Request percentiles combine observations across the three measured traces.
TPS and requests/sec are medians across those traces.

| Max batch | TTFT P50 | TTFT P95 | TPOT P50 | E2E P95 | TPS | Requests/s | Padding | Peak VRAM | GPU P50 | Correctness |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2,419.034 ms | 7,333.210 ms | 72.159 ms | 16,498.458 ms | 13.533 | 0.181 | 0.0% | 4.56 GiB | 36% | pass |
| 2 | 217.028 ms | 5,342.650 ms | 75.360 ms | 14,461.441 ms | 15.429 | 0.207 | 35.5% | 5.00 GiB | 39% | pass |
| 4 | 217.640 ms | 5,288.678 ms | 74.605 ms | 14,328.497 ms | 15.583 | 0.209 | 12.5% | 4.56 GiB | 39% | pass |

Increasing the batch limit reduced queueing and raised aggregate throughput.
It did not improve per-token latency: padded mixed batches perform more work
per decode step. Batch size 1 has no padding but severe head-of-line delay.

Padding differs because arrivals are staggered and dispatch is immediate.
Only requests available at a scheduling boundary are grouped; arrivals during
an active batch wait for the next boundary.

## High-utilization batch stress run

The larger stress run used the same pinned Qwen3-4B FP8 model and a mixed trace
of 12 requests: four short (128/32), four medium (512/64), and four long
(2,048/128) prompt/output shapes. All requests arrived at time zero so the
GPU had work continuously available. Maximum static batch sizes 4, 8, and 16
were measured with one warm-up trace and three measured traces. GPU telemetry
was enabled and system telemetry was disabled to reduce host-side sampling
overhead.

| Max batch | Tokens/sec P50 | TTFT P50/P95 | E2E P50/P95 | TPOT P50 | GPU P50 | Peak VRAM | Correctness |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 22.81 | 15.35/28.50 s | 20.65/39.27 s | 84.2 ms | 98% | 6.3 GiB | pass |
| 8 | 25.06 | 4.72/24.94 s | 18.03/35.74 s | 138.8 ms | 98% | 8.6 GiB | pass |
| 16 | 27.91 | 7.04/7.10 s | 19.30/32.11 s | 194.5 ms | 99% | 10.8 GiB | pass |

The maximum batch-16 configuration contained 12 requests, so its actual model
batch was 12; the scheduler limit was 16. Batch 16 improved aggregate token
throughput by 22.4% versus batch 4 and reduced E2E P95 by 18.2%. TPOT grew
from 84.2 ms to 194.5 ms because the static mixed-shape batch performs each
decode step for the longest active row. This is the throughput/latency tradeoff
that continuous batching is intended to improve.

The biggest P50/P95 separation appeared at batch 8: TTFT P50 was 4.72 seconds
while TTFT P95 was 24.94 seconds. This is queueing from static batches: some
requests wait behind earlier batches even though all work arrived together.

Artifacts:

- `results/stress_batch_final/concurrent_manifest.json`
- `results/figures/concurrency_stress_final.png`
- `results/figures/concurrency_stress_final.csv`

The stress chart includes token throughput, request throughput, TTFT, E2E
latency, TPOT, GPU utilization, peak reserved VRAM, and padding waste. The
runner checkpoints the manifest after every completed batch-size configuration.

## Known limitations

- A static batch runs to completion; new requests cannot join it.
- A shared batch cache cannot be released row-by-row, although each request's
  completion is recorded at its own output limit.
- Cancellation is supported before execution, not as mid-decode preemption.
- Padding wastes prompt and decode compute for mixed shapes.
- Physical cache storage remains Transformers `DynamicCache`.
- Host overhead crossing an arrival timestamp can alter batch composition;
  the pure scheduler is deterministic for a given admission time.

## Entry conditions for continuous batching

The next phase can reuse the lifecycle, trace harness, event schema, and
arrival queue. It must schedule inside each decode iteration, admit new work
while other requests decode, and remove finished rows immediately.
