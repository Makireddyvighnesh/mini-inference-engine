# MiniLLM-L4 Phase 5 — Continuous batching

## Outcome

Phase 5 adds an iteration-level in-flight scheduler. New requests can be
prefilled while existing requests continue decoding, and completed rows are
removed from the active batch immediately. The Phase 4 static runner remains
available as the matched baseline.

The implementation is validated by the repository tests and a tiny real
`Qwen3ForCausalLM` CPU model. The full 4B L4 benchmark is intentionally not
claimed from this workspace because the current runtime cannot initialize a
CUDA device; the reproduction command below is ready for the L4 host.

## What was built

- `engine/scheduler.py`
  - `ContinuousBatchScheduler` with FIFO arrival admission.
  - Maximum active request count and maximum new-prefill-token budget.
  - Optional initial `max_wait_ms` batching window.
  - Controlled cancellation of requests that have not started.
- `benchmarks/runners/continuous_requests.py`
  - Explicit prefill and one-token decode calls.
  - Active-sequence state and lifecycle events.
  - Rebuilt decode batches after every iteration.
  - Immediate removal of finished rows with `DynamicCache.batch_select_indices`.
  - Left-padding and concatenation when new cache rows join an existing batch.
  - Greedy token correctness and per-request output limits.
- `benchmarks/commands/run_continuous_requests.py`
  - Reuses the Phase 4 workload materialization and Phase 1 reference corpus.
  - Writes raw results, event JSONL, scheduler summaries, and a checkpointed
    manifest.
- `configs/workloads/qwen3_fp8_continuous.yaml`
  - Pins the Qwen revision, arrival pattern, prefill budget, wait window,
    repetitions, and telemetry policy.

## Scheduling sequence

```text
pending arrivals
      |
      v
FIFO ready queue --cancel--> cancelled + released
      |
      | select within free slots and prefill-token budget
      v
prefill new prompts
      |
      v
decode every active row once
      |
      +--> finish/remove rows immediately
      |
      +--> next iteration admits more ready requests
```

The wait window is used only when no request is decoding. Once a live decode
batch exists, the scheduler never pauses it just to fill a larger prefill
batch. This prevents the batching policy from adding avoidable delay to active
traffic.

## Cache rebasing

Phase 5 does not yet implement paged physical storage. Each new prefill returns
a `DynamicCache`. When it joins an active batch, the runner:

1. finds the maximum physical cache length of the existing and incoming groups;
2. left-pads the shorter cache tensors along the sequence dimension;
3. concatenates key/value rows along the batch dimension;
4. constructs a new `DynamicCache` and keeps the same active-row order.

Each request retains its logical cached-token count. The decode attention mask
marks only the rightmost logical tokens for that row, and its position ID is
the logical cached-token count. This lets rows with different prompt lengths
share one physical sequence width while preserving their individual positions.

## Correctness gates

The tests cover:

- request and prefill-token budgets;
- the initial wait-window behavior;
- a request joining while another request is decoding;
- immediate removal of a shorter row from a longer decode batch;
- lifecycle/resource cleanup;
- exact outputs from a deterministic cache-updating fake model;
- exact token equality against independent manual generation using a tiny real
  Qwen3 model.

Validation in the current runtime:

```text
98 passed, 1 skipped
```

The tiny real-model test compared continuous outputs against independent
single-request manual generation and matched all token IDs. GPU validation is
still required on the NVIDIA L4 before reporting Phase 5 performance results.

## Reproduction command on the L4

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_continuous_requests \
  --config minillm_l4/configs/workloads/qwen3_fp8_continuous.yaml \
  --workload all \
  --max-batch-sizes 1 2 4 \
  --reference-dir minillm_l4/results/phase1/references_baseline \
  --output-dir minillm_l4/results/continuous_requests
```

The manifest records `maximum_active_batch_size`, decode iteration batch
sizes, prefill token budgets, queue delay, TTFT, ITL/TPOT, E2E latency, TPS,
GPU utilization, and memory telemetry. Compare its results against the Phase 4
manifest using the same workload, arrival interval, warm-up policy, and
repetitions.

The command supports both deterministic fixed-rate arrivals and seeded
Poisson-like arrivals. For the latter, keep the same request count and seed
while changing only the arrival policy, for example:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_continuous_requests \
  --config minillm_l4/configs/workloads/qwen3_fp8_continuous.yaml \
  --workload mixed \
  --arrival-pattern poisson \
  --arrival-rate-per-second 40 \
  --max-batch-sizes 2 4 \
  --output-dir minillm_l4/results/continuous_poisson
```

## Known limitations

- Cache rebasing is copy-based and can add overhead when new rows join.
- Physical KV blocks, fragmentation control, and paged attention belong to
  Phase 6.
- Prefill and decode still use the model's eager execution path.
- Cancellation is supported before execution, not mid-decode preemption.
- The optional wait window applies only at an idle-to-active transition.
- No Phase 5 L4 performance table is recorded until the CUDA-enabled run is
  completed and correctness passes against the pinned reference corpus.

## Entry conditions for Phase 6

- Preserve the continuous scheduler and correctness corpus.
- Replace copy-based cache rebasing with fixed-size logical/physical KV blocks.
- Add randomized allocation/free/aliasing tests before using indirection in the
  model execution path.
