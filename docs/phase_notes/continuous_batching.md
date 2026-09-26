# MiniLLM-L4 Phase 5 — Continuous batching

## Outcome

Phase 5 adds an iteration-level in-flight scheduler. New requests can be
prefilled while existing requests continue decoding, and completed rows are
removed from the active batch immediately. The Phase 4 static runner remains
available as the matched baseline.

The implementation is validated by the repository tests, a tiny real
`Qwen3ForCausalLM` CPU model, and a real Qwen3-4B FP8 L4 capacity sweep. The
capacity methodology and raw-result locations are recorded in
[`capacity_stress.md`](capacity_stress.md).

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
63 passed, 4 skipped
```

The tiny real-model test compared continuous outputs against independent
single-request manual generation and matched all token IDs. The L4 capacity
stress cases also matched exact reference token prefixes.

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
GPU utilization, and memory telemetry. The dedicated capacity matrix also
records peak active requests, total input tokens admitted in one prefill, and
prefills admitted while decode is active.

Run the capacity matrix with:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_capacity_stress \
  --config minillm_l4/configs/workloads/qwen3_fp8_stress.yaml \
  --scenario all \
  --output-dir minillm_l4/results/capacity_stress
```

Compare its results against the Phase 4 manifest using the same workload,
arrival interval, warm-up policy, and repetitions.

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

## 2026-09-26 L4 benchmark and host-sync cleanup

**Change.** The prefill and decode loops copied each row's next token to the
host with its own `.item()` (one device sync per row per step), and the decode
attention mask and position IDs were written into device tensors row by row
(two tiny kernel launches per row per step). Tokens are now copied with one
`.tolist()` per step, and the metadata is built on the host and moved once.
The outputs are unchanged.

**Measurement.** The reproduction command above was run on the L4 for HEAD
(`ef0013b`, clean worktree, `results/continuous_20260926_base`) and for the
modified runner (`results/continuous_20260926_mod`; mixed batch 4 re-run with
the final code in `results/continuous_20260926_final`). Each point uses one
warm-up and three measured traces; every point passed the exact-token gate.

| Trace | Max batch | TTFT P50 (ms) HEAD → new | TPOT P50 (ms) HEAD → new | E2E P95 (ms) HEAD → new | TPS P50 HEAD → new |
|---|---:|---:|---:|---:|---:|
| uniform | 1 | 3,465.1 → 3,391.9 | 71.27 → 69.97 | 9,047 → 8,890 | 14.06 → 14.28 |
| uniform | 2 | 1,351.9 → 1,313.9 | 78.11 → 75.81 | 5,006 → 4,845 | 25.21 → 26.01 |
| uniform | 4 | 155.9 → 149.3 | 76.59 → 73.90 | 2,568 → 2,477 | 49.31 → 51.08 |
| mixed | 1 | 11,740.4 → 11,593.7 | 71.42 → 70.68 | 32,730 → 32,510 | 13.66 → 13.73 |
| mixed | 2 | 4,064.8 → 4,164.7 | 78.87 → 81.50 | 22,568 → 23,222 | 19.74 → 19.20 |
| mixed | 4 | 983.1 → 971.3 | 107.30 → 107.00 | 15,817 → 15,871 | 28.10 → 28.01 |

The uniform trace improves 1.6–3.6% in TPS at every batch size; the mixed
trace moves within ±3%, which is the run-to-run noise seen when re-running
Phases 1–3. The change is kept because it removes host work from a host-bound
loop at no correctness risk, not because of a demonstrated speedup.

Relative to Phase 4 static batching, continuous admission is what matters on
the uniform trace: at batch 4, TTFT P50 is 149 ms and TPS 51.1, because new
requests join the running batch instead of waiting for it to drain.

**Tried and not adopted: trimming stale left padding.** Rows are left-aligned
to the longest sequence ever merged, so after that sequence finishes the
remaining rows keep attending over masked padding columns. A trim that drops
columns which are padding for every remaining row was implemented, passed the
tiny-Qwen reference test and the L4 exact-token gate, but fired once in all
the mixed traces (31 columns): in this workload the long-prompt requests also
generate the most tokens, so the longest row almost always finishes last. It
was removed to keep a numerically sensitive path out of the runner; it may pay
off for traffic where long prompts have short outputs.

## Known limitations

- Cache rebasing is copy-based and can add overhead when new rows join.
- Physical KV blocks, fragmentation control, and paged attention belong to
  Phase 6.
- Prefill and decode still use the model's eager execution path.
- Cancellation is supported before execution, not mid-decode preemption.
- The optional wait window applies only at an idle-to-active transition.
- The continuous runner still uses dense `DynamicCache` rebasing; the L4
  capacity results therefore include temporary cache-copy overhead and are
  not a paged-attention capacity claim.

## Entry conditions for Phase 6

- Preserve the continuous scheduler and correctness corpus.
- Replace copy-based cache rebasing with fixed-size logical/physical KV blocks.
- Add randomized allocation/free/aliasing tests before using indirection in the
  model execution path.
