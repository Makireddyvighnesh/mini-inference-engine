# Continuous-batching capacity stress

## Scope

This stress pass measures the Phase 5 continuous scheduler on the pinned
Qwen3-4B FP8 model and one NVIDIA L4. It separates three questions:

1. How many requests can remain in the active decode batch?
2. How many new prompt tokens can be admitted in one prefill operation?
3. Can new prefills enter while existing requests are decoding?

The runner still uses a dense, left-padded `DynamicCache` with explicit cache
rebasing. These results do not claim Phase 6 paged scheduling or vLLM parity.
For a mixed prompt batch, the scheduler budget counts real prompt tokens, while
the model receives `batch_size × max_prompt_length` rectangular compute slots.

## Instrumentation

`ContinuousRequestTraceRunner.last_summary` now records:

- `maximum_concurrent_requests`: peak active decode rows;
- `maximum_prefill_input_tokens`: largest sum of prompt tokens admitted in one
  prefill call;
- `maximum_prefill_compute_slots` and `maximum_prefill_padding_slots`, which
  expose the rectangular work performed by dense Phase 5 prefill;
- `maximum_active_prompt_tokens` and `maximum_active_cached_tokens`;
- `prefill_batches_while_decoding` and
  `requests_prefilled_while_decoding`;
- queue depth, padding slots, decode batch sizes, and lifecycle records.

The dedicated matrix command is
`benchmarks/commands/run_capacity_stress.py`. Correctness matches each stress
prompt by digest against the Phase 1 corpus and checks the generated output
against the exact reference prefix. This allows the stress cases to use new
request IDs and shorter output limits without weakening token equality.

## Commands

Normal matrix:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_capacity_stress \
  --config minillm_l4/configs/workloads/qwen3_fp8_stress.yaml \
  --scenario all \
  --output-dir minillm_l4/results/capacity_stress
```

Larger active-batch probe:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_capacity_stress \
  --config minillm_l4/configs/workloads/qwen3_fp8_capacity_limit_128.yaml \
  --scenario concurrency \
  --output-dir minillm_l4/results/capacity_limit_128
```

Large prefill probe:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_capacity_stress \
  --config minillm_l4/configs/workloads/qwen3_fp8_capacity_limit_128.yaml \
  --scenario input_budget \
  --output-dir minillm_l4/results/capacity_limit_input
```

All commands save a checkpointed manifest, workload JSON, and one raw result
per case. Capacity probes used one measured repetition and no warm-up to keep
the boundary search practical; they are not final steady-state performance
tables.

## Results

Every completed case below passed exact greedy token correctness.

### Active request capacity

The normal short-prompt burst reached every configured active limit from 1 to
16. The extended probe then reached 24, 32, 48, 64, 96, and 128 active rows.

| Active limit | Observed active rows | Prompt tokens in first batch | TTFT P50 (ms) | Run TPS P50 |
|---:|---:|---:|---:|---:|
| 1 | 1 | 128 | 20,294.7 | 8.52 |
| 2 | 2 | 256 | 4,218.8 | 26.94 |
| 4 | 4 | 512 | 1,896.9 | 52.80 |
| 8 | 8 | 1,024 | 755.8 | 97.27 |
| 16 | 16 | 2,048 | 285.0 | 147.69 |
| 24 | 24 | 3,072 | 12,588.0 | 24.43 |
| 32 | 32 | 4,096 | 1,169.8 | 174.74 |
| 48 | 48 | 6,144 | 1,375.8 | 26.29 |
| 64 | 64 | 8,192 | 934.5 | 201.79 |
| 96 | 96 | 12,288 | 8,517.3 | 22.21 |
| 128 | 128 | 16,384 | 1,904.2 | 122.13 |

The 128-row short-prompt case used approximately 8.1 GiB peak allocated and
9.1 GiB peak reserved PyTorch memory. No CUDA out-of-memory condition occurred.
Therefore the current measured active capacity is **128 requests**, not a
hardware maximum. The practical limit should be selected from TTFT/P95 and
service-level objectives, not memory alone.

The run-level TPS values are intentionally not monotonic: this is a capacity
probe with burst arrivals, varying queue depth, and cache-rebase boundaries,
not a matched steady-state throughput sweep.

### New-prefill token budget

With eight 2,048-token prompts, the scheduler admitted the following maximum
new input totals in one prefill operation:

| Budget | Observed prefill tokens | Active rows | Prefill batches while decoding | TTFT P50 (ms) |
|---:|---:|---:|---:|---:|
| 2,048 | 2,048 | 8 | 7 | 12,964.6 |
| 4,096 | 4,096 | 8 | 3 | 1,959.7 |
| 8,192 | 8,192 | 8 | 1 | 1,974.3 |
| 16,384 | 16,384 | 8 | 0 | 1,971.9 |

The extended 16-request probe passed at **32,768 prompt tokens in one
prefill**. It used approximately 13.1 GiB peak allocated and 14.1 GiB peak
reserved memory. A separate 16,384-token multi-admission case reached about
18.1 GiB allocated and 18.9 GiB reserved because repeated DynamicCache
rebasing creates temporary copies. This is why the conservative Phase 5
operating budget should remain below the largest one-shot result until paged
physical storage replaces cache rebasing.

### Prefill while decode is active

The mixed short/medium/long arrival workload used 5 ms inter-arrivals and
passed exact correctness:

| Active limit | Peak active rows | Prefill batches admitted during decode | Requests admitted during decode |
|---:|---:|---:|---:|
| 4 | 4 | 1 | 1 |
| 8 | 8 | 3 | 7 |

The 2,048-token budget case produced seven such admissions. This confirms the
important continuous-batching behavior: the scheduler does not wait for the
active decode batch to finish before starting a new prompt prefill. It admits
new work into free rows, then rebuilds the decode batch.

## Final interpretation

- **Measured active capacity:** 128 concurrent active requests for the tested
  short-prompt/4-token probe, with exact correctness and no OOM.
- **Measured one-shot prefill capacity:** 32,768 prompt tokens across 16
  long requests, with exact correctness and no OOM.
- **Recommended Phase 5 guardrail:** use a lower production budget, such as
  16,384 prompt tokens, while tracking reserved memory because DynamicCache
  rebasing has workload-dependent temporary peaks.
- **New-prefill behavior:** verified while decoding, including mixed prompt
  lengths and FIFO admission under both request and token budgets.
- **Next architectural improvement:** move this continuous scheduler from
  dense cache rebasing to the Phase 6 paged allocator so admission does not
  copy the entire active cache.
