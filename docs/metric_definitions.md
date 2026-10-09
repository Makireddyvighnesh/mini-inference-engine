# MiniLLM-L4 benchmark metric definitions

Phase 0 records timestamps with `time.perf_counter_ns()` and stores each event
relative to the start of its measured run. Warm-up runs are executed but are
excluded from every reported distribution.

## Request metrics

- **Queue delay:** `admission - arrival`.
- **Prefill time:** `prefill_end - prefill_start`.
- **TTFT:** `first token ready - arrival`.
- **ITL:** each interval between adjacent `token_ready` events.
- **TPOT:** `(last token ready - first token ready) / (generated tokens - 1)`.
  It is the average post-first-token inter-token interval.
- **Decode time:** `completion - first token ready`.
- **E2E latency:** `completion - arrival`.
- **Tokens/sec:** generated tokens divided by decode time.
- **E2E tokens/sec:** generated tokens divided by E2E latency.

Missing events produce a JSON `null` value and are excluded from the related
distribution rather than being treated as zero.

## Run metrics

- **Requests/sec:** completed requests divided by measured run duration.
- **Aggregate tokens/sec:** completed output tokens divided by measured run
  duration.
- **Peak VRAM:** maximum CUDA allocator value observed by the sampler. The
  result preserves allocated and reserved values separately.
- **GPU utilization:** best-effort `nvidia-smi` utilization samples. It is
  marked unavailable when the driver or command is not visible.

## Percentiles

P50, P95, and P99 use a linearly interpolated percentile over the sorted
observations, defined by `benchmarks/core/metrics.py::percentile`. For `n`
observations and percentile `p` (0–100), the zero-based position is
`(n - 1) * p / 100`; interpolate between the observations at its floor and
ceiling. For example, P95 of `[10, 30]` is `29`, not `30`. Showcase summaries
pool non-null request observations across measured runs, excluding warm-up.
The raw observations remain in the request event and repetition
records so summaries can be independently recomputed.

## Case validity

Showcase, fused-kernel, CUDA Graph, and vLLM comparison cases record `completed`
and `expected` request counts, `valid`, and `invalid_reason`. A count mismatch
or nonzero `alloc_retries` (where tracked) makes the case invalid. Expected
counts come from the workload and measured repetition count. Failed requests
remain in the raw records, including missing timings and partial outputs.
Commands save invalid cases, print `INVALID` with the reason, and exit nonzero
after all cases finish. The vLLM report flags invalid cases and excludes them
from ratios and output agreement. CUDA Graph cases that require replay also
reject missing measured graph replays.

## Timing boundaries

The harness uses synchronized wall-clock boundaries for user-visible request
metrics. GPU model runners may additionally use CUDA events for device-only
timings through `benchmarks.core.timing.measure_call`; CUDA event time and wall
time must be reported separately. Model loading, CUDA graph capture,
compilation, and warm-up are not part of steady-state request distributions.

## Component trace diagnostics

Instrumented runners store an `execution_trace` inside each batch diagnostic
when diagnostic tracing is enabled with `--trace-summary`. Each component has:

- `wall_ms_total`: elapsed host-visible time across its spans;
- `device_ms_total`: CUDA event time for GPU spans, or `null` for CPU spans;
- `host_overhead_ms_total`: wall time not covered by the CUDA event;
- `share_of_recorded_span_wall_percent`: share of measured component spans;
- `count`, mean, minimum, maximum, and the raw individual spans.

`trace_window_ms` is the first-to-last recorded span window. A nonzero
`unattributed_wall_ms` indicates time between spans that the runner has not
assigned to a component. `runner_diagnostics[].batch_runner_wall_ms` includes
the complete callable invocation, including any uninstrumented runner work.
Synchronized tracing intentionally adds overhead, so trace runs are for
diagnosis; the untraced asynchronous run is the performance headline.
