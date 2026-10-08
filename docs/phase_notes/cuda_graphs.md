# CUDA Graph decode in continuous and mixed serving

## Design

`ChunkedPrefillPagedRunner(..., cuda_graphs=True)` graphs decode-only iterations.
The default buckets are `(1, 2, 4, 8, 16, 32)`; `graph_batch_sizes` accepts a
nonempty, strictly increasing sequence of positive integers. The smallest
bucket covering the real rows runs one token per row. Changing request IDs,
membership, positions, or physical pages changes buffer contents only.

Capture is **lazy per bucket**. The first use allocates fixed-address input IDs
and position IDs `[bucket, 1]`, sequence lengths `[bucket]`, and int32 block
tables `[bucket, ceil(max_sequence_length / block_size)]`. The maximum sequence
length is the largest `prompt_tokens + max_new_tokens - 1` in this invocation,
bounded by the request KV pool capacity. A new workload capacity discards old
graphs; repeated invocations at the same capacity reuse them. The table width
does not grow at page boundaries or shrink when a request leaves.

Three inference-mode forwards on a side stream warm the model and Triton
specializations before capture. Capture records a Qwen3ForCausalLM forward with
paged kwargs, `use_cache=False`, `logits_to_keep=1`, and argmax. One replay and
synchronization after capture pay lazy CUDA loading costs. Warmup, capture, and
this initial replay use an all-scratch batch and never write real request KV.
`capture_ms` includes buffer creation, warmups, capture, and that first replay;
the subsequent allocator snapshot is outside this timer. The first live decode
step includes all lazy-capture work in its wall time and request metrics.

Before each live replay, host code reads each owner's old length and calls
`cache.reserve_append(owners, 1)`, exactly as eager decode does. Host packing
copies the last generated tokens, old lengths as positions, new lengths, and
current allocator block IDs into the static buffers. Argmax stays in the graph.
Only real outputs are cloned and copied to the host, once per step. The clone
keeps `state.next_token` stable when a later replay overwrites graph output
storage or the scheduler switches to eager decode. Optional `record_logit_gaps`
captures top-k as well and adds its usual extra host transfer outside capture.

## Padding and memory

Padding rows use token 0, position 0, sequence length 1, and **distinct scratch
pages**. Their only live block-table entry is column zero; unused columns are
`-1`. Packing resets all rows on every step, including rows that were real on
the previous replay. Every padding row writes only offset zero of its own
page. There are no concurrent writes to a shared padding slot and no aliases
with request pages.

When graph execution is supported, the runner adds `max(graph_batch_sizes)`
physical pages to the KV pool and reserves them under `__cuda_graph_scratch__`.
The caller's `num_blocks` remains the request capacity: scratch pages do not
consume pages needed for admission, and the single-request capacity check uses
the original count. The scratch owner ID is reserved and rejected as a request
ID. Scratch pages persist until `close()`; request pages are released normally
at completion/failure. `close()` drops graphs and releases scratch pages.

Each bucket has an independent private CUDA graph pool. Pools are not shared
because buckets may replay in arbitrary order while their outputs are live.
Memory reporting uses allocator snapshot `segment_pool_id` matched against
`graph.pool()`, summing reserved segment bytes; global allocation deltas would
also charge unrelated autotuning allocations. Static input buffers and scratch
KV live outside these private pools and are reported separately. Scratch KV
bytes are `pages * layers * kv_heads * block_size * head_dim * dtype_bytes * 2`.
Peak allocated/reserved case memory includes the model, request KV, all graph
pools, scratch KV, and warmup/transient allocations.

## Mixed steps and fallbacks

With `mixed_batch=True`, decode-only steps use graphs. Steps containing prompt
chunks retain the current single eager mixed forward by default. Set
`graph_mixed_decode=True` to replay decode and run prompt chunks in a separate
eager mixed-model forward with no decode rows. Planning budgets still count
decode rows and the same prompt candidates; the planner observes the combined
iteration time. `graph_mixed_decode_order` supports `decode_first` (default,
sends decode tokens before prompt work) and `prefill_first`.

Eager fallback reasons are explicit:

- CPU/non-CUDA device or unavailable CUDA;
- prefix reuse enabled (shared-page ownership is not graph compatible);
- a decode batch exceeds the largest bucket (later smaller batches can graph);
- Torch decode backend/unavailable Triton or storage block size outside 1..128;
- `decode_sdpa_compat=False`: its split/tile policy and split span depend on the
  current active batch and maximum length. A fixed graph could change reduction
  ordering. This implementation graphs the scheduler's default compatible
  policy: one KV split, 128-token tiles, and GQA reuse disabled;
- non-default RoPE (dynamic variants can inspect positions on the host).

Unsupported mixed/prefix combinations remain the scheduler's existing option
error. `graph_mixed_decode=True` requires both `cuda_graphs=True` and
`mixed_batch=True`. Boolean flags, bucket options, and split ordering are
validated. The older inherited `decode_mode="graph"` alone still reports an
eager fallback; `cuda_graphs` controls the new serving path. CUDA capture errors
fail the run and preserve failure diagnostics; they are not silently hidden by
an eager retry after KV reservation.

Run summaries contain actual `decode_mode_used` (`graph`, `eager`, or `mixed`),
`graph_replays`, `eager_decode_steps`, `graph_replay_share`, `captured_buckets`,
`graph_captures_this_run`, `capture_ms`, `graph_pool_memory_bytes` and per-bucket
bytes, static-buffer and scratch bytes, and fallback reasons/counts. Counters
reset each invocation; graph metadata persists with the captured graph. The
post-capture scratch replay is excluded from live replay counts. A one-output
prefill-only run reports zero decode steps, zero replays, and no captures.

## Exactness and capture-safety audit

Eager decode still calls `ContinuousPrefixPagedRunner._decode` with its original
model arguments and kernels. Eager mixed execution retains its row order,
reservation order, model call, and readback. The new attention flag defaults to
false, so existing eager and static-batch graph callers keep their prior checks.
Additional scheduler counters/timestamps do not change model arithmetic.

Graph decode uses the same projections, rotary computation, paged KV writer,
SDPA-compatible Triton attention, final norm, batched lm_head, and argmax as
eager rectangular decode. Each attention program reads only its own table row;
length/position bounds prevent reading unused entries or other requests' pages.
Host reservation and packing prove current private-page ownership; live device
assertions check lengths, positions, and physical IDs before layer-zero writes.

| Decode-path site | Host sync or request-dependent operation | Graph treatment |
| --- | --- | --- |
| Runner `_decode` | Owner lookups, length/max calculation, append reservation, construction of tables | All outside capture; copy current owner data every step. No captured allocator mutation. |
| Qwen3 model forward | Default position/cache inference and causal-mask preparation, including tensor-value decisions in HF mask helpers | Explicit device positions, no dense cache, and `attention_mask={"full_attention": None}` bypass mask preparation. Paged attention does not consume that mask. |
| Qwen3 rotary decorator | Dynamic/long RoPE variants take tensor maxima and Python branches | Only default RoPE is eligible; its tensor rotary computation remains captured. |
| `PagedQwen3Attention.forward` | Normalize/check sequence IDs and choose packed/mixed/rectangular paths | Fixed unique synthetic row IDs, no packed/mixed metadata in capture. The graph path is always rectangular one-token decode. |
| `_paged_attend`, shared-page guard | Tensor position `.item()`, owner lookup, `assert_writable_range` | Prefix reuse falls back. Graph-only flag rejects shared blocks, so the original shared-page branch is unreachable. Runner also checks no shared blocks before packing. |
| `_paged_attend`, layer-zero validation | Allocator lookup for every owner's current reserved length | Graph flag supplies `reserved_lengths=None`; avoids synthetic-ID lookup and stale capture-time reservation limits. Host code already reserves and packs the actual current length. |
| `validate_decode_metadata` | Eager `.detach().cpu().tolist()` on lengths, positions, and tables | Allowed only during warmup. During capture the existing capture-aware branch records asynchronous device assertions on live buffers. Graphs additionally check the fixed maximum read length. No per-owner limits are frozen. |
| `_paged_attend`, slow writer | Per-owner loop and position `.item()` before `write_layer_segment` | Graph flag requires the Triton writer and raises if the contract is unavailable, preventing entry into this branch. |
| `_triton_query_positions` / `_normalize_positions` | Host normalization can copy tensor positions to CPU and inspect allocator owner lengths | Device tensor positions take the CUDA view path. No owner normalization/lookup occurs. |
| `paged_attention` | Torch fallback walks owner tables; metadata validation can rebuild reserved lengths | Graph forces Triton. Fast writer validates once before layer zero and passes `_validate_metadata=False` to attention/writer, avoiding repeated owner lookup/validation. |
| Triton KV writer | Shape/stride checks, selection of position buffer | All fixed shape/stride metadata; the kernel loads positions/lengths/block IDs on-device. |
| Triton attention | Split/tile selection from host arguments; kernels traverse dynamic sequence lengths | Fixed split=1, tile=128, GQA reuse=false; logical loop bounds and page addresses load current buffers on every replay. |
| SM89 FP8 projections | Device-capability dispatch; autotuning can compile, time kernels, and synchronize | Host capability checks are fixed, independent of requests. Side-stream warmups populate all specializations for this bucket before capture. The model loader also pre-tunes SM89. |
| Argmax / optional gaps | `.tolist()` synchronizes token/gap readback | Argmax/top-k are captured; real-row readbacks and token/event bookkeeping are outside capture. |
| Admission, planner, release | ID-based scheduling, `mem_get_info`, eviction/release/zeroing | All outside capture. Fresh physical block IDs are copied before any subsequent replay. |

**Row-count rounding still needs L4 validation.** Padding changes the `M`
dimension from real decode rows to the bucket size for Q/K/V/O and MLP FP8
linears, and for the **batched decode lm_head matmul**. Norms are per row. The
SM89 autotune key is already the power-of-two `M_BUCKET`, but custom buckets
can select different specializations. Split mixed execution changes projection
row count from `decode_rows + prompt_tokens` to bucket-sized decode and
prompt-only forwards. Prefill lm_head still uses one-row calls. Rotary uses a
batched per-row frequency matmul, whose batch dimension also grows with padding.

Existing L4 observations show row-independent FP8/norm behavior. Matmul shape
and dispatch can nevertheless change rounding, especially in the unquantized
batched lm_head, and a near tie can change a token. CPU tests do not establish
that GPU numerical property. The gate compares every token against eager
whole-prompt scheduling with real arrival traffic, odd lengths, padding, page
crossings, and both split orders. No measured GPU equivalence or speedup is
claimed here until the reviewer runs it.

## Reviewer commands and checks

Run Python from the parent directory, using the workspace environment. Use a
fresh output directory; the benchmark refuses nonempty destinations.

```bash
cd /home/ubuntu/STT/LLMPerfLab
.conda-env/bin/python -m pytest -q minillm_l4/tests
.conda-env/bin/python -m pytest -q minillm_l4/tests/test_serving_cuda_graphs.py minillm_l4/tests/test_decode_metadata.py
MINILLM_RUN_MODEL_TESTS=1 .conda-env/bin/python -m pytest -q minillm_l4/tests
.conda-env/bin/python minillm_l4/scripts/check_policy_equivalence.py
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_cuda_graphs --output-dir minillm_l4/results/cuda_graphs_20261008_review
.conda-env/bin/python minillm_l4/scripts/make_cuda_graph_charts.py minillm_l4/results/cuda_graphs_20261008_review/cuda_graphs.json minillm_l4/docs/assets/cuda_graphs
```

The CUDA-only scheduler tests compare tiny-model eager/graph outputs, page
boundary crossings, shrinking/non-power-of-two membership, graph reuse across
repetitions, new workload capacities/IDs, split orders, and cleanup. The existing
decode metadata tests include live graph position validation and isolated
invalid-replay assertions. The pinned-model tests and policy gate verify the
real FP8 path. Required gate variants are `whole_graph`, `mixed_512_graph`,
`adaptive_graph`, and `adaptive_graph_split`; an additional
`adaptive_graph_split_prefill_first` checks the other order. Graph variants must
actually replay graphs, and all outcomes must complete with the requested
token count before comparisons can pass.

Inspect the benchmark's `alloc_retries` (nonzero means memory-pressure timing),
peak memory, completion counts, actual modes, measured capture counts, replay
share, and each bucket's capture time/private-pool bytes. Capture may occur in
a measured serving run if arrival timing creates a bucket absent from its one
warmup; those costs remain in metrics and are explicitly reported. Do not
interpret such a run as entirely steady state. Confirm every decode case
graphs its expected bucket, prefill cases have zero replays, and split serving
has a replay share of 1 when all batches fit. Compare padded batch shapes and
both split orders token for token before accepting throughput results.

## Benchmark outputs and definitions

`run_cuda_graphs` imports showcase `metrics`, `paged`, and `prompt` helpers. It
runs one warmup plus three measured repetitions on one loaded pinned model,
without an HF reference check. Every case calls `close()`, deletes the runner,
collects garbage, and empties the CUDA cache before the next case. No retained
bound `close` method keeps a runner/graph pool alive. Raw case JSON includes
environment/git provenance, request events/tokens, warmup and measured scheduler
summaries, headline metrics, and overhead. `cuda_graphs.json` is the aggregate.

- Decode: identical 512-token prompts, 128 outputs, batches 1/2/4/8/16/32,
  plus batch 8 at 4096 tokens; eager versus graphs in this serving runner.
- Prefill: one request, 128/256/512/1024/2048/4096/8192 tokens, one output,
  eager versus graphs enabled. Prefill remains eager; there is no graph capture
  or decode replay in either case.
- Serving: showcase's shuffled seed-17 16-request trace (128/256/512/1024 x4,
  every 150 ms, 128 outputs), and six alternating 128/6144-token requests every
  400 ms with 64 outputs. Whole-prompt continuous eager/graph and mixed+adaptive
  eager/graph/graph+split run the same requests.
- Overhead: capture time and private-pool memory per bucket per case, plus
  scratch/static-buffer memory. `--sections overhead` runs short capture probes
  for all six default buckets even without the decode section.

TPOT is the median of per-request mean post-first-token intervals. TTFT p50/p95
uses the showcase helper. Decode tok/s counts **real** post-first tokens divided
by summed decode-only step wall seconds, including packing, readback, and host
bookkeeping; it is omitted for mixed steps whose prompt work cannot be separated.
E2E/output tok/s is median per-run completed output tokens divided by run wall
time, including prompt work, scheduled waits, and cleanup. Worst pause is median
over runs of each run's maximum inter-token gap. GPU busy is median of each run's
sampled utilization p50. Replay share counts measured decode steps, rather than
padding tokens or scratch warmup replays.

`make_cuda_graph_charts.py` reuses showcase themes/style helpers and writes
light/dark SVGs: decode TPOT versus batch size, serving TPOT/output-throughput
bars for each workload/policy/mode, and prefill TTFT. Batch-8 long-prompt decode
points are marked separately. Prefill chart labels state that no decode replay
occurs; it should show comparable prefill timing honestly.

## Changed files and eager safety

| File | Purpose / eager-mode impact |
| --- | --- |
| `benchmarks/runners/chunked_prefill.py` | Bucket capture/replay, mixed split choices, fallbacks, counters/memory, cleanup. Default eager decode delegates to its original parent; unsplit mixed model math/order remain unchanged. |
| `benchmarks/runners/decode_graph.py` | CPU-testable bucket validation/selection and static packing; independent scratch rows; graph pool accounting. Used only by graph execution. |
| **`engine/kv_cache/qwen3_paged.py`** | **Engine change:** opt-in `paged_decode_graph` flag removes graph-only allocator-ID/reservation lookups and requires capture-safe writer/attention contracts. Default false leaves eager validation, ownership checks, writers, and kernel math unchanged. |
| `scripts/check_policy_equivalence.py` | Required graph variants, both split orders, complete-output and actual-replay checks. |
| `benchmarks/commands/run_cuda_graphs.py` | Matched one-session workload matrix, raw/aggregate JSON, telemetry, runner cleanup. |
| `scripts/make_cuda_graph_charts.py` | Light/dark comparison SVGs using showcase style. |
| `tests/test_serving_cuda_graphs.py` | CPU bucket/packing/option/fallback/split/cleanup tests and CUDA-only scheduler exactness tests. |
| `tests/test_cuda_graph_benchmark.py` | Metric definitions, benchmark case matrix/export/cleanup, CPU refusal, and SVG smoke checks. |
| `docs/phase_notes/cuda_graphs.md` | Design, safety audit, commands, reporting definitions, and reviewer results placeholders. |
| `CLAUDE.md` | Updated serving architecture, graph configuration, and shape-sensitive exactness guidance. |

No changes to `engine/model_runner/*`, `engine/kernels/*`,
`engine/generation/chunked_prefill.py`, or other `engine/kv_cache/*` files. No
commits or pushes. The static-batch graph runner and showcase helpers are
unchanged.

## Results

Measured 2026-10-08 on the L4 (1 warm-up + 3 measured runs per case, medians; no HF
reference check). Decode, prefill, and overhead ran in one process
(`results/cuda_graphs_20261008_core/`); serving ran in a second process after the
planner fix below (`results/cuda_graphs_20261008_serving/`); both are merged in
`results/cuda_graphs_20261008/cuda_graphs.json`. Every case completed with zero
allocator retries.

| Check | Result |
| --- | --- |
| Full suite with pinned-model GPU tests (`MINILLM_RUN_MODEL_TESTS=1`) | 349 passed |
| `check_policy_equivalence.py` (8 requests x 9 policies, 64 tokens) | **72/72 identical** to eager whole-prompt, including padded buckets and both split orders |
| Graph replay coverage in the gate | `whole_graph` 65/65 decode steps replayed; split modes 100% |
| Allocator retries across all benchmark cases | 0 |

**Reviewer fix.** Graph-replayed decode-only steps (~23 ms) were fed to the adaptive
planner's cost model, which predicts eager mixed steps (~55 ms fixed cost), so it
under-predicted and over-sized chunks (48 vs 44 prompt chunks on the gate trace). Graph
decode-only steps and split steps are no longer observed; eager observations are
unchanged (now 47 vs 46, within timing variation).

### Decode (identical 512-token prompts arriving together, 128 outputs)

| Batch | Eager TPOT | Graph TPOT | Speedup | Decode tok/s eager / graph | End-to-end tok/s eager / graph | GPU busy eager / graph |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 63.8 ms | 22.6 ms | 2.83x | 15.7 / 44.3 | 15.6 / 43.3 | 38% / 88% |
| 2 | 63.7 ms | 23.1 ms | 2.76x | 31.4 / 86.8 | 30.9 / 82.2 | 38% / 99% |
| 4 | 64.1 ms | 24.0 ms | 2.67x | 62.4 / 166.9 | 60.0 / 148.6 | 38% / 88% |
| 8 | 65.8 ms | 25.9 ms | 2.54x | 121.7 / 309.0 | 110.7 / 244.6 | 36% / 92% |
| 16 | 66.9 ms | 29.5 ms | 2.27x | 239.2 / 543.5 | 197.2 / 363.7 | 44% / 94% |
| 32 | 66.8 ms | 37.8 ms | 1.76x | 479.5 / 846.1 | 332.3 / 472.5 | 50% / 87% |
| 8 (4096-token prompt) | 63.3 ms | 46.0 ms | 1.38x | 126.4 / 174.1 | 63.4 / 73.2 | 68% / 96% |

Graph decode removes the per-step launch overhead: 2.8x faster at batch 1, shrinking
to 1.8x at batch 32 as real GPU work grows, and 1.4x with 4096-token contexts where
attention over the KV cache dominates. Eager TPOT in this serving runner (~64-67 ms)
is above the static engine's ~56 ms, most likely from per-step scheduler bookkeeping
(not profiled here); graph replay hides it as well. End-to-end rates include eager
prefill.

### Prefill (one request, one output token)

| Prompt tokens | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Eager TTFT (ms) | 65 | 65 | 95 | 188 | 411 | 956 | 2,170 |
| Graphs enabled TTFT (ms) | 64 | 65 | 96 | 188 | 410 | 954 | 2,168 |

Prefill is never graphed (variable shapes, and it already keeps the GPU 93-100% busy
from 1,024 tokens up); graph replays were 0 across all prefill cases, and TTFT is unchanged.

### Serving

| Workload | Policy | Decode | TTFT p50 / p95 | TPOT p50 | Worst pause (median of runs) | Output tok/s | GPU busy | Graph replay share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 requests, 128-1024 tokens, every 150 ms | Continuous (whole-prompt prefill) | eager | 173 / 331 ms | 68.2 ms | 248 ms | 196.9 | 37% | 0% |
| 16 requests, 128-1024 tokens, every 150 ms | Continuous (whole-prompt prefill) | graph | 140 / 314 ms | 34.7 ms | 229 ms | 341.6 | 98% | 100% |
| 16 requests, 128-1024 tokens, every 150 ms | Mixed + adaptive | eager | 144 / 441 ms | 69.6 ms | 150 ms | 188.2 | 37% | 0% |
| 16 requests, 128-1024 tokens, every 150 ms | Mixed + adaptive | graph | 133 / 347 ms | 34.0 ms | 155 ms | 342.0 | 80% | 90% |
| 16 requests, 128-1024 tokens, every 150 ms | Mixed + adaptive | graph + split | 180 / 405 ms | 34.8 ms | 133 ms | 338.6 | 92% | 100% |
| 6 requests, 128/6144 tokens, every 400 ms | Continuous (whole-prompt prefill) | eager | 3,431 / 4,436 ms | 65.0 ms | 3,266 ms | 41.4 | 58% | 0% |
| 6 requests, 128/6144 tokens, every 400 ms | Continuous (whole-prompt prefill) | graph | 2,201 / 3,179 ms | 61.4 ms | 1,648 ms | 51.2 | 78% | 100% |
| 6 requests, 128/6144 tokens, every 400 ms | Mixed + adaptive | eager | 1,072 / 3,512 ms | 96.8 ms | 163 ms | 39.2 | 46% | 0% |
| 6 requests, 128/6144 tokens, every 400 ms | Mixed + adaptive | graph | 1,101 / 3,522 ms | 77.3 ms | 164 ms | 49.8 | 94% | 69% |
| 6 requests, 128/6144 tokens, every 400 ms | Mixed + adaptive | graph + split | 1,110 / 4,772 ms | 84.1 ms | 145 ms | 43.2 | 90% | 100% |

- Steady serving (short prompts): graphs halve TPOT (68 -> 35 ms) and raise output
  throughput 1.7-1.8x (197 -> 342 tok/s); mixed batching replays a graph on 90% of
  decode steps because most iterations carry no prompt chunk.
- Long prompts arriving constantly: most mixed iterations carry a prompt chunk and stay
  eager, so graphs replay on 69% of steps and the gain is smaller (TPOT 97 -> 77 ms,
  throughput +27%). For whole-prompt prefill the worst pause halves (3.3 -> 1.6 s), but
  graphs do not touch prefill: faster decode shifts arrivals so the two 6144-token
  prompts are no longer prefilled in the same 3.2 s pass. Treat that as a schedule
  effect, not a graph speedup.
- Split mode (graph decode + separate eager prompt forward) graphs every decode step
  but adds a second forward per mixed iteration: it gives the smallest worst pause
  (133-145 ms) at the cost of higher TTFT (p95 4.8 s on the long-prompt trace). The
  default (single eager mixed forward for chunk-carrying steps) is the better trade.
- `serving_whole_graph` captured one bucket inside a measured run (its warm-up never
  produced that batch size); the ~270 ms capture stays in that run's metrics.

### Overhead

| Bucket | Capture time | Graph private pool |
| --- | --- | --- |
| 1 | 294 ms | 22 MiB |
| 2 | 263 ms | 22 MiB |
| 4 | 262 ms | 22 MiB |
| 8 | 268 ms | 22 MiB |
| 16 | 274 ms | 22 MiB |
| 32 | 276 ms | 24 MiB |

Each bucket captures once in about 0.26-0.29 s (three warm-up forwards, capture, one
replay) and keeps a ~22-24 MiB private pool. Scratch pages for padding rows add
72 MiB (32 pages) on top of request KV capacity; static input buffers
are a few hundred bytes per bucket. Measured decode cases captured nothing (all
captures happened in warm-up).
