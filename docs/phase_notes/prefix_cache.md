# Exact-token paged prefix caching

Status (2026-09-29): complete. Cached and uncached outputs match on every L4
trace workload, shared-block, reference-count, and eviction tests pass, and a
reused 512-token prefix cuts TTFT P50 by 23–26%. The only mismatch against the
separate dense reference is `prefix-002`, an exact BF16 logit tie in that
reference (see the 2026-09-29 section). Sections below that section header
describe the original September runs and are kept for history.

## What is implemented

`PagedPrefixCache` retains complete prompt KV blocks in the existing physical
page pool. Its lookup hashes complete token blocks, verifies exact token IDs
even on a hash match, and chooses the longest shared whole-block prefix. The
request reuses those immutable blocks through a new
block table. Per-block reference counts prevent a request release or LRU
eviction from clearing pages still held by another owner. Writes to shared
blocks through the checked cache APIs are rejected.

The prototype leaves at least one prompt token uncached so the model computes
fresh logits for the first output token. A cold request runs dense SDPA prefill
and copies its KV into pages. A warm request gathers the cached prefix into a
Transformers `DynamicCache`, runs only the uncached suffix with SDPA, and
appends the new KV to its own pages. Both paths then decode through the
existing direct paged attention path. The persistent runner serves sequential
batch-1 requests; cache entries are bounded by `max_entries` and evicted in
least-recently-used order under entry or block pressure.

`ContinuousPrefixPagedRunner` integrates the same cache with FIFO arrival
scheduling. It admits at most one prefill between decode iterations, using
uncached suffix length against the token budget while decode is active. It
batches one live decode token per active request without padding KV histories
and uses block tables for ragged attention. The admission check accounts for
each active request's remaining KV growth and defers a new request until
capacity is available. A request larger than total capacity fails explicitly;
cancelled and finished requests release their page references. An uncached
control follows the same schedule. Since active rows and page tables change,
`decode_mode=graph` explicitly falls back to eager paged decode; it does not
silently claim graph replay.

The trace benchmark compares both modes against a separate dense manual
greedy reference for every prompt. It uses a shared tokenized system prompt
with varied tokenized user suffixes, output lengths, arrivals, and target
reuse rates of 0%, 50%, and 100%. It
saves raw request events and before/after results, including TTFT, ITL/TPOT,
TPS, cache hit rate, reused/computed prefill tokens, peak physical KV use,
queue delay, and eviction wall time. Each measured repetition starts with an
empty prefix cache; the first shared-prefix request is therefore cold.

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_prefix_trace \
  --config minillm_l4/configs/workloads/qwen3_fp8_prefix.yaml \
  --output-dir minillm_l4/results/prefix_cache/continuous
```

The L4 was initially isolated from the sandbox but became accessible for the
benchmark. The continuous results below are separate from the earlier
sequential batch-1 comparison.

## Reproduce on the L4

From the workspace root:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_prefix_cache \
  --workload long --output-tokens 8 --block-size 16 --num-blocks 256 \
  --warmup-pairs 1 --repetitions 3 \
  --output minillm_l4/results/prefix_cache/long_b16_o8_sdpa.json
```

The command runs paired cold and warm requests against the pinned Qwen3-4B FP8
checkpoint. It compares both outputs with the saved Phase 1 exact-token
reference. Model loading, one warm-up pair, and cache construction are
excluded from measured requests. Request wall time includes lookup, KV gather,
prefill, decode, streaming events, and release. TTFT starts at `prefill_start`
and ends when the first token is ready. Each repetition clears prior prefix
entries before its cold request; the warm request immediately follows it.

## Earlier sequential L4 result

Superseded by the 2026-09-29 re-run below; kept for history.

One NVIDIA L4, block size 16, batch 1, eight output tokens, one warm-up pair,
and three measured pairs. The same pinned prompt is repeated for the warm
request. All cold and warm outputs matched the saved reference token IDs.

| Prompt tokens | Reused tokens | Cold TTFT P50 | Warm TTFT P50 | TTFT speedup | Cold E2E P50 | Warm E2E P50 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 112 | 62.1 ms | 65.7 ms | 0.95× | 483.1 ms | 484.9 ms |
| 512 | 496 | 101.1 ms | 64.8 ms | 1.56× | 512.3 ms | 473.3 ms |
| 2,048 | 2,032 | 494.7 ms | 65.3 ms | 7.58× | 923.6 ms | 492.5 ms |

Raw paired measurements are saved under `results/prefix_cache/` as
`short_b16_o8_sdpa.json`, `medium_b16_o8_sdpa.json`, and
`long_b16_o8_sdpa.json`.

A separate 2,048-token L4 check changed the final prompt token after caching
the original prompt. The modified warm request reused 2,032 prefix tokens and
produced the same eight output IDs as a cold request with the modified prompt.
Its artifact is `results/prefix_cache/long_divergent_check.json`. That check
had no warm-up pair, so its latency includes first-call initialization and is
not part of the steady-state table above.

The first direct paged warm-suffix implementation was exact but slow: at 128
prompt tokens its TTFT was about 1,271 ms because multi-token suffix attention
used the readable blockwise fallback. The default SDPA suffix adapter brought
that down to about 64 ms. The short prompt does not benefit because KV gather
and cache control cost more than the prefill work saved. Long-prompt TTFT gains
are much larger than end-to-end gains because decode is unchanged.

## Before/after continuous comparison

Superseded by the 2026-09-29 re-run below. This trace ran with BF16 score
rounding forced in paged decode, the pre-fix Triton kernel, and the harness
timing bug fixed in Phase 0; kept for history.

The final L4 trace used the pinned Qwen3-4B FP8 revision above, BF16
activations, block size 16, 512 reserved blocks, maximum decode batch size 4,
and eight requests every 20 ms. Each prompt had a 512-token prefix plus a
16- or 32-token suffix; requested outputs alternated between eight and four
tokens. One warm-up and three measured repetitions ran per mode and reuse
rate. TTFT here includes arrival-to-admission queueing, unlike the earlier
sequential table. The model used the SM89 FP8 kernel; the GPU was an NVIDIA
L4 with driver 580.178.04, PyTorch 2.13.0+cu130. Raw JSON and events are in
[`final_with_sha`](../../results/prefix_cache/continuous/final_with_sha/manifest.json),
including P50/P95/P99, system information, commit
`a5b2aca361f7c5a2eef2c7cbfce13449b8a233b0`, and `git_worktree_dirty=true`.
Because the worktree was dirty, the commit alone is not sufficient to recreate
this exact code state.

| Target reuse | Actual hit rate | TTFT P50 uncached → cached | TTFT P95 uncached → cached | Output TPS P50 uncached → cached | Cached vs uncached | Dense reference |
|---:|---:|---:|---:|---:|---|---|
| 0% | 0% | 666.8 → 670.0 ms | 1307.4 → 1316.7 ms | 27.27 → 27.13 | Pass | Fail, request 002 |
| 50% | 37.5% | 676.8 → 535.7 ms | 1333.8 → 1206.8 ms | 26.86 → 28.59 | Pass | Pass |
| 100% | 87.5% | 679.8 → 505.2 ms | 1331.3 → 986.9 ms | 26.85 → 33.26 | Fail, request 006 | Fail |

At 50% target reuse, three of eight requests hit the cache, 1,536 prompt
tokens were reused per run, and measured prefill work fell from 4,288 to
2,752 tokens. Median TPOT changed from 102.6 to 100.9 ms, median E2E from
1,287 to 1,151 ms, and peak physical KV-block utilization was 34.2% versus
52.9% at 0% reuse. Peak allocated VRAM was approximately 5,518 MiB cached
versus 5,509 MiB uncached after isolating runner allocations. No evictions
occurred in this capacity-rich workload; LRU eviction remains CPU-tested.

The 100% row shows a larger latency benefit, but it is diagnostic, not a
promoted speedup: request 006 differs at its fifth generated token
(uncached ID 895, cached ID 830). The cached top-two logit gap at that step
was zero, consistent with a numerically sensitive greedy decision; this does
not prove the exact source of the difference. At 0% reuse, cached and
uncached paged paths agree, but both differ from dense manual decoding for
request 002 at its fourth token. That mismatch also occurs with one active
request and the PyTorch paged-attention backend, so it is not solely a Triton
or continuous-batch membership bug. The opt-in Triton BF16 score-rounding
path removed an earlier first-decode mismatch but did not resolve these two
remaining cases.

## 2026-09-29 fixes and L4 re-run

### Fixes

A review of the prefix-caching code found no path that corrupts or frees
shared blocks early, but found seven issues, all fixed:

1. Continuous decode forced `paged_decode_match_dense_rounding=True`, which
   rounded attention scores to BF16 and undid the Phase 6 FP32 Q·K precision
   fix; the sequential runner did not, so the two benchmarks used different
   attention numerics. The option was added to chase what turned out to be a
   tie and has been removed everywhere.
2. `PagedPrefixCache.attach` counted hits, misses, and reused tokens at
   attach time. The continuous runner can release an attached request and
   defer it, so a retry counted again and inflated hit rate and reuse.
   Counting now happens once, through `record_admission`, when the request is
   actually admitted.
3. `PagedKvAllocator.snapshot` summed used and reserved slots per sequence, so
   a shared block counted once per owner (including the cache's own owners)
   and reported KV use could exceed 100%. It now counts each physical block
   once and reports the per-owner sum separately as `logical_token_slots`.
4. The shared-block write guard in `PagedQwen3Attention` ran for every row in
   all 36 layers at every decode step once any block was shared, which is host
   work on a host-bound path that only the cached run paid. Every layer writes
   the same positions through the same block tables, so it now runs in layer 0
   only.
5. The continuous runner computed a top-2 logit gap with an extra top-k and
   host transfer on every decode step. This tie diagnostic is now opt-in
   (`record_logit_gaps`, `--record-logit-gaps`).
6. `evict_until_free` evicted least-recently-used entries even when their
   blocks were still referenced by running requests, which freed nothing and
   could empty the cache while still failing. Eviction now skips entries with
   no freeable blocks, in both `evict_until_free` and `attach`.
7. The trace labeled both harness runs `continuous_paged_prefix`; each now
   carries its runner's name.

New tests: `test_attach_counts_hits_only_when_the_request_is_admitted`,
`test_eviction_skips_entries_pinned_by_active_requests`, and
`test_allocator_snapshot_counts_shared_physical_blocks_once`.

### Continuous trace

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_prefix_trace \
  --config minillm_l4/configs/workloads/qwen3_fp8_prefix.yaml \
  --output-dir minillm_l4/results/prefix_trace_20260929
```

Eight requests arriving every 20 ms, each a 512-token prefix plus a 16- or
32-token suffix, 8 or 4 output tokens, maximum decode batch 4, block size 16,
512 physical blocks, `sm89` projections, one warm-up and three measured
repetitions per mode (commit `16ef083` plus these fixes, uncommitted at run
time). Hit rate, reuse, and prefill work were identical in all three
repetitions.

| Target reuse | Hit rate | Reused / computed prefill tokens | Peak KV blocks (cached run) | TTFT P50 uncached → cached | TTFT P95 uncached → cached | TPOT P50 uncached → cached | Output TPS P50 uncached → cached | Cached = uncached |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0% | 0% | 0 / 4,288 | 271 | 645.2 → 655.2 ms | 1,297.5 → 1,311.8 ms | 102.2 → 103.1 ms | 27.77 → 27.46 | pass |
| 50% | 37.5% | 1,536 / 2,752 | 175 | 651.5 → 502.5 ms | 1,311.0 → 1,151.6 ms | 103.2 → 99.1 ms | 27.49 → 30.34 | pass |
| 100% | 87.5% | 3,584 / 704 | 47 | 653.9 → 481.2 ms | 1,310.4 → 962.2 ms | 102.6 → 83.8 ms | 27.36 → 34.39 | pass |

At 100% target reuse, seven of eight requests hit (the first shared request is
cold), prefill work falls by 84% and TTFT P50 by 26%, and output TPS rises 26%.
Peak KV use falls from 271 blocks with no reuse (the 0% workload, same prompt
lengths) to 47 because hits share the same physical prefix pages. The trace
records cache statistics only for the cached runner, so uncached peak blocks at
50% and 100% are not measured directly. At 0%
reuse caching costs nothing measurable (TTFT within run-to-run noise). No
evictions occurred.

Cached and uncached outputs now match at every reuse rate. The earlier 100%
mismatch (`prefix-006`: "the following statement is false" uncached vs "…is
true" cached) happened at a step where the cached run's top-2 logit gap was
exactly 0; with BF16 score rounding removed it no longer flips.

The trace also compares against a separate dense manual-greedy reference.
The only mismatch is `prefix-002` at 0% reuse (no reuse involved, so cached and
uncached agree with each other): the reference emits " the process of creating
a new version of" and the paged path " the process of photosynthesis works in
plants". At the fourth token the reference's logits for " creating" and
" photos" are both exactly 9.5000 in BF16, so its choice is decided by
`argmax` tie-breaking; a dense run with FP64 attention prefers " photos" (see
the Phase 6 note). This is a tie, not a prefix-caching defect. The command
still exits non-zero because the dense-reference comparison is strict.

### Sequential cold/warm pairs

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_prefix_cache \
  --workload long --output-tokens 8 --block-size 16 --num-blocks 256 \
  --warmup-pairs 1 --repetitions 3 \
  --output minillm_l4/results/prefix_sequential_20260929/long_b16_o8.json
```

(and `--workload short` / `medium`). All three passed exact-token checks
against the Phase 1 reference corpus.

| Prompt tokens | Reused tokens | Cold TTFT P50 | Warm TTFT P50 | TTFT speedup | Cold E2E P50 | Warm E2E P50 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 112 | 62.3 ms | 66.0 ms | 0.94× | 478.7 ms | 482.2 ms |
| 512 | 496 | 101.5 ms | 65.3 ms | 1.55× | 509.7 ms | 474.4 ms |
| 2,048 | 2,032 | 492.8 ms | 65.8 ms | 7.49× | 917.2 ms | 492.4 ms |

A warm request's TTFT stays at about 65 ms regardless of prompt length because
only the last block is recomputed; the 128-token prompt gains nothing because
gathering the cached KV costs about as much as recomputing 112 tokens.

## Correctness and remaining limits

- The continuous runner passes CPU tests for mixed arrivals, live decode,
  cached/uncached equality, hash collisions, cancellation, deferred admission,
  and resource cleanup. The paged-attention GPU suite also passes. On the L4,
  cached and uncached outputs match on all three trace workloads.
- Dynamic continuous decode uses eager paged attention; CUDA Graph capture is
  available only in the separate fixed-shape benchmark path.
- Prefill admissions are one request at a time, not packed/chunked. A cold
  request larger than the token budget waits while decode is active; when no
  request is decoding it may run whole to avoid starvation. Phase 8 should
  reduce that waiting with chunking.
- Only complete blocks are shared. The final prompt block is recomputed on an
  identical-prompt hit so fresh first-token logits are available.
- The fast warm path gathers the shared KV into a dense cache for SDPA suffix
  prefill. It saves model computation but still copies prefix KV once per hit.
- Exact-token correctness was checked for the saved short, medium, and long
  prompts, with eight output tokens. The CPU suite also checks divergent
  suffixes, ownership, eviction, and output equality on a tiny Qwen model.
- Cache entries are local to one runner/model instance. The scheduler is
  single-threaded; cross-model reuse, multi-threaded mutation, and durable
  service-level memory policies are out of scope here.

- Evictions did not occur in the L4 trace (512 blocks is capacity-rich), so
  eviction overhead is covered by CPU tests rather than measured on the L4.
