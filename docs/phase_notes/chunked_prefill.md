# Phase 8 — Chunked prefill

## Execution

`ChunkedPrefillPagedRunner` extends continuous paged decoding with resumable
prompt sessions. Configure `prefill_chunk_size` and `max_prefill_tokens`:
each scheduling iteration processes at most one prompt chunk of
`min(prefill_chunk_size, max_prefill_tokens)` new tokens. The final chunk may
be shorter. Intermediate chunks emit no generated tokens.

Each session retains its Transformers `DynamicCache`, advances absolute
RoPE positions, and supplies an attention mask covering the entire valid
prefix plus the new chunk. Only new K/V entries are copied into physical
pages. After the last prompt chunk, the first output token is selected and
the dense session cache is released; subsequent decode uses the paged path.

Active decoding runs before prefill work at each iteration. Partial prompts
rotate in a FIFO round-robin queue, so a long prompt keeps progressing while
other requests decode. Partial prefill and decoding share the active-request
limit. Admission checks enough free pages for every admitted request to finish;
capacity-blocked arrivals stay at the front of the waiting queue.

Optional exact-token prefix reuse attaches immutable whole blocks and gathers
them once to initialize a warm session. New chunks start at the reused
prefix's absolute position. Only completed prompts are published to the
prefix cache. Entries start cold at every benchmark repetition.

Dynamic membership uses eager decode. Requests for graph mode retain the
existing explicit eager fallback. Phase 8 selects the existing
`sdpa_compat` decode policy for both the unchunked control and chunked cases;
the default behavior of the earlier Phase 7 runner is preserved.

## Benchmark

The YAML configuration is
[`qwen3_fp8_chunked.yaml`](../../configs/workloads/qwen3_fp8_chunked.yaml).
It pins the Qwen3 FP8 revision and `sm89` projections, uses four short requests
and a separate six-request mixed trace, and compares the unchunked control
against chunk sizes 128 and 256. All cases share arrivals, output limits,
page capacity, decode policy, and active limit. The prefill budget is 256
tokens per iteration; prefix reuse is disabled by default to isolate chunking.

Run from the parent `LLMPerfLab/` workspace:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_chunked_prefill --dry-run

.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_chunked_prefill \
  --workload all --chunk-sizes 128 256 \
  --repetitions 3 --warmup-repetitions 1
```

The command automatically includes chunk size 0 as the unchunked control.
It respects the prefill budget while decode is active and permits an
oversized whole prompt when decoding is idle, matching the earlier admission
rule. Positive chunk sizes always respect the budget, even when idle.

Default output paths are unique UTC-dated directories in `results/` and
Markdown reports in `benchmark_results/`. Existing nonempty output directories
and existing report files are rejected. Each run saves resolved configuration,
model/environment identity, JSON results, raw JSONL events, scheduler records,
and latency/throughput/telemetry summaries. `source_snapshot/` includes the
actual runtime source, dataset, and reference corpora, with SHA-256 hashes in
`run_provenance.json`, so local uncommitted code is preserved with its results.

Correctness requires exact equality against the existing Phase 1 corpus and
against the unchunked control across every measured repetition. Missing or
wrong-model references are rejected; the command never creates replacement
references. Failed runs retain diagnostics and unavailable metrics, and the
command exits nonzero if any gate fails. Case order is fixed with the control
first, so these measurements are development checkpoints rather than a
randomized multi-session performance study.

The Markdown export reports TTFT and ITL P50/P95/P99, TPOT P50, E2E P95,
aggregate output tokens/s, requests/s, GPU utilization, peak allocated/reserved
VRAM, chunk sizes, chunks while decoding, and both correctness gates.
Unmeasured telemetry is shown as unavailable.

## Validation and limits

CPU tests exercise partial KV continuation and absolute positions, final
logits and greedy token equality, chunk/budget bounds, mixed arrivals,
round-robin progress during live decode, warm-prefix reuse, repetition reset,
EOS, cancellation before execution, capacity backpressure, model failures,
page cleanup, strict reference failures, and raw/Markdown/source-snapshot export.

This implementation retains a dense KV cache for each partially prefilling
request alongside the paged copy. That temporary duplication and the dense
prefix gather are measured limitations; a fused paged/varlen chunk-prefill
kernel is separate optimization work. GPU chunk boundaries synchronize the
current stream before recording completion and yielding to the scheduler.
Cancellation is currently configured before execution, rather than exposed
as a service API for cancelling an already-running chunk.

L4 exact-token checks and the measured latency/throughput trade-off must pass
before Phase 8 is marked complete. CPU correctness alone does not establish
FP8 numerical equivalence or a serving speedup.

The 2026-10-03 L4 smoke tests pass exact HF/control checks for 128- and
256-token chunks on both traces. A 64-token chunk is experimental: it changes
`baseline-short-000` at output index 25 from token 2326 to 1378, in both the
short and mixed workloads. A diagnostic that padded 64 logical tokens to 128
compute rows did not resolve this mismatch. The cause has not been established;
it is not waived as a tie or accepted by changing the reference. The command
still allows explicit exploratory sizes and exits nonzero when they fail.
Reducing the prefill budget below 128 similarly enters an unvalidated setting.

Earlier smoke runs also exposed incompatible split/tile settings for the
SDPA-compatible decode kernel. That runtime defect is fixed: this policy now
uses one KV split and a 128-token key tile, as in the validated Phase 6 path.
The legacy accurate policy retains its previous selectors. A CPU regression
test checks this dispatch contract even when the normal selectors choose a
long-context split. Failed smoke artifacts remain in their dated directories.

## 2026-10-03 L4 validation

The final default matrix uses one warm-up and three measured repetitions per
case, the pinned native-FP8 checkpoint, `sm89` projections, compatible paged
decode, a 256-token prefill budget, and four active slots. GPU and system
sampling are enabled. Prefix reuse is disabled for this matched comparison.
All **6/6 cases** pass both exact-token gates, with **90/90 measured request
outputs** matching the existing HF corpus. No expected token or reference
file was changed. The host regression suite passes **246 tests**, with 14
unrelated opt-in pinned-model tests skipped.

| Trace | Chunk tokens | TTFT P50 (ms) | ITL P95 (ms) | Output TPS P50 | Exact HF/control |
| --- | --- | --- | --- | --- | --- |
| uniform | 0 | 206.08 | 63.06 | 55.77 | pass |
| uniform | 128 | 203.59 | 62.85 | 55.71 | pass |
| uniform | 256 | 206.37 | 63.06 | 55.53 | pass |
| mixed | 0 | 6,354.41 | 62.58 | 17.02 | pass |
| mixed | 128 | 1,749.10 | 125.17 | 33.10 | pass |
| mixed | 256 | 968.78 | 132.57 | 37.33 | pass |

Short prompts fit in a single default chunk, so throughput stays within
run-to-run noise. On the mixed trace, chunking lets long prompts progress
without waiting for the decode batch to drain: 256-token chunks increase
aggregate throughput from 17.02 to 37.33 tokens/s and reduce median TTFT
from 6.35 to 0.97 seconds. The trade-off is higher P95 ITL, from 62.6 to
132.6 ms, because bounded prefill work is now interleaved with live decode.
The control avoids those interruptions by delaying oversized prefills.

This satisfies the Phase 8 checkpoint for the saved corpus and default
128-/256-token presets. It does not validate the experimental 64-token
setting or establish production tail-latency guarantees. The remaining
numerical mismatch is retained as follow-up work.

Full results: [Phase 8 measurements](../../benchmark_results/phase_08_chunked_prefill_20261003.md).
Raw JSON, events, source snapshot, and configuration are in
`results/phase8_chunked_prefill_final_20261003/`.

## 2026-10-04 focused confirmation

The same six-case default matrix passed again with one warm-up and three
measured runs per case: 90/90 outputs exactly match HF and whole-prompt
controls. An independent audit verified raw events, latency percentiles,
throughput, bounded chunks, decode-first ordering, complete prompt coverage,
active limits, page cleanup, and all 96 source/input snapshot hashes.

For the mixed trace, 256-token chunks improve aggregate throughput from
16.41 to 36.05 tokens/s and median TTFT from 6646.30 to 1006.33 ms. P95 ITL
rises from 64.98 to 136.20 ms. This confirms the earlier checkpoint's
throughput/TTFT benefit and token-gap trade-off. The extended sweep is deferred,
with its existing raw results preserved.

Results: [focused Phase 8 confirmation](../../benchmark_results/phase_08_focused_20261004.md).
Raw evidence: `results/phase8_focused_20261004/`.

## 2026-10-05 expanded-corpus limits

The extended matrix completed with 199 passing, 237 failing, and 12
memory-guard-skipped configurations. All 436 executed cases complete at the
requested lengths and are token-stable across three repetitions. Long mixed
outputs fail across static, whole-prompt, and chunked policies; native batched
decoding also has failures. A fresh HF batch-2 replay reproduces one divergence,
so the single-request reference is sensitive to batch shape in this example.

The new 512-token synthetic prompt also fails at chunk 256 and active limits
1/2/4, including the 128-output trace. A fresh single-request replay confirms
its first difference at output index 106, with winning logit margin 3.5;
whole-prompt and chunk-128 probes match the saved prefix through their tested
limits. This is not an exact tie, and the precise operator/cache source is
unresolved. It prevents promoting chunk 256 as a generally validated preset.
The earlier checkpoint remains scoped to its original corpus.

Passing comparisons show chunking benefits at active limits 4/8, but losses
at lower concurrency with long prompts, as well as increased P95 token gaps.
Full analysis and fresh replay evidence:
[extended sweep interpretation](../../benchmark_results/prefill_decode_analysis_20261005.md).

## 2026-10-07 resource admission, flattened prefill, and mixed batching

### Problems found

- **Requests queued while resources were free.** Whole-prompt mode deferred
  any prompt longer than the 256-token budget while a row was decoding; at an
  active limit of 8 it never exceeded 2 in-flight requests (504 deferrals,
  13–26% KV use). Chunked mode admitted one request and ran one chunk per
  iteration. The sweep also sized the page pool to `batch_limit` requests.
- **Chunk exactness.** Later chunks fell back from FlashAttention to cuDNN
  because they need an offset mask; one bf16 rounding step per layer grew into
  a different token at chunk 256 on a 512-token prompt.
- **Simultaneous prompts were prefilled one at a time**, each as its own
  forward, although the project already had a flattened (packed) prefill path.
- **Prefill and decode ran as separate forwards**, so a long prefill froze
  every decoding row (3.2 s with 6144-token prompts).

### Changes

- **Admission** (`benchmarks/runners/chunked_prefill.py`): every arrived
  request is admitted while a slot and KV pages for its full length are free;
  requests queue only on exhausted pages. The whole-prompt budget deferral is
  removed. The sweep's `--admission resource` makes batch size offered load and
  sizes the pool for every request.
- **Aligned chunk attention** (`engine/generation/chunked_prefill.py`): a later
  chunk's queries are zero-padded to the cached length and attend through the
  same Transformers SDPA call as a whole prompt; causal rows are independent,
  so KV and logits are bitwise identical to whole-prompt prefill.
- **Flattened whole-prompt prefill**: prompts admitted together share one
  `qwen3_packed_prefill` forward (`cu_seqlens` metadata, no padding).
- **Mixed batching** (`mixed_batch=True`; `engine/model_runner/qwen3_mixed.py`,
  `engine/kv_cache/mixed.py`): each iteration is one flat forward holding every
  decode token plus prompt chunks, shortest remaining prompt first, within a
  fixed token budget (decode rows included). Projections, norms, the MLP, and
  `lm_head` run once over all rows; attention is dispatched per row type —
  decode rows use the decode path's own writer and paged kernel
  (`PagedQwen3Attention._paged_attend`), prompt chunks use one Triton K/V
  scatter per layer and the aligned SDPA call over their gathered prefix.
  If decode rows alone fill the budget, prompts still advance one page per
  step. Prefix reuse is not yet supported in this mode.

Exactness rests on measured row independence on the L4: the SM89 FP8 linear,
RMSNorm, and `lm_head` give each row bitwise the same result alone or inside a
700-row call, so a decode row inside a mixed forward equals a decode-only step
and a prompt chunk equals single-request prefill.

### Validation

- CPU: 276 tests, including mixed-batch token equality against the manual
  reference across budgets 6/9/64 with arbitrary chunk boundaries, the budget
  invariant, decode-every-step, shortest-first ordering, and resource
  admission (admit-all with free pages; queue only on exhausted pages).
- L4: 290 tests with `MINILLM_RUN_MODEL_TESTS=1`. `stream_generate.py
  --check-hf` passes exactly for whole-prompt, chunked, and mixed (budgets 512
  and 2048) on 128/2048 and 128/6144 mixes.

### Same-session comparison

`scripts/stream_generate.py`, 6 requests arriving every 400 ms, prompts
alternating 128 and N tokens, 64 outputs, one warm-up, FP8 kernel pre-tuned:

| N | Policy | TTFT median / max | Worst decode pause | Output tokens/s |
|---|---|---|---|---|
| 6144 | whole prompt, flattened | 3.63 / 4.63 s | 3,183 ms | 40.4 |
| 6144 | chunked 512 | 3.10 / 8.50 s | 617 ms | 26.9 |
| 6144 | mixed, budget 512 | 1.48 / 6.20 s | 342 ms | 31.0 |
| 6144 | mixed, budget 2048 | 1.63 / 3.84 s | 651 ms | 38.2 |
| 2048 | whole prompt, flattened | 371 / 513 ms | 556 ms | 60.4 |
| 2048 | chunked 512 | 544 / 875 ms | 216 ms | 57.0 |
| 2048 | mixed, budget 512 | 395 / 611 ms | 137 ms | 57.2 |
| 2048 | mixed, budget 2048 | 336 / 554 ms | 407 ms | 56.4 |

Mixed batching beats the separate-forward chunked mode on every column and
bounds the decode pause by the budget; against whole-prompt prefill it costs
5–7% throughput (aligned-attention padding and more, smaller steps). These are
single measured runs; the harness sweep below repeats them with the exact-HF
gate.

## 2026-10-08 adaptive chunking, lower-right attention, and Phase 8 completion

### Lower-right causal attention for chunks

A later chunk previously zero-padded its queries to the whole prompt, so a
small chunk late in a long prompt cost a full-prompt attention (a 64-token
chunk at position 8000 cost about as much as an 8k prefill's attention).
`aligned_prefill_sdpa` now uses FlashAttention with a
`causal_lower_right(chunk, prefix)` mask for chunks of more than 128 rows. It
is bitwise identical to whole-prompt rows on 138 random (length, chunk) pairs
up to 12k tokens and up to 14x faster per layer for small late chunks. Chunks
of 128 rows or fewer (one query block with 32 heads, which FlashAttention
splits across keys on the L4) keep the padded path.

### Adaptive chunk sizing (`engine/step_planner.py`)

A forward pass cannot be interrupted, so a request arriving mid-pass waits for
the rest of it. `adaptive_chunking=True` therefore limits each pass by time:

- **busy** (rows decoding, prompts waiting, or an arrival in the last 0.5 s):
  150 ms per pass; **idle** (one prompt, nothing else): 400 ms per pass;
- `StepCostModel` predicts pass time (fixed cost + per-token cost + attention
  over each chunk's prefix + decode rows), starting from L4 priors and refit
  by ridge regression from measured passes;
- shortest remaining prompt first with aging (2,000 tokens of priority per
  second waited), a minimum chunk of 144 tokens, never a final remainder of
  1-128 tokens, and tokens per pass capped by free GPU memory. The memory cap
  is hard; the time limit is a target that the minimum chunk and the tail rule
  may exceed (each plan reports `over_limit`; see "Robustness fixes" below).

Whole fresh prompts (single or packed together) now always take the packed
path, which writes K/V straight into pages and is 9-20% faster than the
Transformers forward plus page copy.

### Validation

- `scripts/check_policy_equivalence.py` (L4, no HF): 8 staggered requests of
  128-8192 tokens with 64 outputs each through whole-prompt, chunked-256,
  mixed-512, mixed-2048, and adaptive. **32/32 request outputs identical** to
  whole-prompt prefill, including 1- and 16-token final chunks (padded path)
  and large chunks (lower-right path).
- CPU: 280 tests, including adaptive token equality, the time limit, aging,
  the tail rule, and cost-model recalibration.

### Results

Prefill-only TTFT (`benchmarks/commands/run_prefill_ttft.py`, 3 measured runs;
full tables in `benchmark_results/prefill_ttft_20261007.md` and
`benchmark_results/adaptive_chunking_20261007.md`):

| Prompt tokens (one prompt) | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|
| TTFT (ms) | 62 | 68 | 104 | 224 | 501 | 1,152 | 2,591 |

Prefill has a ~60 ms floor and the GPU saturates near 512-1024 tokens per
forward (~4,900 prompt tokens/s); short prompts must be batched, and forwards
beyond ~2048 tokens only make more requests wait.

A 128-token request arriving while an 8192-token prompt prefills on an idle
engine:

| Policy | Newcomer TTFT | Long prompt TTFT |
|---|---|---|
| flattened (no chunks) | 1.2-2.2 s | 2.2 s |
| fixed chunk 4096 | 1.2-2.1 s | 2.1-2.5 s |
| fixed chunk 2048 | 0.8-1.0 s | 2.4 s |
| fixed chunk 512 | 0.17-0.27 s | 2.4 s |
| **adaptive** | **0.28-0.59 s** | **2.0 s** |

Sixteen mixed-length prompts (128-8192) arriving together: adaptive median
TTFT 0.62 s (flattened 3.9 s), all done in 7.5 s (8.3-9.0 s for the others),
4,386 prompt tokens/s (+11%).

With decode traffic (`stream_generate.py`, 6 requests every 400 ms, 64
outputs, same session):

| Mix | Policy | TTFT median / max | Worst decode pause | Tokens/s |
|---|---|---|---|---|
| 128 / 6144 | whole prompt | 2.14 / 3.10 s | 1,622 ms | 42.4 |
| 128 / 6144 | mixed 512 | 0.94 / 3.29 s | 156 ms | 40.4 |
| 128 / 6144 | mixed 2048 | 1.27 / 2.81 s | 522 ms | 42.5 |
| 128 / 6144 | adaptive | 1.13 / 3.45 s | 158 ms | 39.6 |
| 128 / 2048 | whole prompt | 292 / 460 ms | 462 ms | 59.5 |
| 128 / 2048 | mixed 512 | 371 / 574 ms | 118 ms | 57.0 |
| 128 / 2048 | mixed 2048 | 340 / 564 ms | 406 ms | 57.6 |
| 128 / 2048 | adaptive | 326 / 551 ms | 137 ms | 58.1 |

Decode interference (`scripts/analyze_prefill_interference.py` over the
2026-10-07 sweep): whole-prompt prefill adds the newcomer's entire prefill to
every decoding row in one gap (102 ms at 512 tokens, 495 ms at 2048, 1.14 s at
4096, 2.55 s at 8192); budgeted mixed batching spreads and, up to 4096 tokens,
shrinks that stall, bounding the worst gap by the budget.

### Definition of done

- Chunked and unchunked outputs match: satisfied (bitwise attention; 32/32
  policy outputs identical end to end; the 2026-10-07 sweep passed 79/80 with
  the one failure a static-batching out-of-memory, not a mismatch).
- Long prompts continue making progress: satisfied (aging, minimum chunk,
  resource admission).
- Active decode requests are not starved: satisfied (every decode row runs in
  every mixed pass; worst pause 3.2 s -> ~0.16 s on the 6144-token mix).
- Latency/throughput trade-off documented: this note and the reports above.

Phase 8 is complete. Recommended default: `mixed_batch=True,
adaptive_chunking=True`. Remaining limits: prefix reuse is not supported in
mixed mode; mixed passes cost ~3 ms more than a plain decode step; decode is
still eager (CUDA Graph decode inside the mixed runner is Phase 9).

### Robustness fixes (2026-10-08)

A read-only review found three serving bugs, each reproduced on CPU and now
covered by a regression test in `tests/test_chunked_prefill.py`:

- **Late cancellation.** A request marked for cancellation that arrived while
  other rows were decoding was admitted at the second admission point, which
  skipped the cancellation check, and ran to completion. Both admission points
  now cancel.
- **Context length.** Admission checked only KV pool capacity, so a prompt
  longer than the model's positions was accepted. Like vLLM, a request whose
  prompt plus `max_new_tokens` exceeds `max_model_len` (default: the model's
  `max_position_embeddings`) now fails at admission without running.
- **Soft memory cap.** The planner's 144-token progress floor overrode the
  free-memory token cap, and the memory estimate never went below one page.
  The cap is now hard: decode rows that use it all make prompts wait; with no
  decode rows a prompt still advances one page. Steps whose plan exceeds the
  time limit are counted (`planned_steps_over_time_limit` in the run summary).

CUDA out-of-memory behavior is now defined: an `OutOfMemoryError` during any
forward fails every admitted, unfinished request (their KV may be partly
written), releases their pages, empties the allocator cache, and keeps serving
the queue; finished requests keep their outputs. The run summary reports
`oom_errors` and `oom_failed_requests`. Any other exception still aborts the
run. Whole-prompt and fixed-budget paths have no memory cap; only adaptive
chunking sizes steps from free memory.
