# Paged KV allocation and direct attention

## What was built

Phase 6 adds a fixed-block KV memory layer without removing the earlier dense
cache path.

- `PagedKvAllocator` owns a deterministic free-block pool.
- Each request has a logical block table containing physical block IDs.
- Appending tokens allocates only newly required blocks.
- Releasing a request immediately returns all of its blocks.
- Capacity exhaustion raises a controlled `PagedKvOutOfMemoryError` before
  ownership is partially changed.
- `PagedKvCache` stores keys and values as
  `[layers, blocks, kv_heads, block_size, head_dim]` tensors.
- `gather_layer` reconstructs one request as
  `[1, kv_heads, tokens, head_dim]`.
- `gather_batch` left-pads variable-length rows into the dense representation
  currently expected by the Hugging Face model.
- `block_table_tensor` exposes logical-to-physical mappings for an indirection
  implementation.
- `append_from_dynamic_cache` copies only the newly appended right-aligned
  suffix from the dense model result back into page storage.
- `paged_attention` reads physical blocks directly and performs an online
  softmax without building a dense batch KV tensor.
- `PagedAttentionBatchRunner` installs a Qwen3 attention adapter that reserves
  the next logical position, writes each layer's K/V into its pages, and calls
  direct block-wise attention for prefill and decode.
- `PagedHybridBatchRunner` uses the model's optimized SDPA path for prefill,
  materializes that prefill cache into physical pages once, and then uses
  direct block-table attention for every decode token.
- `PackedPagedPrefillBatchRunner` provides the variable-length prefill path:
  it flattens all prompts into `[sum(prompt_lengths)]`, records request
  boundaries with `cu_seqlens`, and writes each request's K/V directly into
  its own physical pages. No prompt padding or per-request full-model prefill
  is used. CUDA `auto` selects the exact-token-safe project-owned Triton
  backend. Packed K/V page writes use one Triton scatter kernel per layer on
  CUDA; fused SDPA remains an experimental comparison backend.
- `triton_paged_decode_attention` fuses block-table lookup, physical K/V page
  loads, grouped-query head mapping, QK reduction, online softmax, and weighted
  value accumulation into one GPU launch per model layer.
- The Triton decode and packed-prefill kernels use explicit compute tiles.
  Decode selects a 16-token reduction tile below 1,024 context tokens and a
  64-token tile at or above 1,024, independently of physical KV page size.
  Keeping those dimensions separate stabilizes online-softmax reduction order
  and keeps greedy-token validation stable when physical block sizes are swept.
- `sm89_fp8_linear` replaces the generic Transformers projection dispatcher
  on the L4. It fuses dynamic per-block activation scaling, FP8 conversion,
  block-scaled matrix multiplication, and FP32 accumulation. Unsupported
  tensors retain the original Transformers fallback.
- `PagedKvBatchRunner` remains as the dense-gather fallback/control path.
- Benchmark artifacts separate prefill model time, prefill cache
  materialization, decode model time, and logical page visits.

## Packed varlen prefill

The regular Qwen3 model interface expects a rectangular `[batch, sequence]`
prompt tensor. The packed runner drives the decoder layers directly with a
flat token tensor and metadata:

```text
requests:       A=[a0 a1 a2], B=[b0 b1 b2 b3 b4], C=[c0 c1]
flat tokens:    [a0 a1 a2 b0 b1 b2 b3 b4 c0 c1]
cu_seqlens:     [0, 3, 8, 10]
token request:  [A  A  A  B  B  B  B  B  C  C]
local position: [0  1  2  0  1  2  3  4  0  1]
last indices:   [2, 7, 9]
```

Every decoder layer still performs its projections and MLP over one flat
batch. The packed attention operation uses the request index and local
position to read only that request's pages up to the current token. The
Triton path launches one program per flat token/query-head and uses online
softmax; the PyTorch path computes the same page traversal as a readable
fallback. After prefill, the first-token logits are gathered only at
`last_indices`, and the existing one-token direct paged decode path continues
with per-request sequence lengths.

This is an independent implementation of the same ragged dataflow idea used
by production serving engines. It does not use vLLM internals. `paged_hybrid`
remains the dense SDPA comparison path, while `paged_packed` is the no-padding
path for mixed prompt lengths. The optimized packed SDPA backend still makes
one fused SDPA call per request per decoder layer; it is an intermediate
optimization until a true fused varlen FlashAttention/FlashInfer-style kernel
is integrated.

The implementation was validated with 63 passing CPU tests and four expected
CUDA-only skips. A real L4 run with two 128-token prompts completed with exact
reference tokens after page-aligned Triton reduction. A mixed real-Qwen run
with 128- and 512-token prompts packed 640 input tokens instead of the padded
1,024-token rectangle, recorded `cu_seqlens=[0, 128, 640]`, and completed both
requests through direct paged decode.

## TTFT optimization

The first packed implementation used one Triton program per flat token and
query head and copied each request's K/V page segments from Python. A CUDA
profiler trace for the 128+512-token workload attributed 87.4 ms of 181.4 ms
prefill time to packed attention; many small copy operations also appeared in
the trace. Removing padding did not automatically make that path faster than
dense SDPA. The attention kernel was correct but not sufficiently tiled to
match a production attention backend.

The exact-token-safe optimization was a Triton scatter-write kernel. It maps
each flat token to `(request, local_position)`, looks up the physical page, and
writes both K and V directly. This removes the Python request/page copy loop
without changing attention arithmetic.

The packed runner also has an experimental `sdpa` backend. It keeps the flat
hidden states, request boundaries, and direct page writes, but uses PyTorch's
fused scaled-dot-product attention separately for each request. With one
warm-up and three measured repetitions on the L4, block size 32, output length
32, and prompts of 128 and 512 tokens:

| Packed prefill backend | Prefill P50 (ms) | TTFT P50 (ms) | Aggregate TPS P50 | Correctness |
|---|---:|---:|---:|---|
| Project Triton kernel before fused page writes | 218.2 | 219.1 | 29.58 | pass |
| Project Triton kernel + fused K/V page writes (`auto`) | 198.0 | 198.9 | 29.96 | pass |
| Fused SDPA ragged path (`sdpa`, explicit) | 132.6 | 133.5 | 31.31 | pass* |

The exact safe page-write optimization reduces prefill time by about 9% while
still processing 640 real prompt tokens instead of 1,024 padded positions.
The experimental SDPA path reduces prefill time by about 39% while still
processing 640 real prompt tokens instead of 1,024 padded positions (384
positions, or 37.5%, avoided). It passed the two-request mixed sample, but it does not pass
the entire exact-token reference corpus on this checkpoint, so it is not the
default. These numbers are for the static packed runner and should not be
presented as vLLM parity.

## What vLLM still does better

vLLM uses production attention backends such as FlashAttention and FlashInfer,
selected from its attention-backend registry. Its ragged/paged prefill kernels
receive cumulative query offsets and paged KV metadata and process the whole
ragged batch with tiled GPU kernels. The important differences are:

- our `sdpa` path has one attention call per request per layer; vLLM uses one
  fused varlen/paged attention operation for the batch;
- our educational Triton kernel launches one program per token/head and walks
  page tiles, while production kernels tile query and key blocks to reuse data
  and use tensor-core matrix multiplication;
- production engines use fused slot-mapping/cache-write kernels. We now have a
  fused scatter write, but it is still a separate kernel per layer rather than
  a production attention/cache-write fusion;
- our packed runner is a static batch and currently requires equal output
  limits, while vLLM's scheduler can admit, prefill, decode, and retire rows
  continuously;
- our CUDA Graph support is primarily a decode optimization, whereas a
  production engine manages a small set of reusable shapes and metadata for
  the serving workload.

The next meaningful TTFT optimization is therefore a true fused varlen
prefill backend, using the same `cu_seqlens`/page-table contract already
implemented here, followed by integration with the continuous-batching
scheduler.

## Architecture decisions

The allocator is independent of PyTorch tensors, so randomized ownership tests
can run on CPU without model weights. Physical storage is allocated once for a
configured block pool; request ownership is sparse within that pool. A final
partial block contributes internal fragmentation, but a request does not need
physically adjacent blocks.

The three model-facing paths are intentionally retained:

```text
fallback: page tables -> gather dense DynamicCache -> model forward
          model DynamicCache -> copy new suffix -> page blocks

direct:   Q/K/V projections -> write K/V to physical pages
          query + block table -> Triton or fallback online-softmax attention

hybrid:   optimized SDPA prefill -> one-time copy to physical pages
          direct block-table attention for decode
```

The direct path is paged attention semantically: the attention routine follows
each request's logical-to-physical block mapping and never gathers the whole
batch into a rectangular `DynamicCache`. Supported one-token CUDA decode and
packed CUDA prefill use fused Triton kernels. CPU execution and unsupported
shapes retain the readable Python/PyTorch fallback.

## Correctness tests

`tests/test_paged_kv_cache.py` covers:

- fragmented block allocation and logical-to-physical lookup;
- OOM rejection without partial allocation;
- randomized allocate/append/release sequences with no duplicate block IDs;
- variable-length left-padded gathers and attention masks;
- block-table tensor construction;
- conversion to `transformers.DynamicCache`;
- copying only new tokens from a dense cache;
- shape validation and release isolation.

`tests/test_paged_runner.py` compares the paged gather runner with an
independent manual greedy decode on a tiny deterministic model.
`tests/test_paged_attention.py` compares block-wise GQA attention with a dense
reference across fragmented pages and compares Triton with the PyTorch
reference on CUDA for block sizes 8, 16, 32, and 64.
`tests/test_direct_paged_runner.py` checks the complete Qwen3 direct and hybrid
paths against trusted manual generation.
`tests/test_packed_prefill.py` checks packed metadata and mixed-length
no-padding prefill against independent greedy generation.

## Benchmark configuration

The reproducible configuration is
`configs/workloads/qwen3_fp8_paged.yaml`. It pins the Qwen3 FP8 model and
sweeps block sizes 8, 16, 32, and 64 while holding total physical capacity at
32,768 token slots. The default compares the contiguous runner with optimized
prefill plus direct paged decode over 128-, 512-, and 2,048-token prompts and
batch sizes 1, 2, and 4. The all-blockwise reference can be selected with
`--modes paged_direct`; the gather control can be selected with
`--modes paged_gather`; `paged` remains its compatibility alias.

Run it on the L4 with:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_paged_kv \
  --config minillm_l4/configs/workloads/qwen3_fp8_paged.yaml \
  --workload all \
  --output-dir minillm_l4/results/paged_kv
```

The manifest records block capacity, used and reserved token slots, internal
fragmentation, physical KV bytes, allocation configuration, TTFT/TPOT, and
correctness. Raw request events remain JSONL.

The default model configuration selects `fp8_kernel_path: sm89`. Change it to
`auto` for a matched generic-Transformers control run.

## Before/after L4 result

The completion sweep used one NVIDIA L4, the pinned Qwen3 FP8 checkpoint, a
128-token prompt, 32 output tokens, batch size 1, greedy decoding, one warm-up,
three measured repetitions, and 4,096 physical token slots. Raw artifacts are
under `results/paged_completion_all_blocks/`.

| Backend | Block | TTFT P50 (ms) | TPOT P50 (ms) | Internal waste | Correctness |
|---|---:|---:|---:|---:|---|
| Contiguous DynamicCache | - | 75.522 | 72.785 | - | pass |
| SDPA + Triton paged decode | 8 | 88.065 | 76.762 | 0.625% | pass |
| SDPA + Triton paged decode | 16 | 82.664 | 75.868 | 0.625% | pass |
| SDPA + Triton paged decode | 32 | 77.778 | 74.890 | 0.625% | pass |
| SDPA + Triton paged decode | 64 | 76.545 | 74.689 | 17.188% | pass |

The matched pre-kernel PyTorch TPOT values were 254.395, 172.790, 129.095,
and 110.183 ms for block sizes 8, 16, 32, and 64. The fused kernel reduced
them by about 70%, 56%, 42%, and 32%, respectively. Final paged TPOT is within
roughly 3--5% of contiguous at this short batch-1 workload.

The hybrid path also fixes pathological all-blockwise prefill cost: prefill
model time remains about 74--77 ms and one-time page materialization is about
3--14 ms. The Triton implementation still traverses logical tokens inside one
kernel rather than launching once per page; page size remains a trade-off
between reduction work, indirection, and allocator fragmentation. Raw final
artifacts are under `results/paged_triton_final/`; the pre-kernel comparison is
under `results/paged_completion_all_blocks/`.

### L4 projection-kernel optimization

Profiling the batch-1 decode path showed 252 FP8 projection calls per token:
seven projections in each of 36 decoder layers. The generic fine-grained FP8
kernel accounted for roughly 72% of self CUDA time, while the model issued
about 1,820 GPU launches per token. This made projection dispatch the first
optimization target.

The matched validation used a 128-token prompt, 32 output tokens, batch size
1, one warm-up, three repetitions, and block size 32. Exact greedy token IDs
matched the existing reference.

| Backend | FP8 projection | TTFT P50 (ms) | TPOT P50 (ms) | Decode TPS P50 |
|---|---|---:|---:|---:|
| Contiguous | Transformers auto | 75.522 | 72.785 | 13.74 |
| Contiguous | MiniLLM SM89 | 57.123 | 56.091 | 18.40 |
| Paged hybrid | Transformers auto | 77.778 | 74.890 | 13.35 |
| Paged hybrid | MiniLLM SM89 | 62.713 | 60.572 | 16.51 |

The custom projection path reduced contiguous TPOT by 22.9% and raised decode
throughput by about 34%. Paged attention now adds about 4.5 ms/token at this
shape. Remaining batch-1 limits include the language-model-head GEMV, numerous
small elementwise kernels, and CPU/GPU launch overhead; maximizing utilization
is therefore not equivalent to minimizing single-request latency.

A separate telemetry-enabled run measured 55.238 ms P50 TPOT, 18.685 decode
tokens/s, and 43% P50 GPU utilization. The coarse utilization sample combines
memory stalls, launch gaps, and active computation; it is evidence that batch
1 leaves throughput capacity available, not evidence that all of that capacity
can be converted into lower single-request latency.

### CUDA Graph replay

`PagedCudaGraphBatchRunner` retains the direct paged-attention math while
removing per-operation CPU submission from steady-state decode. It reserves a
request's pages up front so every KV address is stable, keeps the block table
and input/output tensors alive, and captures one reusable graph for the fixed
active batch. Before each replay, the host updates the static input token,
position IDs, and sequence-length buffers. The device-side K/V writer and
attention kernel read those values to select the current logical page slot.
Prefill is eager but uses the project-owned flat packed-token path by default,
so it writes K/V directly into pages without a dense-to-paged copy. A dense
Transformers prefill remains available with
`--graph-prefill-backend dense` as an explicit fallback for comparison.
A graph includes the 36-layer one-token model forward, paged attention, KV
writes, language-model head, and greedy argmax.

The clean asynchronous validation used a 128-token prompt, 32 output tokens,
block size 32, one capture warm-up, and three measured repetitions.
Synchronized component tracing was disabled for these headline latency
numbers.

| Metric | Packed eager direct | Reusable CUDA Graph | Change |
|---|---:|---:|---:|
| TTFT P50 | 59.072 ms | 57.714 ms | -2.3% |
| ITL P50 | 59.305 ms | 22.315 ms | -62.4% |
| ITL P95 | 60.375 ms | 22.360 ms | -63.0% |
| TPOT P50 | 59.350 ms | 21.701 ms | -63.4% |
| Decode TPS P50 | 17.363 | 46.036 | +165.1% |
| E2E latency P50 | 1902.122 ms | 752.821 ms | -60.4% |
| E2E TPS P50 | 16.823 | 42.507 | +152.7% |
| GPU utilization P50 | 39.5% | 86.0% | +46.5 points |

| Variance statistic | Paged eager | Paged CUDA Graph |
|---|---:|---:|
| TPOT standard deviation | 0.061 ms | 0.005 ms |
| TPOT coefficient of variation | 0.10% | 0.02% |
| ITL standard deviation | 0.519 ms | 3.383 ms* |
| Decode TPS standard deviation | 0.018 | 0.010 |
| E2E latency standard deviation | 2.051 ms | 0.739 ms |

All three measured repetitions were stable and matched the trusted token IDs
exactly for batch sizes 1, 2, and 4. Capturing one reusable graph took about
55--89 ms in the warm-up and is excluded from steady-state metrics. The clean
batch-1 artifacts are under `results/async_vs_graph/`; the graph batch stress
artifacts are under `results/graph_batch_stress_packed_prefill/`.

The graph batch stress run measured 46.04, 44.94, and 41.31 decode tokens/s at
batch sizes 1, 2, and 4, respectively, with exact correctness. The lower
per-request rate at larger batches is expected because the graph executes more
rows per iteration and the reported request-level metric includes each row's
latency. *The graph ITL standard deviation includes the first replay interval;
steady-state replay intervals are approximately 22--24 ms.*

### Long-context prefill optimization

The long-context workload used 3,072-token prompts with 64 generated tokens and
4,608-token prompts with 32 generated tokens. The original packed Triton
prefill path was correct but took 3.7--16.0 seconds for these cases because its
educational page-walking attention kernel was not sufficiently tiled. A dense
SDPA control reduced that cost to 0.7--2.4 seconds. The graph runner now accepts
`graph_prefill_backend: auto`; because its static graph batches require equal
prompt lengths, `auto` selects dense SDPA without padding and keeps the packed
path available explicitly for comparison.

The dense-to-paged KV materialization was then changed from a nested
request/layer loop to one batched indexed scatter per layer. Exact greedy token
IDs still match the saved reference corpus:

| Prompt / batch | Dense control TTFT P50 | Optimized TTFT P50 | Improvement | Optimized TPOT P50 |
|---|---:|---:|---:|---:|
| 3,072 / 1 | 712.2 ms | 646.6 ms | 9.2% | 26.91 ms |
| 3,072 / 2 | 1,534.7 ms | 1,393.1 ms | 9.2% | 29.70 ms |
| 4,608 / 1 | 1,183.2 ms | 1,082.5 ms | 8.5% | 28.02 ms |
| 4,608 / 2 | 2,433.5 ms | 2,210.2 ms | 9.2% | 33.21 ms |

In the traced 3,072-token batch-1 run, page materialization fell from about
73.4 ms to 2.1 ms; the model forward remained about 641.5 ms. The optimized
artifacts are under `results/long_context_optimized_20260920/`. The model
forward is already resolved to PyTorch SDPA, so closing the remaining gap to
vLLM requires a fused varlen/paged prefill attention kernel rather than another
cache-copy optimization.

### Batch-size scaling at long context

The extended corpus was expanded to eight deterministic samples per prompt
length so batches 2, 4, and 8 could be measured. This sweep used block size 32,
`graph_prefill_backend: auto`, a 49,152-token physical capacity, one warm-up,
two measured repetitions, and exact correctness references. MiniLLM-L4 results
are under `results/long_context_batch248_20260920/`; matched vLLM results are
under `results/vllm_long_context_batch248_20260920/`.

The following are request-level P50 metrics. `TPS` means decode tokens/s for
one request; aggregate batch throughput is reported separately in each raw
manifest because vLLM's console `TPS` label uses a different aggregate
definition.

| Prompt | Batch | Mini TTFT (ms) | vLLM TTFT (ms) | Mini TPOT (ms) | vLLM TPOT (ms) | Mini decode TPS | vLLM decode TPS | Mini E2E TPS | vLLM E2E TPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3,072 | 2 | 1,435.7 | 633.8 | 30.35 | 29.09 | 33.47 | 35.35 | 19.12 | 25.94 |
| 3,072 | 4 | 2,899.3 | 1,704.9 | 37.25 | 29.67 | 27.27 | 34.24 | 12.20 | 17.91 |
| 3,072 | 8 | 5,805.6 | 3,361.2 | 51.85 | 37.10 | 19.59 | 27.38 | 7.05 | 11.24 |
| 4,608 | 2 | 2,266.3 | 1,007.3 | 34.10 | 38.04 | 30.27 | 29.31 | 9.63 | 14.62 |
| 4,608 | 4 | 4,626.8 | 2,633.3 | 46.28 | 33.25 | 22.30 | 31.05 | 5.28 | 8.73 |
| 4,608 | 8 | 9,231.7 | 5,211.0 | 69.58 | 44.30 | 14.83 | 23.30 | 2.81 | 4.86 |

All twelve cases passed exact greedy-token validation. MiniLLM-L4 peak
allocated memory was approximately 16.1 GiB for 3,072/batch-8 and 18.7 GiB
for 4,608/batch-8; the largest reserved value was approximately 21.4 GiB.
Batching increases GPU utilization, but this implementation still has a
prefill cost that scales almost linearly with total prompt tokens. vLLM's fused
varlen prefill and production decode kernels retain lower latency at the same
batch sizes.

### Profile-guided long-context decode tuning

A synchronized 3,072-token, batch-8 trace showed that model execution consumed
more than 99% of the request window. The original paged-attention kernel used a
16-token reduction tile and four split-KV partitions. A dedicated microbenchmark
then swept reduction tiles 8/16/32/64, split counts 1/2/4/8, and batch sizes
1/2/4/8. The accepted L4 configuration uses a 64-token tile for contexts of at
least 1,024 tokens and a batch-aware split policy. In the 2,049–4,096-token
range, batch 8 uses one split while smaller tested batches use eight; contexts
above 4,096 use eight splits. The graph records the selected tile and split
count in every result artifact.

The GQA-reuse hypothesis was also implemented and tested. It computes all four
Qwen query heads sharing one KV head in one Triton program, but the larger live
state reduced occupancy. At 3,072 tokens and batch 8, it increased TPOT from
about 51.9 ms to 58.5 ms. It remains available to the kernel microbenchmark as
an experimental comparison, while the production graph path explicitly uses
the faster per-query-head implementation.

| Prompt / batch | Before TPOT P50 | Tuned TPOT P50 | Improvement | vLLM TPOT P50 |
|---|---:|---:|---:|---:|
| 3,072 / 8 | 51.85 ms | 39.48 ms | 23.9% | 37.10 ms |
| 4,608 / 8 | 69.58 ms | 47.87 ms | 31.2% | 44.30 ms |

Both end-to-end runs passed exact greedy-token validation. At 3,072/batch-8,
CUDA Graph replay fell from about 3.25 seconds to 2.47 seconds for 63 decode
steps. At 4,608/batch-8, it fell to 1.48 seconds for 31 decode steps. The tuned
artifacts are under `results/tuned_attention_20260920/`, and the reusable sweep
command is `benchmarks.commands.benchmark_paged_attention`.

A matched follow-up isolated split count at 3,072 prompt tokens, 64 output
tokens, batch 8, with one warm-up and two measured repetitions. Reducing the
decode split count from eight to one lowered TPOT P50 from 39.46 ms to 38.80 ms
(1.7%) and raised decode TPS P50 from 25.74 to 26.18 (1.7%). TTFT stayed about
5.8 s because this decode-only change does not accelerate prefill. Both runs
passed exact greedy-token checks. Results are in
`results/decode_tune_batch8_eight_split/` and
`results/decode_tune_batch8_one_split/`.

TTFT did not improve: prefill remained about 5.8 seconds and 9.2 seconds in the
two batch-8 cases. A clean 3,072/batch-8 experiment rejected multi-output FP8
gate/up and K/V fusion for prefill because it increased prefill from 5,809 ms
to 6,473 ms. The extra accumulator state helps some decode shapes but hurts
large-prefill occupancy. Therefore the fusion was not integrated into this
engine. The next TTFT optimization needs a separately tuned prefill kernel or
compiled fused model path rather than reusing a decode-oriented kernel.

## Component latency trace

The paged benchmark accepts `--trace-summary` and writes the complete trace to
the result JSON:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_paged_kv \
  --config minillm_l4/configs/workloads/qwen3_fp8_paged.yaml \
  --workload short \
  --modes paged_packed \
  --block-sizes 32 \
  --batch-sizes 1 \
  --repetitions 3 \
  --warmup-repetitions 1 \
  --trace-summary \
  --output-dir minillm_l4/results/diagnostics_packed
```

For the measured eager packed run, the trace window was 1,871.7 ms. Decode
model forward accounted for 1,797.0 ms, or 96.2% of recorded component time;
metadata materialization was 4.5 ms, token selection 3.0 ms, streaming 2.3
ms, and scheduler/KV reservation 0.6 ms. This establishes that the 16--18
decode-TPS result is not caused by Python queueing or allocator overhead.

For the current packed-prefill graph trace, graph replay accounted for 689.7 ms
of a 756.5 ms trace window, or 91.6%, with 22.4 ms diagnostic TPOT. Packed
prefill accounted for 59.6 ms, or 7.9%; input updates were 1.0 ms, replayed
token transfer/streaming was 1.9 ms, and sampling plus output materialization
was below 0.2 ms. GPU utilization was materially higher because graph replay
submits the many small one-token operations as a single captured execution
schedule. Diagnostic tracing synchronizes around these stages, so its latency
is intentionally separate from the clean asynchronous table above. The trace
artifact is under `results/trace_graph_packed/`.

## 2026-09-30 follow-up API and hybrid-numerics fixes

The follow-up review reproduced four missing API protections on the L4:

| Case | Before | Fix |
|---|---|---|
| Earlier query in the Qwen adapter, position 1 with KV length 4 | Fast writer overwrote position 3; `[1, 2, 100, 200]` became `[1, 2, 100, 7]` | Pass the live query position into the writer; now `[1, 7, 100, 200]` |
| Strided sequence lengths / block-table columns | Wrong row length or page; examples returned 4 instead of 15, or 1.5 instead of 76 | Pass both metadata strides to all four attention kernels and the KV writer |
| CUDA query position -1 / beyond the live KV span | NaN / silently accepted, unlike the host-position path | Eager range validation before launch; live graph device assertions |
| Split limit 2 for a four-token attention read | Silently omitted the final two tokens | Reject limits that do not cover the actual causal read span |

The writer additionally uses independent key/value source strides. Validation
checks integer metadata on the correct device, positive lengths within the
reserved/table capacity, and valid live physical block IDs. It happens once
before layer-zero decode writes, rather than synchronizing each model layer.
Unused table columns may remain `-1`. Graph capture performs no device-to-host
copies; captured device assertions validate each replay's current metadata.
An invalid replay assertion invalidates the CUDA context, so the serving
boundary must continue to reject invalid request metadata before replay.

### Hybrid greedy mismatch: backend numerical policy

Identical dense queries and dense KV tensors were replayed through SDPA,
FP64, and both paged attention implementations at the original failing step.
The FP32 Triton reductions were consistently closer to FP64 than the dense
BF16 SDPA output, but were not numerically identical to SDPA. This is not an
incorrect prefill copy or a causal mask problem. Unnormalized softmax weight
rounding and online-softmax tile traversal substantially explain the backend
gap; small attention differences propagate through BF16/FP8 model operations
and can change greedy logits.

An explicit `sdpa_compat` policy retains direct page reads and FP32 Q*K
products, but traverses 128-token tiles in reverse, rounds unnormalized P to
the activation dtype before P*V, and uses one KV split. Auto selects it for
the pinned BF16 SDPA dense-prefill hybrid/graph paths. `accurate` retains the
existing FP32 reductions and split-KV selection; packed paths remain on that
policy by default. No expected token IDs were changed and no numerical test
was removed. This is tested compatibility for the pinned model/L4, not a
universal bitwise SDPA emulator. The compatibility mode sacrifices split-KV
parallelism; its performance is recorded separately.

`tests/test_decode_metadata.py` covers the reproduced defects, independent
source strides, invalid lengths/IDs/dtypes, valid short causal split spans,
and mutable graph positions. Opt-in `tests/test_paged_model_integration.py`
loads the pinned checkpoint, generates its trusted dense reference, anchors
the original `" two"` token, and checks all block sizes 8/16/32/64 at batches
1/2/4. It also tests dense-prefill graph replay on the actual failing request.

Initial validation: all 228 then-present CPU/GPU/real-model tests passed.
The matched 128/32, block-16, batch-1 smoke run now passes the unchanged
reference: contiguous TTFT/TPOT 57.44/55.79 ms; hybrid 60.60/60.03 ms. The
previous matched smoke was 57.26/55.92 and 61.15/59.88 ms respectively, with
the hybrid correctness failure. These single-repetition numbers are a
diagnostic before/after, not a statistically established speedup.

Raw artifacts: `results/phase6_fixes_20260930/hybrid_compat/`,
`precision/`, `precision_online/`, and `precision_final/`. The reproducible
`scripts/inspect_decode_attention.py` compares identical real Q/K/V inputs
and saves per-layer attention and projection errors. On the 36 captured
layers, the average absolute attention difference from SDPA fell from
9.48e-5 (`accurate`) to 1.46e-5 (`sdpa_compat`), approximately 6.5 times
smaller. This is a numerical comparison on one identical-input diagnostic,
not a latency improvement or a general bitwise-equivalence claim.

### Final validation and sign-off

The final CPU/GPU suite, including the pinned-model integration tests,
passes **233 tests with no skips**. An isolated child CUDA context also
confirms that a negative query position supplied *after capture* is rejected
on replay; it does not poison the main test process. No failing test was
removed, no expected token was changed, and no reference file was regenerated.

The repeated matrix passes all **27/27 points**: contiguous, hybrid, and
dense-prefill graph paths; 128/32, 512/64, and 2,048/128 prompt/output lengths;
batches 1/2/4; block size 16; 32,768 physical token slots. Every point uses one
warm-up and three measured repetitions. GPU/system sampling was disabled;
these runs do not establish a GPU-utilization or peak-VRAM comparison.
Every result records model revision, FP8 projections/BF16 activations,
software/hardware identity, parent commit SHA, and dirty-worktree status.

Current P50 measurements in milliseconds (not a before/after speedup table):

| Prompt / output | Batch | Contiguous TPOT | Hybrid TPOT | Graph TTFT | Graph TPOT |
|---|---:|---:|---:|---:|---:|
| 128 / 32 | 1 | 56.03 | 60.50 | 59.52 | 21.95 |
| 128 / 32 | 2 | 57.22 | 60.62 | 59.98 | 22.38 |
| 128 / 32 | 4 | 56.77 | 60.84 | 92.49 | 22.87 |
| 512 / 64 | 1 | 56.60 | 60.48 | 94.88 | 22.78 |
| 512 / 64 | 2 | 57.00 | 61.07 | 192.47 | 23.24 |
| 512 / 64 | 4 | 56.44 | 60.75 | 406.59 | 24.03 |
| 2,048 / 128 | 1 | 56.79 | 60.36 | 423.42 | 24.71 |
| 2,048 / 128 | 2 | 56.95 | 60.77 | 926.91 | 25.90 |
| 2,048 / 128 | 4 | 56.24 | 60.68 | 1,910.85 | 28.28 |

The graph runner measures one stable request group per run (1/2/4 requests),
whereas contiguous/hybrid run all four workload requests. The table compares
request-level latency, not matched whole-run throughput. The manifest
contains both request-level rates and aggregate completed-run throughput;
these must not be conflated. All final points pass exact-token checks and
repeat-stability checks against the existing Phase 1 reference corpus.

The unchanged packed-prefill numerical policy is additionally validated at
batches 1 and 4, for both eager packed and packed-prefill graph execution,
with three repetitions: **4/4 points pass**, for **31/31 total passing points**.
The graph P50 TPOT in these packed controls is 22.45/23.32 ms (batch 1/4).

Reproduction:

```bash
MINILLM_RUN_MODEL_TESTS=1 .conda-env/bin/python -m pytest minillm_l4/tests -q
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_paged_kv \
  --config minillm_l4/configs/workloads/qwen3_fp8_paged.yaml \
  --workload all --modes contiguous paged_hybrid paged_graph \
  --block-sizes 16 --batch-sizes 1 2 4 --graph-prefill-backend dense \
  --repetitions 3 --warmup-repetitions 1 \
  --no-gpu-sampling --no-system-telemetry \
  --output-dir minillm_l4/results/phase6_fixes_20260930/validation
```

Full raw JSON, token events, P50/P90/P95/P99, and repeated-run variance are
saved under [`validation`](../../results/phase6_fixes_20260930/validation/paged_kv_manifest.json)
and [`packed_validation`](../../results/phase6_fixes_20260930/packed_validation/paged_kv_manifest.json).
The pinned-model regression separately covers all block sizes 8/16/32/64 at
batches 1/2/4, including the formerly failing request's full rollout.

Phase 6's recorded definition of done is now satisfied for the validated
pinned-model corpus and default numerical policies. `--decode-numerics
accurate` deliberately preserves the earlier comparison implementation and
may still differ from BF16 SDPA greedy tokens. Broader prompts/backends are
not claimed to be universally token-identical. Chunked prefill and
general dynamic-shape graph scheduling remain later-phase work. Historical
results below describe the previous implementation, not the current default.

## 2026-09-28 correctness investigation and memory quantification

### 2026-09-30 causal-position dispatch fix

The public paged attention API previously ignored `query_start_positions` on
its Triton path and used the full supplied KV length, exposing future tokens
when a one-token query addressed an earlier position. All four Triton decode
kernels now limit page loads and online softmax to
`min(sequence_length, query_position + 1)` for causal attention. Noncausal
attention retains the full KV span. CUDA device positions are read by the
kernel at execution time; the graph runner passes its replay-updated position
buffer rather than a fixed Python position. Static paged runners reuse their
existing device position tensors to avoid extra per-layer host transfers.

The GPU regression uses values `[1, 2, 100, 200]` and a query at position 1:
the old Triton path returned 76, while the corrected path returns exactly 1.5
from positions 0 and 1. Coverage includes mixed positions within a batch,
host and strided device position inputs, automatic and explicit Triton
dispatch, grouped-query and split-KV kernels, noncausal attention, and graph
replay after changing the query position. This addresses a separate API bug;
the hybrid greedy-token numerical mismatch documented below remains open.

Validation on the L4: the complete CPU/GPU suite passes (`190 passed`, no
skips). A matched short-prompt run (128 input / 32 output tokens, block size
16, batch 1, one warm-up and one measured repetition) passes exact token
checks for contiguous, packed-paged, and CUDA Graph paths. The hybrid path
retains exactly its previous rollout, including the known token-17 mismatch,
so the benchmark command exits nonzero for that pre-existing failure. Raw
results and events are in
[`causal_position_fix_20260930`](../../results/causal_position_fix_20260930/paged_kv_manifest.json).

### Definition of done (historical sweep, updated sign-off)

| Criterion | Evidence | Status |
|---|---|---|
| Randomized allocation/free tests pass | `tests/test_paged_kv_cache.py` (randomized allocate/append/release, no duplicate block IDs) | Done |
| Blocks never leak or alias incorrectly | same tests; every sweep run releases all blocks | Done |
| Out-of-memory behavior is controlled | OOM rejection without partial allocation (tests); admission guard in the capacity experiment | Done |
| Generation remains correct | Historical hybrid failures are described below. The 2026-09-30 default policy passes 31/31 repeated benchmark points, plus the all-block-size pinned-model regression; see final validation above | Done for validated corpus/default policies |
| Memory-utilization effect is quantified | full L4 sweep below plus the allocator capacity experiment | Done |

### The prefix-002 mismatch is an exact tie, not a paged bug

The Phase 7 continuous trace reported that paged decode differs from dense
manual decoding for request `prefix-002` (528-token prompt, 0% reuse) at its
fourth generated token, even with one active request and the PyTorch paged
backend. It was reproduced with Phase 6 code only (dense SDPA prefill, pages
built by `_make_paged_cache`, direct paged decode; `sm89` projections as in the
trace) and each path's step-3 logits were compared for the two competing
tokens:

| Path | Logit " photos" (7249) | Logit " creating" (6825) | Continuation |
|---|---:|---:|---|
| Dense bf16 SDPA (the reference path) | 9.5000 | 9.5000 | " the process of creating a new version of" |
| Paged, PyTorch backend | 9.5000 | 9.3125 | " the process of photosynthesis works in plants" |
| Paged, Triton backend | 9.3125 | 9.3750 | " the process of creating a new version of" |
| Dense with fp64 attention | 9.8750 | 9.4375 | " the process of photosynthesis works in plants" |

In the reference path the two logits are exactly equal in BF16, so its choice
is made by `argmax` tie-breaking (the lower token ID wins). The paged paths
move those logits by at most 0.19, less than two BF16 steps at this magnitude,
and either choice is a fluent continuation of "Explain how". The most precise
computation (attention in FP64) prefers " photos" by 0.44, as the PyTorch
paged path does. Once the tokens are the same, the paths agree again: teacher-
forced with identical inputs, every path continues " photos" with
"ynthesis".

Two measurement mistakes were made and corrected while establishing this. A
first script printed each path's chosen token from `topk`, which orders an
exact tie arbitrarily, while the model was fed the `argmax` token; this made a
dense run that had been fed " creating" look as if it had been fed " photos"
and produced a spurious "33-logit" divergence at the next step. The analysis
above uses the `argmax` token throughout. Per-layer comparisons (dense vs paged
hidden states, flash vs fp64 attention on captured inputs, six repeated runs,
and the autotuned FP8 kernel's chosen configurations across four processes)
found no kernel error and no run-to-run nondeterminism.

### Triton decode precision fix

The 128/32 batch-2 hybrid point failed exact-token on `baseline-short-000`
(output token 25: " two" instead of " three"), while batch 1 and 4 passed. With
identical inputs, the Triton decode kernel raised the " two" logit by about 7
relative to the PyTorch paged path. A kernel-level comparison against FP64
attention on real Qwen3 q/k/v from all 36 layers showed the Triton kernel was
2–5× less accurate than the PyTorch paged reference (worst error 0.69 vs 0.15),
identically at batch 1 and 2, so this was precision, not batch indexing.

The kernels computed `tl.sum(keys * query)` with both operands in BF16, so every
Q·K product was rounded to BF16 before the sum; Qwen3's layer-0 keys reach
about 290. All four decode kernels (plain, split-KV, and their GQA variants)
and the packed-prefill kernel now upcast Q and K to FP32 before multiplying.

| Kernel | Worst error vs FP64 over 36 real layers |
|---|---:|
| Triton decode, BF16 products (before) | 0.690 |
| **Triton decode, FP32 products (after)** | **0.059** |
| PyTorch paged reference (scores rounded to BF16 by design) | 0.148 |

`test_triton_decode_is_at_least_as_accurate_as_torch_with_large_keys` fails
on the previous kernel and passes on the fixed one. The change costs nothing
measurable: hybrid TPOT is unchanged within noise, and packed-prefill TTFT
moved 59 → 61 ms, 226 → 232 ms, and 2,546 → 2,585 ms (128/512/2,048 tokens,
batch 1). It also fixes a packed-path failure: on the previous kernel the
128/32 batch-1 packed point failed exact-token; after the fix all nine packed
points pass.

### Full L4 sweep after the fix

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_paged_kv \
  --config minillm_l4/configs/workloads/qwen3_fp8_paged.yaml \
  --workload all --output-dir minillm_l4/results/paged_kv_20260928_fp32qk
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_paged_kv \
  --config minillm_l4/configs/workloads/qwen3_fp8_paged.yaml \
  --workload all --modes paged_packed --block-sizes 16 \
  --output-dir minillm_l4/results/paged_packed_20260928_fp32qk
```

One warm-up and three measured repetitions per point, `sm89` projections,
32,768 physical token slots. Provenance: commit `86d5855` with this change and
unrelated uncommitted Phase 7 prefix-sharing bookkeeping in the worktree
(allocator reference counts and write guards that are inactive without shared
blocks).

| Shape | Batch | Contiguous TPOT | Paged TPOT (b8–b64) | Contiguous TTFT | Paged TTFT b8 / b64 | Paged exact-token |
|---|---:|---:|---:|---:|---:|---|
| 128/32 | 1 | 55.1 ms | 58.4–59.2 ms | 56 ms | 68 / 57 ms | fail (4/4) |
| 128/32 | 2 | 55.9 ms | 58.3–58.5 ms | 59 ms | 81 / 60 ms | fail (4/4) |
| 128/32 | 4 | 56.2 ms | 58.4–58.8 ms | 88 ms | 127 / 91 ms | fail (4/4) |
| 512/64 | 1 | 56.1 ms | 58.1–59.1 ms | 88 ms | 128 / 92 ms | pass |
| 512/64 | 2 | 56.2 ms | 58.7–59.6 ms | 182 ms | 268 / 188 ms | pass |
| 512/64 | 4 | 56.7 ms | 59.2–60.0 ms | 392 ms | 576 / 406 ms | pass |
| 2,048/128 | 1 | 55.5 ms | 60.9–61.0 ms | 408 ms | 592 / 417 ms | pass |
| 2,048/128 | 2 | 56.1 ms | 60.4–61.0 ms | 899 ms | 1,293 / 938 ms | pass |
| 2,048/128 | 4 | 55.9 ms | 61.4–62.3 ms | 1,884 ms | 2,683 / 1,983 ms | pass |

Paged decode is 4–11% slower than contiguous per token (decode is host-bound,
and the paged path adds per-layer host work). Paged TTFT includes copying the
prompt KV into pages; small blocks cost the most (up to +45% at block 8),
while block 64 is within 2–5% of contiguous. Internal fragmentation is
0.05–0.62% at blocks 8–32 and 17.2% only for 159-token sequences at block 64.
The paged pool is preallocated (32,768 slots × 144 KiB per token = 4.5 GiB),
so peak VRAM is not compared across modes; contiguous KV holds exactly the
live tokens (for example 8,700 tokens ≈ 1.2 GiB at 2,048/128 batch 4).

### Historical open case: `baseline-short-003`, output index 16

This means the seventeenth generated token. The new `sdpa_compat` policy
addresses this case; the table below describes the older `accurate` policy.

After the fix, every hybrid 128/32 point fails on the same request and token
at every batch and block size: the reference emits " two" and the paged path
emits " explicit". Unlike prefix-002 this is not a reference tie:

| Path (teacher-forced to step 16) | " two" | " explicit" | Picks |
|---|---:|---:|---|
| Dense BF16 (reference) | 22.13 | 20.25 | two (by 1.9) |
| Dense with FP64 attention | 21.75 | 18.63 | two (by 3.1) |
| Paged, PyTorch backend | 21.88 | 21.50 | two (by 0.4) |
| Paged, Triton backend | 21.88 | 22.00 | explicit (by 0.1) |

Both paged backends raise " explicit" by 1.3–1.8 logits relative to dense decode
at this step, while the higher-precision reference agrees with the BF16
reference. The packed path, which prefills with the project's Triton kernel
instead of BF16 flash SDPA, passes this request. The root cause is not yet
identified, so the "generation remains correct" criterion is recorded as open
for this case rather than waived.

### Allocator capacity under a fixed budget

`run_kv_capacity` is a CPU-only experiment on the real `PagedKvAllocator`
with the Phase 6 request shapes (128/32, 512/64, 2,048/128; final lengths 159,
575, and 2,175 KV positions) and a 32,768-slot budget:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_kv_capacity \
  --output minillm_l4/results/kv_capacity_20260927/kv_capacity.json
```

Static capacity, admitting a seeded mix of requests at their final length
until the first one does not fit:

| Policy | Requests admitted | Internal waste | Budget holding real tokens |
|---|---:|---:|---:|
| Contiguous, reserve each request's final length | 25 | 0% | 94.8% |
| Contiguous, reserve the longest length (2,175) for every request | 15 | 44.3% | 55.4% |
| Paged, block 8 / 16 / 32 | 25 | 0.08% | 94.8% |
| Paged, block 64 | 25 | 0.69% | 94.8% |

Churn, 600 requests in FIFO order, each growing one token per decode step and
releasing on completion. Paged admission holds only tokens that exist but
admits a request only if every active request can still grow to its final
length, so nothing is preempted. Contiguous admission must place each request's
full reservation in one first-fit region:

| Policy | Decode steps | Mean active requests | Budget holding real tokens | Steps head-of-line blocked by external fragmentation |
|---|---:|---:|---:|---:|
| Contiguous, longest-length reservation | 3,040 | 14.74 | 62.2% | 0 |
| Contiguous, final-length reservation, first fit | 2,176 | 20.59 | 86.9% | 1,280 (59%) |
| Paged, any block size 8–64 | 2,080 | 21.54 | 90.9% | 0 |

Paging fits 1.67× more requests than reserving the longest length per request
and finishes the churn backlog 1.46× faster. Against contiguous reservation of
exact final lengths, paging's advantage is removing external fragmentation:
the contiguous allocator had enough total free space but no single large
enough hole in 59% of steps, which cost 4.4% in completion time. Larger blocks
only change reserved (not used) space: 91.1% of the budget at block 8 versus
93.1% at block 64. The gain over exact contiguous reservation is modest here
because admission guarantees completion; engines that overcommit and preempt
(as vLLM does) recover more.

## Known limitations

- Unsupported devices, dtypes, and multi-token direct-attention shapes fall
  back to the readable PyTorch path.
- The packed `sdpa` backend avoids prompt padding but launches one fused
  attention call per request per layer; it is not yet a single fused ragged
  FlashAttention/FlashInfer kernel and is experimental because accumulation
  order can change close greedy-token decisions.
- The packed runner currently requires equal output limits inside one static
  batch; prompt lengths may differ. Variable output limits belong in the
  lifecycle scheduler, where completed rows can leave the active batch.
- The hybrid path performs a one-time dense-to-paged copy after prefill.
- The focused completion sweep covers every configured block size at batch 1;
  larger prompt and batch stress sweeps remain available through the same
  command but are not used to claim production parity.
- Eager physical cache tensors are created per benchmark invocation; the graph
  runner retains its cache and captured addresses across repetitions.
- The SM89 projection kernel is intentionally specific to the L4 and this
  checkpoint's 128x128 FP8 scaling layout. Other devices and layouts fall back.
- The reusable graph path captures one graph per fixed active shape, but it
  still requires stable request IDs and pre-reserved page addresses. A general
  continuous scheduler needs a small graph-shape pool plus dynamic slot
  mappings.
- The graph path reserves all request blocks up front and currently requires
  equal request shapes and stable request IDs across replays.

## Entry conditions for prefix caching

Satisfied: every configured block size passed exact-token validation on the
L4; allocator randomization, controlled OOM, fragmented lookup, block release,
and zero-active-request cleanup tests pass. Prefix caching can now build on
immutable shared physical blocks without changing the attention interface.

## 2026-10-07 packed SDPA exactness and FP8 autotune buckets

**Packed SDPA is now exact.** `_sdpa_packed_prefill_attention` repeated the
GQA K/V heads before calling SDPA, matching an older Transformers dispatch.
Transformers 5.14 calls SDPA for an unpadded prompt with no mask,
`is_causal=True`, and `enable_gqa=True`, which selects a different kernel. The
backend now delegates each request to Transformers' own
`sdpa_attention_forward`. On the L4, packed prefill of mixed lengths (128,
512, 1024, 2048, 4096) produces KV bitwise identical to single-request HF
prefill in every layer. `qwen3_packed_prefill(row_logits=True)` projects each
final hidden state separately, because a multi-row `lm_head` GEMM rounds one
bf16 step differently from the one-row projection.

**FP8 autotuning moved out of timed requests.** The SM89 kernel's autotune key
was the exact row count `M`. Continuous batching and flattened prefill keep
producing new totals, and a staggered-arrival run measured 180 autotune
searches inside the timed window (a 2048-token prefill took 7.8 s instead of
0.5 s). The key is now the power-of-two bucket of `M` with
`cache_results=True`, and `load_qwen_fp8(..., pretune_tokens=16384)` tunes
every bucket at load (about 90 s once per machine, then ~0.4 s from the disk
cache). The kernel's K reduction order does not depend on the tile config, so
outputs are unchanged. Earlier sweep timings whose shapes were new to their
warm-up may include autotune time.
