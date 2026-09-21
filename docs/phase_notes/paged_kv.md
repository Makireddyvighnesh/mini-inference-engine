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
  Decode uses a fixed 16-token reduction tile while storage page size remains
  independently configurable. Keeping those dimensions separate stabilizes
  online-softmax reduction order and keeps greedy-token validation stable when
  physical block sizes are swept.
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
least 1,024 tokens and a batch-aware split policy. Contexts above 2,048 tokens
use eight splits. The graph records the selected tile and split count in every
result artifact.

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
