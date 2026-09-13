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
- `triton_paged_decode_attention` fuses block-table lookup, physical K/V page
  loads, grouped-query head mapping, QK reduction, online softmax, and weighted
  value accumulation into one GPU launch per model layer.
- The Triton kernel uses fixed 16-token compute tiles independent of physical
  page size. This keeps numerical reduction order stable across block sizes
  and separates compute tuning from memory-fragmentation tuning.
- `sm89_fp8_linear` replaces the generic Transformers projection dispatcher
  on the L4. It fuses dynamic per-block activation scaling, FP8 conversion,
  block-scaled matrix multiplication, and FP32 accumulation. Unsupported
  tensors retain the original Transformers fallback.
- `PagedKvBatchRunner` remains as the dense-gather fallback/control path.
- Benchmark artifacts separate prefill model time, prefill cache
  materialization, decode model time, and logical page visits.

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
batch into a rectangular `DynamicCache`. Supported one-token CUDA decode uses
the fused Triton kernel. Multi-token reference prefill, CPU execution, and
unsupported shapes retain the readable Python/PyTorch fallback.

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
3--14 ms. Unlike the Python fallback, Triton latency is nearly independent of
physical page size because its 16-token compute tile traverses logical tokens
inside one kernel. Physical block size therefore primarily controls allocator
fragmentation instead of Python dispatch count. Raw final artifacts are under
`results/paged_triton_final/`; the pre-kernel comparison is under
`results/paged_completion_all_blocks/`.

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
and input/output tensors alive, and captures one graph for each logical decode
position. Prefill remains eager. A graph includes the 36-layer one-token model
forward, paged attention, KV writes, language-model head, and greedy argmax.

The final matched run used a 128-token prompt, 32 output tokens, batch 1, block
size 32, one capture warm-up, ten measured repetitions, and GPU telemetry.

| Metric | Paged eager | Paged CUDA Graph | Change |
|---|---:|---:|---:|
| TTFT P50 | 63.236 ms | 59.671 ms | -5.6% |
| ITL P50 | 59.493 ms | 22.480 ms | -62.2% |
| ITL P95 | 60.934 ms | 23.117 ms | -62.1% |
| TPOT P50 | 59.606 ms | 22.466 ms | -62.3% |
| Decode TPS P50 | 17.316 | 45.936 | +165.3% |
| E2E latency P50 | 1911.359 ms | 756.158 ms | -60.4% |
| E2E TPS P50 | 16.742 | 42.319 | +152.8% |
| GPU utilization P50 | 41% | 99% | +58 points |

| Variance statistic | Paged eager | Paged CUDA Graph |
|---|---:|---:|
| TPOT standard deviation | 0.264 ms | 0.177 ms |
| TPOT coefficient of variation | 0.44% | 0.78% |
| ITL standard deviation | 0.755 ms | 0.224 ms |
| Decode TPS standard deviation | 0.077 | 0.353 |
| E2E latency standard deviation | 8.323 ms | 6.172 ms |

All ten graph repetitions were stable and matched the trusted token IDs
exactly. Capturing 31 decode-position graphs took 1.900 seconds and is excluded
from steady-state metrics. Raw artifacts are under `results/paged_graph_final/`;
the matched three-path run is under `results/batch1_graph_final/`.

## Known limitations

- Unsupported devices, dtypes, and multi-token direct-attention shapes fall
  back to the readable PyTorch path.
- The hybrid path performs a one-time dense-to-paged copy after prefill.
- The focused completion sweep covers every configured block size at batch 1;
  larger prompt and batch stress sweeps remain available through the same
  command but are not used to claim production parity.
- Eager physical cache tensors are created per benchmark invocation; the graph
  runner retains its cache and captured addresses across repetitions.
- The SM89 projection kernel is intentionally specific to the L4 and this
  checkpoint's 128x128 FP8 scaling layout. Other devices and layouts fall back.
- The first graph implementation captures one graph per decode position. This
  preserves exact fixed KV writes but scales capture count and graph memory
  with configured output length; production engines use a smaller shape set
  plus dynamic slot mappings.
- The graph path reserves all request blocks up front and currently requires
  equal request shapes and stable request IDs across replays.

## Entry conditions for prefix caching

Satisfied: every configured block size passed exact-token validation on the
L4; allocator randomization, controlled OOM, fragmented lookup, block release,
and zero-active-request cleanup tests pass. Prefix caching can now build on
immutable shared physical blocks without changing the attention interface.
