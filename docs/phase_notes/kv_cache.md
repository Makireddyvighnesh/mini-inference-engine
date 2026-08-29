# Explicit KV-cache lifecycle

## What was built

Phase 3 adds an explicit cache owner at
`engine/kv_cache/contiguous.py` and a no-cache reference path at
`engine/generation/recompute.py`.

The cache owner is responsible for:

- creating one cache for an ordered batch of request IDs;
- rejecting owner changes and duplicate request IDs;
- enforcing a logical maximum number of cached positions;
- tracking the current position without synchronizing on every token;
- checking that the model updates the cache object owned by MiniLLM;
- inspecting every layer's key/value shape, dtype, device, and tensor bytes;
- reporting physical allocation, logical capacity, utilization, and free-memory
  capacity estimates;
- replacing the backing cache on reset and dropping it on release.

The model-compatible backing implementation is Transformers `DynamicCache`.
Its tensors are contiguous and grow as tokens are appended. The logical
capacity check is owned by MiniLLM, while physical fixed-block allocation is
reserved for the paged-cache work. This keeps the default path numerically
identical to the already validated Qwen eager path.

## Execution paths

For a prompt of length `P` and `N` output tokens, cached decoding performs:

```text
prefill: model(prompt, use_cache=True)                  -> first token + cache
decode:  model(one token, past_key_values=cache)        -> next token + cache
         repeat N - 1 times
```

The no-cache path performs:

```text
model(prompt, use_cache=False)                          -> first token
model(prompt + token 1, use_cache=False)                -> second token
model(prompt + token 1 + token 2, use_cache=False)      -> third token
```

The expected model-input work per sequence is therefore approximately:

```text
cached:    P + (N - 1) tokens sent through model forwards
recompute: N*P + N*(N - 1)/2 tokens sent through model forwards
```

Both paths use greedy `argmax`, the same attention-mask semantics, and the
same Hugging Face reference corpus.

## Correctness and unit tests

`tests/test_kv_cache.py` covers:

- real `DynamicCache` layer shape and byte accounting;
- capacity overflow before an unsafe append;
- owner mismatch, cache isolation, reset, and idempotent release;
- cached versus recomputed token equality;
- cached one-prefill/one-token-decode call shapes;
- recomputation's increasing full-prefix shapes;
- runner metadata and resource cleanup;
- exact output correctness through the common benchmark runner.

The project test suite passed with 26 tests before the real-model benchmark.

## Benchmark configuration

- Model: `Qwen/Qwen3-4B-Instruct-2507-FP8`
- Revision: `8591804019c8b22094c3b5b4454e0edc05dffc98`
- Hardware: one NVIDIA L4, 24 GB
- Dtype: checkpoint-native fine-grained FP8 with BF16 non-FP8 modules
- Workload: one long request, 2,048 prompt tokens, 128 output tokens
- Batch size: 1
- Warm-up: 1 repetition
- Measurement: 3 repetitions after 1 warm-up repetition
- Sampling: greedy, EOS disabled, exact output length
- GPU/system telemetry: disabled for this focused timing run

Command:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_kv_cache \
  --config minillm_l4/configs/workloads/qwen3_fp8_kv_cache.yaml \
  --workload long \
  --modes contiguous recompute \
  --batch-sizes 1 \
  --count 1 \
  --repetitions 1 \
  --warmup-repetitions 1 \
  --reference-dir minillm_l4/results/phase1/references_baseline \
  --output-dir minillm_l4/results/kv_cache_long_repeated \
  --no-gpu-sampling \
  --no-system-telemetry
```

## Real-model results

| Mode | TTFT P50 | TPOT P50 | Aggregate TPS | Correctness |
|---|---:|---:|---:|---|
| Contiguous KV reuse | 400.164 ms | 69.994 ms | 13.777 | pass |
| Full-prefix recomputation | 418.930 ms | 443.584 ms | 2.255 | pass |

Relative to recomputation, contiguous KV reuse measured:

- `6.34x` lower TPOT;
- `6.11x` higher aggregate TPS;
- identical generated token IDs.

The short prompt smoke test also passed exact correctness. At 128 prompt
tokens, the two modes were close in TPOT because batch-1 execution was
dominated by reading the 4B model weights. The long prompt made repeated
prefix computation visible.

The short batch-shape validation also passed exact correctness for all tested
batch sizes:

| Batch | Contiguous TPOT | Recompute TPOT | Contiguous TPS | Recompute TPS |
|---:|---:|---:|---:|---:|
| 1 | 71.490 ms | 70.257 ms | 13.966 | 14.227 |
| 2 | 70.981 ms | 80.298 ms | 28.142 | 24.989 |
| 4 | 71.245 ms | 116.479 ms | 55.536 | 34.460 |

This one-repetition batch sweep is a shape and ownership validation, not a
final latency claim.

## Memory accounting

For the long run, the cache contained 36 layers with per-layer key/value
shapes equivalent to `[1, 8, 2175, 128]` and BF16 storage. The measured
contiguous KV tensor allocation was:

```text
147,456 bytes per sequence token
320,716,800 total KV tensor bytes
```

The CUDA allocator reported 320,750,592 incremental allocated bytes, only
33,792 bytes above the tensor-byte calculation. This is a useful accounting
check, not a claim that all model workspace is KV storage.

The per-token calculation is:

```text
2 tensors × 36 layers × 8 KV heads × 128 head dimension × 2 BF16 bytes
= 147,456 bytes
```

The manifest records the CUDA allocator's before/after allocation and the
difference between observed incremental bytes and tensor bytes. That small
difference includes cache object metadata and allocator effects; it is not
attributed to KV storage.

## Important finding

An initial experiment used Transformers `StaticCache` with a full fixed
capacity. It produced one greedy token difference from the dynamic-cache
reference because the fixed-capacity attention execution shape changed the
numerical path. It was rejected as the correctness default. This is why the
current Phase 3 backend owns a contiguous growable cache and logical capacity;
fixed physical blocks belong in a separately measured optimization.

## Known limitations

- The backing tensor growth is still performed by Transformers `DynamicCache`.
- The logical capacity is enforced, but unused capacity is not preallocated.
- A static batch still requires equal prompt and output lengths.
- No concurrent scheduler, continuous batching, paged blocks, prefix sharing,
  CUDA Graphs, or custom kernels exist yet.
- The focused real-model result uses one measured repetition; it is a
  validation checkpoint, not a full production benchmark matrix.

## Entry conditions for the next phase

Phase 4 will add multiple request lifecycles, queues, cleanup, cancellation,
and a deterministic static-batching baseline for variable request arrivals.
The cache owner and request IDs created here provide the resource-isolation
boundary that the scheduler will use.
