# MiniLLM-L4

MiniLLM-L4 is the isolated inference-engineering project for one NVIDIA L4.
It is intentionally kept under this directory so it does not change the
existing `adaserve/`, `llmperflab/`, `scripts/`, or root benchmark results.

## Current status

- Benchmark harness: complete.
- Hugging Face baseline: complete.
- Manual decode loop: complete.
- Explicit KV-cache lifecycle and no-cache comparison: complete.
- Concurrent request lifecycle and FIFO static batching: complete.
- Iteration-level continuous batching: implemented and CPU/tiny-Qwen
  validated; L4 benchmark pending in a CUDA-enabled runtime.
- Paged KV allocation: complete. Fixed-size allocation, per-request block
  tables, controlled out-of-memory behavior, physical K/V storage, dense
  gather fallback, direct block-table attention, and an optimized-prefill /
  fused-Triton-paged-decode path are implemented and validated on the L4.
- Packed ragged prefill: complete for the static-batch path. Prompts are
  flattened without padding. The exact-token-safe Triton page-walking kernel
  is the default; fused SDPA is available as an experimental benchmark path.
- Component tracing: complete for packed eager decode and CUDA Graph decode.
  `--trace-summary` enables synchronized diagnostic mode; normal performance
  runs keep GPU work asynchronous so tracing does not distort the headline
  latency. Diagnostic artifacts contain request preparation, allocator,
  metadata, model forward, sampling, streaming, output, and cleanup spans
  with wall and CUDA event timings.
- Fixed-shape CUDA Graph decode: exact reusable one-graph replay is validated
  for stable request shapes and IDs; graph capture remains an experimental
  serving constraint rather than a general scheduler.
- Model: `Qwen/Qwen3-4B-Instruct-2507-FP8`.
- Snapshot: `8591804019c8b22094c3b5b4454e0edc05dffc98`.
- Next work: prefix caching.

## What is implemented

The reusable benchmark package provides deterministic request and workload
schemas, short/medium/long and mixed workloads, event recording, synchronized
timing, percentile summaries, repeated runs, JSON/JSONL artifacts, and
best-effort GPU/VRAM telemetry.

The Hugging Face baseline adds the pinned tokenizer and native FP8 model
loader, exact-token workloads, greedy `model.generate()` execution, static
batching for batch sizes 1/2/4, token streaming events, reference token
corpora, and correctness checks across the benchmark matrix.

The manual decode backend adds a forward-only generation path: one prompt
prefill, greedy selection of the first token, one-token decode forwards, and
explicit `past_key_values` handoff. It reports the same request events and
checks its outputs against the Hugging Face reference corpus.

The KV-cache backend adds explicit contiguous-cache ownership, capacity checks,
position tracking, reset/release, per-layer shape inspection, byte accounting,
and a full-prefix recomputation reference path. Both paths are checked against
the Hugging Face token corpus.

The concurrent-request backend adds the waiting/prefill/decoding/finished
lifecycle, staggered arrivals, a deterministic FIFO queue, padded static
batches with mixed prompt/output lengths, cancellation before execution,
per-request completion events, padding-waste accounting, and guaranteed cache
cleanup. Static batches still run to completion before new work can execute.

The continuous backend adds iteration-level admission, a token-budgeted
prefill queue, an active decode batch rebuilt after every iteration, immediate
finished-row removal, an optional initial batching wait, and explicit cache
rebasing for new rows. The rebasing path is copy-based and uses Transformers
`DynamicCache`.

The paged backend adds fixed-size physical KV blocks, a deterministic free
block pool, logical-to-physical block tables, append/release accounting,
capacity backpressure, and three model paths. `PagedKvBatchRunner` is the
token-correct dense-gather control. `PagedAttentionBatchRunner` runs readable
block-wise attention during both prefill and decode. `PagedHybridBatchRunner`
uses optimized SDPA during prefill, copies the resulting KV tensors once into
  physical pages, and then decodes directly from block tables without a dense
  gather. A fused Triton kernel performs page lookup, QK reduction, online
  softmax, and weighted-value accumulation in one launch per layer. Stage
  timings and logical page visits are included in benchmark artifacts, and
  the readable PyTorch implementation remains available as a fallback.

The packed-prefill backend drives projections and MLPs over one flat token
buffer with `cu_seqlens`, so mixed prompt lengths do not become a rectangular
prompt batch. Its experimental CUDA `sdpa` backend runs fused
scaled-dot-product attention for each ragged request and writes the resulting
K/V directly into pages. `triton` remains the exact-token-safe default for the
project-owned page-walking attention kernel. On CUDA, packed K/V page writes
use one project-owned scatter kernel per layer instead of Python looping over
each request and page.

## Layout

```text
minillm_l4/
├── README.md
├── requirements.lock
├── engine/
│   ├── model_loading.py            # pinned native-FP8 model loading
│   ├── request.py                  # request lifecycle state machine
│   ├── scheduler.py                # deterministic FIFO batch queue
│   ├── generation/
│       ├── __init__.py
│       ├── huggingface.py          # trusted model.generate backend
│       ├── manual.py               # explicit prefill/decode backend
│       └── recompute.py            # no-cache correctness reference
│   └── kv_cache/
│       ├── __init__.py
│       ├── contiguous.py             # owned contiguous KV lifecycle
│       ├── packed.py                 # flat-token request metadata
│       ├── paged.py                  # fixed blocks, tables, and cache storage
│       ├── paged_attention.py        # paged and packed attention dispatch
│       ├── qwen3_paged.py             # Qwen3 direct-attention adapter
│       ├── triton_packed_attention.py # packed prefill kernel
│       ├── triton_packed_kv.py        # packed K/V scatter kernel
│       ├── triton_paged_attention.py # fused one-token decode kernel
│       ├── triton_paged_kv.py         # dynamic decode K/V writer
│       └── ...
│   └── model_runner/
│       └── qwen3_packed.py            # flat-token Qwen3 prefill runner
├── benchmarks/
│   ├── core/
│   │   ├── harness.py              # shared execution and metric orchestration
│   │   ├── schemas.py              # request, workload, and result types
│   │   ├── workloads.py            # deterministic fixture workload builders
│   │   ├── metrics.py              # request metrics and summaries
│   │   ├── timing.py               # wall/CUDA timing helpers
│   │   ├── hardware.py             # environment, VRAM, and GPU telemetry
│   │   └── tracing.py              # per-component execution spans
│   ├── runners/
│   │   ├── simulated.py            # deterministic CPU fixture runner
│   │   ├── huggingface_baseline.py # Qwen loader and HF baseline runner
│   │   ├── manual_decode.py        # manual backend benchmark runner
│   │   ├── kv_cache.py             # cache/recompute comparison runner
│   │   ├── concurrent_requests.py   # staggered static-batch trace runner
│   │   ├── continuous_requests.py   # iteration-level continuous runner
│   │   ├── paged_kv.py              # gather and direct paged runners
│   │   ├── packed_paged.py          # no-padding packed prefill runner
│   │   └── paged_cuda_graph.py      # reusable fixed-shape graph runner
│   ├── commands/
│       ├── run_harness.py          # harness command-line entry point
│       ├── run_hf_baseline.py      # Hugging Face baseline entry point
│       ├── run_manual_decode.py    # manual decode entry point
│       ├── run_kv_cache.py          # cache/recompute comparison entry point
│       ├── run_concurrent_requests.py # concurrent static-batch entry point
│       ├── run_continuous_requests.py # continuous-batching entry point
│       └── run_paged_kv.py            # contiguous versus paged sweep
│   └── plots/
│       ├── prefill_decode.py        # context-length metric chart
│       └── concurrency_stress.py    # batch-size stress report
├── configs/
│   ├── __init__.py
│   ├── loader.py                   # shared YAML loader and validation
│   └── workloads/
│       ├── minillm_l4_harness.yaml
│       ├── qwen3_fp8_baseline.yaml
│       ├── qwen3_fp8_manual.yaml
│       ├── qwen3_fp8_kv_cache.yaml
│       ├── qwen3_fp8_concurrent.yaml
│       ├── qwen3_fp8_continuous.yaml
│       └── qwen3_fp8_paged.yaml
├── data/
│   └── synthetic/
│       └── workloads_v1.jsonl      # deterministic prompt seeds
├── docs/
│   ├── metric_definitions.md
│   └── phase_notes/
│       ├── benchmark_harness.md
│       ├── hf_baseline.md
│       ├── manual_decode.md
│       ├── kv_cache.md
│       ├── concurrent_requests.md
│       ├── continuous_batching.md
│       └── paged_kv.md
├── results/
│   ├── phase0/                  # harness fixture results
│   ├── phase1/                  # canonical Qwen baseline results
│   ├── manual_decode/           # manual decoder results
│   ├── real_inference/          # small HF inference smoke run
│   ├── baseline_smoke/          # small smoke-run artifacts
│   └── figures/                 # generated PNG/CSV benchmark charts
└── tests/
    ├── test_benchmark_harness.py
    ├── test_configs.py
    ├── test_hf_baseline.py
    ├── test_manual_decode.py
    ├── test_kv_cache.py
    ├── test_concurrent_requests.py
    ├── test_continuous_requests.py
    ├── test_paged_kv_cache.py
    ├── test_paged_runner.py
    ├── test_prefill_decode_plot.py
    └── test_concurrency_stress_plot.py
```

The deterministic synthetic workload is versioned under
`data/synthetic/workloads_v1.jsonl`. Model weights remain in the Hugging Face
cache and are never copied into the project.

Project configuration is YAML. Generated benchmark results and event streams
remain JSON/JSONL so they are easy to inspect and process programmatically.
The old benchmark module entry points remain as thin compatibility wrappers;
new work should use the organized paths shown above.

## Commands

Install the pinned Python dependencies in your CUDA-enabled environment:

```bash
python -m pip install -r minillm_l4/requirements.lock
```

Run the harness fixture:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_harness \
  --config minillm_l4/configs/workloads/minillm_l4_harness.yaml \
  --output-dir minillm_l4/results/phase0
```

Run the full Hugging Face baseline matrix on the L4:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_hf_baseline \
  --config minillm_l4/configs/workloads/qwen3_fp8_baseline.yaml \
  --output-dir minillm_l4/results/phase1
```

Run the isolated tests:

```bash
.conda-env/bin/python -m pytest -q minillm_l4/tests
```

See the [harness note](docs/phase_notes/benchmark_harness.md), the
[Hugging Face baseline note](docs/phase_notes/hf_baseline.md), the [manual
decode note](docs/phase_notes/manual_decode.md), the
[KV-cache note](docs/phase_notes/kv_cache.md), and the [concurrent-request
note](docs/phase_notes/concurrent_requests.md) for definitions, measurements,
correctness gates, and limitations. The [continuous-batching note]
(docs/phase_notes/continuous_batching.md) documents the Phase 5 scheduler,
cache rebasing, tests, and L4 reproduction command.

The [paged-KV note](docs/phase_notes/paged_kv.md) documents fixed block
ownership, fragmentation accounting, the dense gather correctness path, and
the Phase 6 benchmark command.

Run the manual decoder against the Phase 1 reference corpus:

    .conda-env/bin/python -m minillm_l4.benchmarks.commands.run_manual_decode --config minillm_l4/configs/workloads/qwen3_fp8_manual.yaml --output-dir minillm_l4/results/manual_decode

Compare contiguous KV reuse with full-prefix recomputation:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_kv_cache \
  --config minillm_l4/configs/workloads/qwen3_fp8_kv_cache.yaml \
  --workload short \
  --output-dir minillm_l4/results/kv_cache
```

Run staggered uniform and mixed request traces:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_concurrent_requests \
  --config minillm_l4/configs/workloads/qwen3_fp8_concurrent.yaml \
  --workload all \
  --max-batch-sizes 1 2 4 \
  --output-dir minillm_l4/results/concurrent_requests
```

Run the Phase 5 continuous-batching comparison:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_continuous_requests \
  --config minillm_l4/configs/workloads/qwen3_fp8_continuous.yaml \
  --workload all \
  --max-batch-sizes 1 2 4 \
  --output-dir minillm_l4/results/continuous_requests
```

Run the Phase 6 contiguous-versus-paged sweep on a CUDA-enabled L4:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_paged_kv \
  --config minillm_l4/configs/workloads/qwen3_fp8_paged.yaml \
  --workload all \
  --output-dir minillm_l4/results/paged_kv
```

The paged command keeps total physical token capacity fixed while sweeping
block sizes. It records block utilization, internal fragmentation, physical
KV bytes, stage timings, page visits, TTFT/TPOT, and exact-token correctness.
Add `--trace-summary` when a synchronized component trace is needed.
The default config runs optimized SDPA prefill followed by direct paged
decode, selecting Triton on supported CUDA inputs. It also selects the
L4-specific `sm89` FP8 projection kernel; set `model.fp8_kernel_path` to
`auto` for the Transformers control. Use `--decode-backend
torch` for the readable fallback, `--decode-backend triton` to require the
kernel, `--modes paged_graph` for fixed-address CUDA Graph replay,
`--modes paged_direct` for all-blockwise educational prefill, or
`--modes paged_gather` for the dense-gather control. Use
`--modes paged_packed` for vLLM-like ragged prefill: prompts are flattened
into one token buffer, described by `cu_seqlens`, and written directly to
per-request KV pages without padding. `auto` selects the exact-token-safe
Triton kernel. Use `--prefill-backend sdpa` explicitly to measure the faster
experimental fused-SDPA path, or use `--prefill-backend triton` to require the
project-owned packed CUDA kernel. `torch` remains the readable reference.
CUDA Graph mode uses packed prefill by default; use
`--graph-prefill-backend dense` for the dense-prefill control.

For a packed-prefill smoke test on the L4:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_paged_kv \
  --config minillm_l4/configs/workloads/qwen3_fp8_paged.yaml \
  --workload short \
  --modes paged_packed \
  --block-sizes 32 \
  --batch-sizes 2 \
  --repetitions 3 \
  --warmup-repetitions 1 \
  --trace-summary \
  --output-dir minillm_l4/results/packed_prefill
```

`--trace-summary` enables synchronized diagnostic tracing and prints the most
recent measured batch's component totals. The complete trace is stored in each
result JSON under `runs[].runner_diagnostics[].execution_trace`. GPU spans
report synchronized wall time and CUDA event time; their difference is
host/synchronization cost. Because synchronization is intentional, diagnostic
latencies must not be used as the performance headline. Run without the flag
for the asynchronous benchmark number. The trace also reports unattributed
time so missing instrumentation is visible.

For a focused batch-1 graph comparison:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_paged_kv \
  --config minillm_l4/configs/workloads/qwen3_fp8_paged.yaml \
  --workload short \
  --modes paged_packed paged_graph \
  --block-sizes 32 \
  --batch-sizes 1 \
  --repetitions 10 \
  --warmup-repetitions 1 \
  --trace-summary \
  --output-dir minillm_l4/results/graph_comparison
```

The graph runner uses packed prefill by default, then captures one reusable
one-token decode graph for a fixed active batch. Before each replay it updates
the static token, position, and sequence-length buffers; the device-side page
writer and attention kernel then select the current KV slot. Capture time is
reported separately and excluded from steady-state latency. The graph path
requires equal prompt/output shapes and stable request IDs across replays.

For a quick correctness-focused run:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_continuous_requests \
  --config minillm_l4/configs/workloads/qwen3_fp8_continuous.yaml \
  --workload mixed \
  --max-batch-sizes 2 \
  --count-per-bucket 1 \
  --repetitions 1 \
  --warmup-repetitions 0 \
  --reference-dir minillm_l4/results/phase1/references_baseline \
  --output-dir minillm_l4/results/continuous_smoke
```

Plot prefill and decode latency over 128, 512, and 2,048-token prompts:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_manual_decode \
  --config minillm_l4/configs/workloads/qwen3_fp8_manual.yaml \
  --workload all \
  --batch-sizes 1 \
  --count 1 \
  --output-tokens 32 \
  --repetitions 3 \
  --warmup-repetitions 1 \
  --reference-dir minillm_l4/results/phase1/references_baseline \
  --output-dir minillm_l4/results/manual_context_fixed

.conda-env/bin/python -m minillm_l4.benchmarks.plots.prefill_decode \
  --manifest minillm_l4/results/manual_context_fixed/manual_manifest.json \
  --batch-size 1 \
  --output minillm_l4/results/figures/prefill_decode_context.png
```

The command also writes a CSV beside the PNG containing the plotted P50, P95,
and P99 measurements.

Run the larger Phase 4 static-batching stress test and generate its report:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_concurrent_requests \
  --config minillm_l4/configs/workloads/qwen3_fp8_concurrent.yaml \
  --workload mixed \
  --max-batch-sizes 4 8 16 \
  --count-per-bucket 4 \
  --arrival-interval-ms 0 \
  --repetitions 3 \
  --warmup-repetitions 1 \
  --reference-dir minillm_l4/results/phase1/references_baseline \
  --output-dir minillm_l4/results/stress_batch_final \
  --no-system-telemetry

.conda-env/bin/python -m minillm_l4.benchmarks.plots.concurrency_stress \
  --manifest minillm_l4/results/stress_batch_final/concurrent_manifest.json \
  --workload mixed \
  --output minillm_l4/results/figures/concurrency_stress_final.png
```

This stress report includes aggregate token throughput, requests/sec, TTFT,
E2E latency, TPOT, GPU utilization, peak reserved VRAM, and static-batch
padding waste. The runner checkpoints its manifest after each batch size so
raw results remain indexed if a long run is interrupted.
