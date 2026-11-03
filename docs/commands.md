# Command reference and project layout

Detailed commands for every benchmark runner. The project overview and headline results are in the [README](../README.md).

## Layout

```text
minillm_l4/
├── README.md
├── engine/                # the engine; never imports benchmarks/
│   ├── serving/           # scheduler, batching policies, prefix cache, CUDA Graph decode
│   ├── kv_cache/          # paged KV cache and attention kernels
│   ├── kernels/           # FP8 matmul, fused kernels, merged projections
│   ├── model_runner/      # flat-batch forwards for packed and mixed steps
│   ├── generation/        # simple generation loops and chunked prefill sessions
│   ├── events.py          # requests and request events
│   └── step_planner.py    # adaptive chunk sizing
├── benchmarks/
│   ├── commands/          # one run_*.py per benchmark
│   ├── core/              # harness, metrics, workloads
│   ├── runners/           # older per-phase runners and the HF baseline
│   ├── configs/           # workload YAML files
│   ├── data/              # synthetic prompts
│   └── charts/            # chart scripts
├── scripts/               # correctness gates and tools
├── tests/
├── docs/                  # benchmarks, study notes, older results in archive/
├── experiments/           # separate work, not part of the engine
└── results/               # local benchmark output, not in git
```

The deterministic synthetic workload is versioned under
`benchmarks/data/synthetic/workloads_v1.jsonl`. Model weights remain in the Hugging Face
cache and are never copied into the project.

Project configuration is YAML. Generated benchmark results and event streams
remain JSON/JSONL so they are easy to inspect and process programmatically.
Benchmark entry points live in `benchmarks/commands/`; use the module paths
shown below.

## Commands

For the extended 128–8192 prompt-length sweep with up to 1048 output tokens,
see [prefill/decode isolation and batching](phase_notes/prefill_decode_sweep.md).
It separates phase timing and compares matched static and continuous traces,
with checkpointed raw results and Markdown export.

Compare chunked and unchunked prefill on identical scheduled traffic:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_chunked_prefill --dry-run
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_chunked_prefill \
  --workload all --chunk-sizes 128 256
```

This command uses the existing exact-token references and saves dated raw
results, a source snapshot, and a Markdown report under `docs/archive/`.
See the [chunked-prefill note](phase_notes/chunked_prefill.md) for scheduling
behavior, timing boundaries, correctness gates, and validation limits.

Install the pinned Python dependencies in your CUDA-enabled environment:

```bash
python -m pip install -r minillm_l4/requirements.lock
```

Run the harness fixture:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_harness \
  --config minillm_l4/benchmarks/configs/workloads/minillm_l4_harness.yaml \
  --output-dir minillm_l4/results/phase0
```

Run the full Hugging Face baseline matrix on the L4:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_hf_baseline \
  --config minillm_l4/benchmarks/configs/workloads/qwen3_fp8_baseline.yaml \
  --output-dir minillm_l4/results/phase1
```

Run the isolated tests:

```bash
.conda-env/bin/python -m pytest -q minillm_l4/tests
```

See the [harness note](phase_notes/benchmark_harness.md), the
[Hugging Face baseline note](phase_notes/hf_baseline.md), the [manual
decode note](phase_notes/manual_decode.md), the
[KV-cache note](phase_notes/kv_cache.md), and the [concurrent-request
note](phase_notes/concurrent_requests.md) for definitions, measurements,
correctness gates, and limitations. The [continuous-batching note]
(phase_notes/continuous_batching.md) documents the Phase 5 scheduler,
cache rebasing, tests, and L4 reproduction command.

The [paged-KV note](phase_notes/paged_kv.md) documents fixed block
ownership, fragmentation accounting, the dense gather correctness path, and
the Phase 6 benchmark command.
The [prefix-cache note](phase_notes/prefix_cache.md) documents both
the batch-1 reference and continuous integration, exact-token gates, L4
baseline results, and current limits.

Run the manual decoder against the Phase 1 reference corpus:

    .conda-env/bin/python -m minillm_l4.benchmarks.commands.run_manual_decode --config minillm_l4/benchmarks/configs/workloads/qwen3_fp8_manual.yaml --output-dir minillm_l4/results/manual_decode

Compare contiguous KV reuse with full-prefix recomputation:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_kv_cache \
  --config minillm_l4/benchmarks/configs/workloads/qwen3_fp8_kv_cache.yaml \
  --workload short \
  --output-dir minillm_l4/results/kv_cache
```

Run staggered uniform and mixed request traces:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_concurrent_requests \
  --config minillm_l4/benchmarks/configs/workloads/qwen3_fp8_concurrent.yaml \
  --workload all \
  --max-batch-sizes 1 2 4 \
  --output-dir minillm_l4/results/concurrent_requests
```

Run the Phase 5 continuous-batching comparison:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_continuous_requests \
  --config minillm_l4/benchmarks/configs/workloads/qwen3_fp8_continuous.yaml \
  --workload all \
  --max-batch-sizes 1 2 4 \
  --output-dir minillm_l4/results/continuous_requests
```

Run the Phase 6 contiguous-versus-paged sweep on a CUDA-enabled L4:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_paged_kv \
  --config minillm_l4/benchmarks/configs/workloads/qwen3_fp8_paged.yaml \
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

Decode has two explicit numerical policies. `--decode-numerics auto` selects
`sdpa_compat` for the pinned BF16 Qwen dense-prefill hybrid/graph paths; other
paths retain `accurate`. The compatibility kernel still reads physical pages
directly, but uses reverse 128-token tiles, activation-dtype unnormalized
softmax weights, and one KV split to match the L4 SDPA backend more closely.
`--decode-numerics accurate` preserves FP32 softmax/value reductions and the
profiled split-KV choices. It can be closer to FP64 while choosing a different
greedy token than BF16 SDPA. Both implementations remain available; benchmark
artifacts record the policy, effective tile size, and split count. Neither
policy promises universal bitwise equality across attention backends.

Paged decode validates query positions, live physical block IDs, KV lengths,
and split coverage before page access. Sliced metadata tensors and independent
K/V source strides are supported. CUDA Graphs record live device assertions;
invalid replay metadata raises a CUDA device assertion and requires a fresh
CUDA context, so serving callers should reject invalid requests before replay.

Run the full suite including the pinned-model regression (local weights and
an L4 are required):

```bash
MINILLM_RUN_MODEL_TESTS=1 .conda-env/bin/python -m pytest minillm_l4/tests -q
```

Without that environment variable, the small CPU/GPU tests still run and the
real-model integration tests are explicitly skipped. The integration tests
retain the dense reference, including the previously failing seventeenth
token of `baseline-short-003`, and cover all four KV block sizes at batches
1/2/4 plus dense-prefill graph replay.

For a packed-prefill smoke test on the L4:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_paged_kv \
  --config minillm_l4/benchmarks/configs/workloads/qwen3_fp8_paged.yaml \
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
  --config minillm_l4/benchmarks/configs/workloads/qwen3_fp8_paged.yaml \
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
  --config minillm_l4/benchmarks/configs/workloads/qwen3_fp8_continuous.yaml \
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
  --config minillm_l4/benchmarks/configs/workloads/qwen3_fp8_manual.yaml \
  --workload all \
  --batch-sizes 1 \
  --count 1 \
  --output-tokens 32 \
  --repetitions 3 \
  --warmup-repetitions 1 \
  --reference-dir minillm_l4/results/phase1/references_baseline \
  --output-dir minillm_l4/results/manual_context_fixed

.conda-env/bin/python -m minillm_l4.benchmarks.charts.prefill_decode \
  --manifest minillm_l4/results/manual_context_fixed/manual_manifest.json \
  --batch-size 1 \
  --output minillm_l4/results/figures/prefill_decode_context.png
```

The command also writes a CSV beside the PNG containing the plotted P50, P95,
and P99 measurements.

Run the larger Phase 4 static-batching stress test and generate its report:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_concurrent_requests \
  --config minillm_l4/benchmarks/configs/workloads/qwen3_fp8_concurrent.yaml \
  --workload mixed \
  --max-batch-sizes 4 8 16 \
  --count-per-bucket 4 \
  --arrival-interval-ms 0 \
  --repetitions 3 \
  --warmup-repetitions 1 \
  --reference-dir minillm_l4/results/phase1/references_baseline \
  --output-dir minillm_l4/results/stress_batch_final \
  --no-system-telemetry

.conda-env/bin/python -m minillm_l4.benchmarks.charts.concurrency_stress \
  --manifest minillm_l4/results/stress_batch_final/concurrent_manifest.json \
  --workload mixed \
  --output minillm_l4/results/figures/concurrency_stress_final.png
```

This stress report includes aggregate token throughput, requests/sec, TTFT,
E2E latency, TPOT, GPU utilization, peak reserved VRAM, and static-batch
padding waste. The runner checkpoints its manifest after each batch size so
raw results remain indexed if a long run is interrupted.
