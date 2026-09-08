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
- Model: `Qwen/Qwen3-4B-Instruct-2507-FP8`.
- Snapshot: `8591804019c8b22094c3b5b4454e0edc05dffc98`.
- Next work: iteration-level continuous batching.

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
cleanup. Static batches still run to completion before new work can execute;
iteration-level admission is intentionally reserved for continuous batching.

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
│       └── contiguous.py            # owned contiguous KV lifecycle
├── benchmarks/
│   ├── core/
│   │   ├── harness.py              # shared execution and metric orchestration
│   │   ├── schemas.py              # request, workload, and result types
│   │   ├── workloads.py            # deterministic fixture workload builders
│   │   ├── metrics.py              # request metrics and summaries
│   │   ├── timing.py               # wall/CUDA timing helpers
│   │   └── hardware.py             # environment, VRAM, and GPU telemetry
│   ├── runners/
│   │   ├── simulated.py            # deterministic CPU fixture runner
│   │   ├── huggingface_baseline.py # Qwen loader and HF baseline runner
│   │   ├── manual_decode.py        # manual backend benchmark runner
│   │   ├── kv_cache.py             # cache/recompute comparison runner
│   │   └── concurrent_requests.py  # staggered mixed-request trace runner
│   ├── commands/
│       ├── run_harness.py          # harness command-line entry point
│       ├── run_hf_baseline.py      # Hugging Face baseline entry point
│       ├── run_manual_decode.py    # manual decode entry point
│       ├── run_kv_cache.py         # cache/recompute comparison entry point
│       └── run_concurrent_requests.py # concurrent static-batch entry point
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
│       └── qwen3_fp8_concurrent.yaml
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
│       └── concurrent_requests.md
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
decode note](docs/phase_notes/manual_decode.md), and the
[KV-cache note](docs/phase_notes/kv_cache.md), and the [concurrent-request
note](docs/phase_notes/concurrent_requests.md) for definitions, measurements,
correctness gates, and limitations.

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
