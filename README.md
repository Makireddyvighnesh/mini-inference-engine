# MiniLLM-L4

MiniLLM-L4 is the isolated inference-engineering project for one NVIDIA L4.
It is intentionally kept under this directory so it does not change the
existing `adaserve/`, `llmperflab/`, `scripts/`, or root benchmark results.

## Current status

- Benchmark harness: complete.
- Hugging Face baseline: complete.
- Manual decode loop: complete.
- Model: `Qwen/Qwen3-4B-Instruct-2507-FP8`.
- Snapshot: `8591804019c8b22094c3b5b4454e0edc05dffc98`.
- Next work: explicit KV-cache lifecycle and accounting.

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

## Layout

```text
minillm_l4/
├── README.md
├── requirements.lock
├── engine/
│   ├── model_loading.py            # pinned native-FP8 model loading
│   └── generation/
│       ├── __init__.py
│       ├── huggingface.py          # trusted model.generate backend
│       └── manual.py               # explicit prefill/decode backend
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
│   │   └── manual_decode.py        # manual backend benchmark runner
│   └── commands/
│       ├── run_harness.py          # harness command-line entry point
│       ├── run_hf_baseline.py      # Hugging Face baseline entry point
│       └── run_manual_decode.py    # manual decode entry point
├── configs/
│   ├── __init__.py
│   ├── loader.py                   # shared YAML loader and validation
│   └── workloads/
│       ├── minillm_l4_harness.yaml
│       ├── qwen3_fp8_baseline.yaml
│       └── qwen3_fp8_manual.yaml
├── data/
│   └── synthetic/
│       └── workloads_v1.jsonl      # deterministic prompt seeds
├── docs/
│   ├── metric_definitions.md
│   └── phase_notes/
│       ├── benchmark_harness.md
│       ├── hf_baseline.md
│       └── manual_decode.md
├── results/
│   ├── phase0/                  # harness fixture results
│   ├── phase1/                  # canonical Qwen baseline results
│   ├── manual_decode/           # manual decoder results
│   ├── real_inference/          # small HF inference smoke run
│   └── baseline_smoke/          # small smoke-run artifacts
└── tests/
    ├── test_benchmark_harness.py
    ├── test_configs.py
    ├── test_hf_baseline.py
    └── test_manual_decode.py
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
[Hugging Face baseline note](docs/phase_notes/hf_baseline.md), and the [manual
decode note](docs/phase_notes/manual_decode.md) for definitions, measurements,
correctness gates, and limitations.

Run the manual decoder against the Phase 1 reference corpus:

    .conda-env/bin/python -m minillm_l4.benchmarks.commands.run_manual_decode --config minillm_l4/configs/workloads/qwen3_fp8_manual.yaml --output-dir minillm_l4/results/manual_decode
