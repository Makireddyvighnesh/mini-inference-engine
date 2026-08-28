# MiniLLM-L4 Phase 1 — Hugging Face generation baseline

Date completed: 2026-08-27

## Outcome

Phase 1 is complete. MiniLLM-L4 now loads the pinned
`Qwen/Qwen3-4B-Instruct-2507-FP8` checkpoint with the Hugging Face tokenizer,
uses `model.generate()` for greedy decoding, streams token IDs into the common
request-event schema, and measures prompt/output shapes across static batch
sizes on one NVIDIA L4.

The baseline is deliberately still simple: requests are grouped into fixed
static batches and each batch runs to completion before the next batch starts.
There is no scheduler, shared KV allocator, prefix cache, or custom kernel in
this phase.

## What was built

- `minillm_l4/benchmarks/runners/huggingface_baseline.py`
  - Resolves and validates an exact local Hugging Face snapshot.
  - Loads the Qwen tokenizer and native fine-grained FP8 model path.
  - Builds exact-length request prompts with the pinned tokenizer.
  - Runs greedy `model.generate()` with EOS disabled so every workload emits
    the requested number of tokens.
  - Records `prefill_start`, first-token, per-token, and completion events via
    a small Hugging Face streamer adapter.
  - Creates and checks token-ID reference corpora.
- `minillm_l4/benchmarks/commands/run_hf_baseline.py`
  - Runs the reproducible Phase 1 matrix and writes a manifest, workload JSON,
    result JSON, raw event JSONL, and reference token corpus.
- `minillm_l4/benchmarks/core/harness.py`
  - Adds `run_batched()` while preserving the original sequential `run()` API.
  - Keeps request-level metric definitions identical across Phase 0 and Phase 1.
- `minillm_l4/configs/workloads/qwen3_fp8_baseline.yaml`
  - Pins the model, revision, native FP8 precision, workload shapes, batch
    sizes, warm-up count, repetitions, and telemetry settings.
- `minillm_l4/tests/test_hf_baseline.py`
  - Covers streamer events, exact output lengths, static-batch isolation,
    tokenizer-backed workloads, local snapshot validation, and reference
    corpus creation/checking.

## Model and environment

| Field | Value |
|---|---|
| Model | `Qwen/Qwen3-4B-Instruct-2507-FP8` |
| Exact revision | `8591804019c8b22094c3b5b4454e0edc05dffc98` |
| Checkpoint source | Local Hugging Face snapshot under `~/.cache/huggingface/hub` |
| Checkpoint quantization | Native FP8, E4M3, dynamic activation scaling, block size 128 × 128 |
| Model shape | 36 layers, hidden size 2560, 32 attention heads, 8 KV heads |
| Hardware | NVIDIA L4, SM89, 23,034 MiB |
| PyTorch | `2.13.0+cu130` |
| Transformers | `5.14.1` |
| Triton | `3.7.1` |
| Python | `3.12.13` |
| Git commit | Unavailable; workspace is not a valid Git checkout |

The model-loading path uses the checkpoint-native Transformers FP8 backend.
No MiniLLM-L4 custom GPU kernel or vLLM code is used in this phase.

## Benchmark configuration

Command:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_hf_baseline \
  --config minillm_l4/configs/workloads/qwen3_fp8_baseline.yaml \
  --output-dir minillm_l4/results/phase1
```

Configuration:

- Prompt lengths: 128, 512, and 2,048 tokens.
- Requested output lengths: 32, 64, and 128 tokens respectively.
- Requests per workload: 4.
- Static batch sizes: 1, 2, and 4.
- Warm-up repetitions: 1; measured repetitions: 3.
- Greedy argmax decoding; EOS stopping disabled.
- Closed-loop ordered execution; tokenization and model loading excluded from
  steady-state request timings.
- GPU allocator and best-effort `nvidia-smi` telemetry enabled.

Raw artifacts are under `minillm_l4/results/phase1/`, using descriptive
`baseline_*.json` and `baseline_*_events.jsonl` filenames. Each prompt bucket
has one shared reference corpus, so batch-size runs are checked against one
another as well as against their repeated runs.

## Results

Values are medians from the common harness summaries. TPS is aggregate
generated output tokens per second for the measured run. Peak allocated VRAM
is the maximum CUDA allocator value observed over the result's measured runs.

| Prompt / output | Batch | TTFT P50 (ms) | TPOT P50 (ms) | E2E P50 (ms) | Aggregate TPS P50 | Peak allocated VRAM (GiB) | GPU util P50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 / 32 | 1 | 74.440 | 71.682 | 2,296.476 | 13.848 | 4.15 | 33% |
| 128 / 32 | 2 | 74.270 | 72.463 | 2,320.540 | 27.279 | 4.18 | 34% |
| 128 / 32 | 4 | 94.545 | 73.037 | 2,359.439 | 53.117 | 4.23 | 34% |
| 512 / 64 | 1 | 95.210 | 72.448 | 4,659.625 | 13.691 | 4.25 | 34% |
| 512 / 64 | 2 | 180.054 | 72.478 | 4,746.275 | 26.833 | 4.38 | 35% |
| 512 / 64 | 4 | 387.215 | 72.924 | 4,982.041 | 50.869 | 4.56 | 37% |
| 2,048 / 128 | 1 | 404.787 | 72.439 | 9,606.537 | 13.300 | 4.56 | 37% |
| 2,048 / 128 | 2 | 911.765 | 72.965 | 10,178.830 | 25.130 | 4.99 | 41% |
| 2,048 / 128 | 4 | 1,960.646 | 73.316 | 11,272.402 | 45.213 | 5.85 | 48% |

The run-level aggregate throughput scales close to batch size, while TPOT is
roughly stable near 72–73 ms. TTFT and E2E latency rise with prompt length and
with the amount of queued static-batch work. This is the expected baseline
behavior that later continuous batching and chunked prefill phases should
improve for mixed traffic.

## Correctness gate

All 9 matrix points passed:

- Every measured request completed.
- Each request emitted exactly its configured number of tokens.
- Output token IDs were stable across all three measured repetitions.
- Batch-size runs for each prompt bucket matched the shared reference corpus.
- Streamer token events and output token IDs remained aligned.

Test command and result:

```text
.conda-env/bin/python -m pytest -q
70 passed, 1 skipped in 5.30s
```

The one skipped test is the pre-existing CUDA timer test in the sandboxed
namespace. The host-level Phase 1 benchmark itself had working CUDA and
recorded `cuda_available: true` in its result environment metadata.

## Before / after checkpoint

| Capability | End of Phase 0 | End of Phase 1 |
|---|---|---|
| Runner | Deterministic CPU fixture | Real Qwen native FP8 `model.generate()` |
| Tokenizer | Not involved in measured fixture | Pinned Hugging Face tokenizer |
| Execution | Sequential request calls | Sequential static batches |
| Events | Synthetic prefill/token timings | Hugging Face streamer token callbacks |
| Correctness | Harness event/metric tests | Exact model output token-ID corpus |
| GPU evidence | Best-effort/unavailable in sandbox | L4 allocator, utilization, and VRAM measurements |

## Known limitations

- The baseline is not a serving scheduler. A static batch runs to completion,
  so new requests cannot join an active decode loop.
- Static batching requires equal prompt and output lengths within a batch;
  mixed-length requests are intentionally left for later scheduling phases.
- The streamer callback observes tokens after Hugging Face moves them to CPU.
  It provides a reproducible first-token boundary, but is not a network socket
  or production streaming implementation.
- Prefill end is observed at the first generated-token callback because
  `model.generate()` does not expose a separate prefill event. Phase 2 will
  provide a direct prefill/decode split.
- The model-load configuration currently records the source snapshot and
  model metadata, but Git identity is unavailable in this workspace.
- This is a Hugging Face baseline, not a claim of parity with vLLM or an
  optimized MiniLLM-L4 engine.

## Entry conditions for Phase 2

- Keep the Phase 1 token corpus and `model.generate()` output as the trusted
  correctness reference.
- Replace the generation call with an explicit manual prefill/decode loop.
- Compare token IDs and logits within tolerance before comparing timings.
- Preserve the same exact-token workloads, batch sizes, repetitions, and
  hardware metadata so the Phase 2 delta is attributable to the custom loop.
