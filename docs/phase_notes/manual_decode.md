# MiniLLM-L4 — Manual decode loop

Date completed: 2026-08-28

## Outcome

The project now has a separate generation backend that does not call
Hugging Face model.generate(). It performs one prompt prefill through the
model forward method, selects the first token with greedy argmax, and then
performs one-token forward calls while carrying past_key_values from one step
to the next.

The Hugging Face path remains available as the trusted reference. Both
backends use the same request, workload, event, and result contracts.

## What was built

- minillm_l4/engine/generation/manual.py
  - Validates batched input IDs and attention masks.
  - Preallocates the prompt-plus-decode attention mask.
  - Runs one prefill forward call.
  - Runs one cached forward call per subsequent output token.
  - Performs greedy argmax selection.
  - Supports exact-length generation and opt-in EOS stopping.
  - Exposes callbacks for prefill completion and token streaming.
- minillm_l4/benchmarks/runners/manual_decode.py
  - Adapts the manual backend to the common benchmark harness.
  - Records prefill, token-ready, token-sent, and completion events.
  - Keeps static batches deterministic and rejects mixed prompt/output shapes.
- minillm_l4/benchmarks/commands/run_manual_decode.py
  - Loads the pinned Qwen checkpoint from YAML configuration.
  - Runs the manual backend matrix.
  - Checks outputs against the Phase 1 reference corpus.
- minillm_l4/configs/workloads/qwen3_fp8_manual.yaml
  - Pins the Qwen model, native FP8 path, workload shapes, and manual runner.

## Architecture

For a request with prompt length P and requested output length N:

```text
prompt [P tokens]
       │
       ├── model forward(use_cache=True) ──> first token + KV cache
       │
       ├── model forward(one token, past_key_values) ──> token 2 + new cache
       ├── model forward(one token, past_key_values) ──> token 3 + new cache
       └── repeat until N tokens
```

The benchmark configuration disables EOS stopping so that the manual and
Phase 1 outputs have identical requested lengths. The core function still
supports EOS and records true per-row lengths when EOS is explicitly enabled.

## Correctness

The manual runner was executed on the NVIDIA L4 with:

- Model: Qwen/Qwen3-4B-Instruct-2507-FP8
- Revision: 8591804019c8b22094c3b5b4454e0edc05dffc98
- Workload: 4 short requests, 128 prompt tokens, 32 output tokens
- Batch size: 1
- Warm-up: 1; measured repetitions: 1

All four generated token sequences matched the Phase 1 reference corpus
exactly. Each request used one prefill forward and 31 cached decode forwards.

The manual implementation also has forward-only unit tests covering:

- Prefill and decode call shapes
- Cache handoff
- Attention-mask lengths
- Greedy token selection
- EOS stopping and true sequence lengths
- Token event recording
- Static-batch validation

## Benchmark checkpoint

These are matched short/batch-1 smoke results from the common harness. The HF
run used three measured repetitions; the manual validation run used one, so
this is a correctness checkpoint rather than a formal speedup claim.

| Runtime | Requests | TTFT P50 (ms) | TPOT P50 (ms) | Aggregate TPS |
|---|---:|---:|---:|---:|
| Hugging Face baseline | 4 | 73.970 | 71.759 | 13.871 |
| Manual decode | 4 | 73.712 | 71.576 | 13.885 |

## 2026-09-26 full matched matrix

The checkpoint above was one cell with one measured repetition. The full
Phase 1 matrix was run through the manual backend on the L4 with one warm-up
and three measured repetitions per point, and compared with the same-day
Phase 1 re-run (`results/phase1_20260926`):

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_manual_decode \
  --config minillm_l4/configs/workloads/qwen3_fp8_manual.yaml \
  --output-dir minillm_l4/results/manual_decode_20260926
```

All 9 points passed; 108/108 outputs exactly matched
`results/phase1/references_baseline`. Provenance: commit `07fe935`; the
worktree was dirty only with uncommitted Phase 5 runner edits that the manual
path does not import.

| Prompt / output | Batch | HF TTFT P50 (ms) | Manual TTFT P50 (ms) | HF TPOT P50 (ms) | Manual TPOT P50 (ms) | HF TPS P50 | Manual TPS P50 | Manual vs HF TPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 / 32 | 1 | 72.9 | 73.2 | 71.47 | 71.51 | 14.00 | 13.96 | -0.3% |
| 128 / 32 | 2 | 74.5 | 74.4 | 71.86 | 72.09 | 27.81 | 27.70 | -0.4% |
| 128 / 32 | 4 | 96.7 | 94.4 | 72.20 | 70.44 | 54.77 | 56.17 | +2.6% |
| 512 / 64 | 1 | 97.2 | 95.4 | 71.24 | 71.16 | 13.97 | 13.98 | +0.1% |
| 512 / 64 | 2 | 181.9 | 179.6 | 72.06 | 71.16 | 27.10 | 27.41 | +1.1% |
| 512 / 64 | 4 | 391.4 | 386.3 | 71.83 | 71.51 | 52.04 | 52.32 | +0.5% |
| 2,048 / 128 | 1 | 406.2 | 403.3 | 70.12 | 71.52 | 13.74 | 13.49 | -1.8% |
| 2,048 / 128 | 2 | 918.2 | 898.6 | 70.03 | 72.36 | 26.08 | 25.37 | -2.7% |
| 2,048 / 128 | 4 | 1,965.2 | 1,933.5 | 71.71 | 72.19 | 46.23 | 46.11 | -0.3% |

The manual loop is at parity with `model.generate()` (every TPS difference
within ±2.7%, no consistent sign). Both paths spend each step on the same
Transformers forward, so removing `generate()`'s bookkeeping does not change
the host-bound ~71 ms step; speedups have to come from the forward itself.

## Artifacts

- Result: minillm_l4/results/manual_decode/manual_short_b1.json
- Events: minillm_l4/results/manual_decode/manual_short_b1_events.jsonl
- Manifest: minillm_l4/results/manual_decode/manual_manifest.json
- Reference: minillm_l4/results/phase1/references_baseline/short.json

## Known limitations

- Batches must currently have equal prompt and output lengths.
- KV memory is still owned by the model cache implementation; MiniLLM-L4
  does not yet control allocation, capacity, or release.
- There is no request scheduler, continuous batching, paging, or prefix cache.
- The manual benchmark currently uses eager execution and no custom kernels.

## Entry conditions for the next phase

Phase 3 implements the cache lifecycle in
`engine/kv_cache/contiguous.py`. The Phase 3 note records the cache/recompute
comparison and the transition to request scheduling.
