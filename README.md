# MiniLLM-L4

A small, readable LLM inference engine, built one technique at a time and measured at every step.

It serves `Qwen/Qwen3-4B-Instruct-2507-FP8` on one NVIDIA L4 (24 GB, capped at 72 W). Each technique was tested on its own, and outputs are checked token for token.

## How it works

```
request ──► wait for KV memory ──► prefill ──► decode, one token per step ──► done
```

- **Prefill** reads the prompt and fills the KV cache. The GPU's math speed is the limit.
- **Decode** makes one token per step for every running request. Memory speed is the limit.
- **The scheduler** runs both on one GPU. It adds new requests between steps and cuts long prompts into chunks, so they don't freeze users who are already getting tokens.

## What each technique gained

| Technique | Problem it fixes | Before | After | Gain |
|---|---|---:|---:|---:|
| KV cache | recomputing the whole prompt for every token | 113 ms/token | 56 ms/token | **2.0x** |
| Continuous batching | new requests waiting for a whole batch to finish | 9.65 s to first token | 0.57 s | **17x** |
| Paged KV | copying caches as requests join and leave | 84 tok/s | 195 tok/s | **2.3x** |
| CUDA Graphs | GPU idle while the CPU launches kernels | 56 ms/token | 23 ms/token | **2.5x** |
| Prefix cache | re-reading the same system prompt | 1,481 ms to first token | 114 ms | **13x** |
| Chunked prefill | a long prompt freezing everyone | 3.26 s worst pause | 0.17 s | **19x** |

Each row uses its own workload ([details](docs/benchmarks.md#results-at-a-glance)).

## Compared with other engines

One request or 8 at once, 2,048-token prompts, 128 output tokens. Each engine used its own defaults.

| | HF `generate()` | vLLM | SGLang | MiniLLM |
|---|---:|---:|---:|---:|
| Time per token, 1 request | 70.8 ms | 23.9 ms | 22.8 ms | **22.6 ms** |
| First token, 1 request | 403 ms | **255 ms** | 269 ms | 265 ms |
| Output tok/s, 8 requests | 78 | 165 | **167** | 161 |
| Worst pause between tokens, 8 requests | 75 ms* | 269 ms | 1,690 ms | **155 ms** |

\* HF prefills all 8 prompts together, so no token comes out for about 4 s.

The three serving engines are close, and gaps of a few percent are not meaningful. All 27 compared requests matched HF exactly. This test sends requests all at once, so it says little about traffic arriving over time ([full comparison and limits](docs/phase_notes/engine_compare.md)).

## Our kernels

Written in Triton, and judged against SGLang's in [the kernel track](docs/phase_notes/kernel_track.md).

| Kernel | What it does | Gain | Output |
|---|---|---|---|
| FP8 matmul | runs the FP8 weights | 1.02x SGLang | bit-exact |
| [Fused kernels](docs/phase_notes/fused_kernels.md) | norms, SiLU and RoPE in one pass | prefill 10–23% faster | bit-exact |
| [Merged projections](docs/phase_notes/gemm_peak.md) | q/k/v as one matmul, gate/up as one | eager decode 11–17% faster | bit-exact |
| [Tiled prefill attention](docs/phase_notes/attention_compare.md) | FlashAttention-style tiles over the KV cache | first token 4–10% faster | opt-in |
| [Tensor-core decode attention](docs/phase_notes/kernel_track.md) | query heads share each K/V read | time per token 0.5–7% lower | opt-in |

Opt-in kernels round slightly differently from the exact path, so they stay off by default.

## Correctness

- All scheduling policies produce identical tokens, checked by `scripts/check_policy_equivalence.py`.
- Each engine stage was compared token for token with HF `generate()` on the validation workloads.
- On long batched outputs, batch shape alone can flip a near-tied token. HF `generate()` does this too.

## Try it

Run from the folder that contains `minillm_l4/`:

```bash
# Tests
.conda-env/bin/python -m pytest -q minillm_l4/tests

# Watch tokens stream, with live TTFT and time per token
.conda-env/bin/python minillm_l4/scripts/stream_generate.py \
    --prompt-length 128 6144 --max-new-tokens 64 --batch-size 6 \
    --arrival-interval-ms 400 --adaptive

# Run the main benchmarks (about 30 minutes)
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_showcase \
    --output-dir minillm_l4/results/showcase_$(date +%Y%m%d)
```

## Learn more

- [All benchmarks and charts](docs/benchmarks.md)
- [Questions we measured](docs/inference_questions.md)
- [Commands and folder layout](docs/commands.md)
- [Design notes for each step](docs/phase_notes/)
