# Speculative Decoding Lab

An isolated experiment for speculative decoding on the existing NVIDIA L4 setup. This folder does not modify `minillm_l4` or copy model weights.

## Model pair

- **Target (teacher):** `Qwen/Qwen3-8B`, using the official `Qwen/Qwen3-8B-GGUF` Q4_K_M file (about 5.03 GB).
- **Draft (student):** `Qwen/Qwen3-0.6B`, using the official `Qwen/Qwen3-0.6B-GGUF` Q8_0 file (about 639 MB).
- **Backend:** a llama.cpp-compatible server configured at `../../serve_bonsai_2/bin/llama-server` in this workspace, using `draft-simple`. The binary must support Qwen3 GGUF models and llama.cpp speculative decoding.

Both models are from the Qwen3 family, which makes them a plausible tokenizer-compatible pair, but family name alone is not proof of token-ID equality. The strict GGUF tokenizer comparison below is required before speculative inference. Q8_0 keeps the small draft's quantization error low while its file remains much smaller than the target; acceptance rate and end-to-end throughput are measured by the benchmark.

## Setup

From this directory, install the small Python harness and its development tools into an existing environment:

```bash
python -m pip install -e '.[dev]'
hf download Qwen/Qwen3-8B-GGUF \
  Qwen3-8B-Q4_K_M.gguf \
  --revision 7c41481f57cb95916b40956ab2f0b139b296d974 \
  --local-dir models
hf download Qwen/Qwen3-0.6B-GGUF \
  Qwen3-0.6B-Q8_0.gguf \
  --revision 23749fefcc72300e3a2ad315e1317431b06b590a \
  --local-dir models
```

The two model files total about 5.7 GB and are kept under `models/`, which is ignored by Git. The configured llama-server binary and its shared libraries must also be available at the paths in `configs/experiment.yaml`. In a different checkout, set `SPECULATIVE_LLAMA_SERVER_BINARY` to the absolute path of your `llama-server`; the binary's directory is used for shared libraries by default. Set `SPECULATIVE_LLAMA_SERVER_LIBRARY_DIR` separately if needed. No local binary or model weights are committed.

## Correctness gates

1. Compare the relevant GGUF tokenizer metadata and exact token-ID ordering before inference; required fields missing from either model fail the gate:

   ```bash
   python -m speculative_decoding_lab.tokenizer_check \
     models/Qwen3-8B-Q4_K_M.gguf \
     models/Qwen3-0.6B-Q8_0.gguf
   ```

2. Run unit tests for the readable greedy accept/reject reference and percentile calculations:

   ```bash
   pytest
   ```

3. The benchmark renders the chat prompt with `/apply-template`, then uses llama.cpp's native `/completion` stream with raw token IDs enabled. It fails on an SSE error, a truncated stream, missing token IDs, missing stop reason, missing timing fields, or empty final text. Baseline and speculative runs must match token IDs, stop reason, and non-empty text. The speculative run must also report at least one draft token before a speedup is reported. `greedy.py` and `sampling.py` are educational, model-independent references, not the live model runtime; target verification is performed by llama.cpp.

The greedy verification rule is: draft up to K tokens, evaluate the target over that candidate block, accept the matching prefix, and at the first mismatch emit the target's token instead of the draft token. If every draft token matches, emit one extra target token. That is why one target block pass can commit several output tokens. `greedy.py` isolates this rule; the llama.cpp runtime owns KV rollback and the actual model execution.

For non-greedy sampling, `sampling.py` implements the probability rule: accept a proposed token with probability `min(1, p(token)/q(token))`; if rejected, sample from the normalized positive residual `max(0, p-q)`. If all proposals are accepted, sample one bonus token from the next target distribution. A seeded CPU test checks the first emitted token's empirical marginal against the target distribution.

For an additional live check on this pinned Qwen3 pair, run:

```bash
PYTHONPATH=src python -m speculative_decoding_lab.validation --samples 64
```

This first requires baseline and speculative greedy runs to stop naturally at EOS with identical token IDs, text, and correct visible answers for `OK`, arithmetic, and a JSON-format task. It then samples a fixed creative prompt with independent seed sets and compares a non-degenerate output-token position using a permutation test. This is a **statistical smoke test**, not proof that the full sampled sequence distributions or probabilities are identical. Sampled outputs are not required to match seed by seed.

## Run the experiment

Inspect the two server commands first:

```bash
PYTHONPATH=src python -m speculative_decoding_lab.benchmark --dry-run
```

Then run the matched workload. Each measured repetition starts a fresh server for each mode, warms it up, runs the prompts, and stops it. The first mode order is randomized; later repetitions alternate it. The full order is recorded. Set the first order explicitly to reproduce it:

```bash
PYTHONPATH=src python -m speculative_decoding_lab.benchmark
# To choose and reproduce the first mode explicitly, use either:
PYTHONPATH=src python -m speculative_decoding_lab.benchmark --mode-order baseline-first
PYTHONPATH=src python -m speculative_decoding_lab.benchmark --mode-order speculative-first
```

The default workload contains coding, arithmetic, and general-text prompts. It performs one warm-up and three measured repetitions for each prompt in both modes. Model startup is excluded from request timing, while the warm-up is repeated after each server restart. The server is single-slot, disables prompt-cache reuse, and uses the same teacher, context, prompts, generation limit, seed, and greedy sampler in each mode. Raw per-request JSONL, per-repetition server logs, a summary, a manifest, and copies of the exact YAML configuration and prompts are written beneath `results/`.

Reported metrics include TTFT to the first raw token ID, time spent rendering the chat template, E2E latency **through final text materialization**, llama.cpp prompt and decode throughput, and server decode TPOT. `stream_e2e_ms` stops when the SSE stream closes, while `postprocessing_ms` captures work after that, including a `/detokenize` fallback when stream text is absent. TTFT includes the template-rendering call. Decode TPOT is averaged over the server's decode steps after the first token; it is undefined for a one-token completion. Aggregate output TPS and speculative draft/accepted-token counts are also recorded. Speedup is reported only when token correctness passes and speculative draft tokens were observed. Raw throughput from this sequential, batch-1 experiment is not a concurrent-serving throughput claim. Results saved before this timing change use the older stream-only E2E definition; do not mix them in comparisons.

## Current limits

- The pinned Qwen3 tokenizer match and `draft-simple` model-pair load have been validated on one NVIDIA L4, but other model files or runtimes must pass the same gates again.
- The main speed benchmark uses greedy decoding. The separate live sampled-distribution check is limited to one prompt, one token position, and finite samples; it does not establish exact stochastic correctness for every sequence. The sampled verifier in Python is a CPU reference, not integrated into the model runtime. Batching and scheduler behavior remain out of scope.
- The EOS probes check three short, naturally terminating answers; the default 128-token benchmark may truncate responses while the model is still thinking. Do not treat matching truncated outputs as an answer-quality evaluation.
- The runtime path points to a local llama.cpp-compatible build. The Python project does **not** implement production draft-model execution, target verification, or KV rollback; those live operations belong to llama.cpp. This model pair does not depend on Bonsai's custom packed weights.
- Model weights and benchmark results are not committed.

## References

- [Qwen3-8B model](https://huggingface.co/Qwen/Qwen3-8B)
- [Official Qwen3-8B GGUF files](https://huggingface.co/Qwen/Qwen3-8B-GGUF)
- [Qwen3-0.6B model](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Official Qwen3-0.6B GGUF files](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF)
