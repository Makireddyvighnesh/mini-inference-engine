# LFM2.5-2.6B + DSpark on a Colab T4

Does Liquid AI's DSpark draft model make LFM2.5-2.6B generate faster on a small, old GPU, and by how much?
Liquid reports 2.67x on an H100 (323 → 864 tok/s) and 2.27x on an M4 Max. Nobody has published a T4 number.

**Open in Colab:** [lfm2_dspark_t4.ipynb](https://colab.research.google.com/github/Makireddyvighnesh/mini-inference-engine/blob/claude/project-thread-negoja/experiments/lfm2_dspark_colab/lfm2_dspark_t4.ipynb)
(set the runtime to T4 GPU, then run the cells top to bottom).

## The models

| | Hugging Face repo | What it is |
|---|---|---|
| Target | [`LiquidAI/LFM2.5-2.6B-GGUF`](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF) | the 2.6B chat model whose answers we want |
| Draft | [`LiquidAI/LFM2.5-2.6B-DSpark-GGUF`](https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark-GGUF) | a 328M "drafter" that guesses the target's next tokens |

DSpark is not a model you chat with on its own. It only works next to its target.

## What speculative decoding is (and why it can be free speed)

Generating one token means reading **every weight** of the model from GPU memory once.
On a T4 (320 GB/s) the F16 target is 5.4 GB, so one token costs at least 5.4 / 320 ≈ 17 ms, whatever the GPU's compute.
Decode is *memory-bound*: the tensor cores sit mostly idle waiting for weights.

Checking 10 tokens in one forward pass also reads the weights once, so it costs about the same as producing 1.
Speculative decoding uses that:

1. a small, cheap **draft** model guesses the next k tokens,
2. the **target** checks all k in one forward pass,
3. with greedy decoding, the target keeps the guesses up to the first one it disagrees with, then adds its own token.

If the draft guesses well, one expensive target pass yields several tokens. The output is meant to be exactly what the
target would have written alone, because the target makes every final decision.

**DSpark's twist** (from [Liquid's blog](https://www.liquid.ai/blog/lfm2-5-dspark)): the drafter is not a separate small LLM.
It reads the target's own hidden states, drafts a whole block of up to 9 tokens in one parallel pass (5 transformer layers),
then a tiny sequential "Markov head" fixes each guess to fit the token before it. It reuses the target's embedding and
`lm_head`, so it adds only 328M parameters.

## What we measure

For each **target quantization** (F16 = 5.4 GB, Q4_K_M = 1.7 GB):

- `base`: target alone,
- `dspark3`, `dspark6`, `dspark10`: with the drafter, at most 3 / 6 / 10 drafted tokens per step (`--spec-draft-n-max`).

12 prompts in 4 categories (code, math, chat, structured JSON) × 2 repetitions × 256 new tokens, greedy, one request at a time,
through `llama-server` (llama.cpp, the engine Liquid used for DSpark outside SGLang). Then `llama-bench` for raw
prefill (pp512) and decode (tg128) speed.

Why llama.cpp and not Transformers, vLLM or SGLang: DSpark support exists only in llama.cpp (merged, PR #27383)
and in an unmerged SGLang PR (#31041). The T4 has no bf16, so we use the F16 GGUF.

## Predictions (write them down before looking)

| | F16 target | Q4_K_M target |
|---|---|---|
| decode ceiling (320 GB/s ÷ file size) | ~59 tok/s | ~190 tok/s |
| expected base decode | 40–50 tok/s | 90–130 tok/s |
| expected DSpark speedup | **2–3x** | **1.2–2x** |

Why the Q4 speedup should be smaller: the drafter (0.66 GB in F16, plus the shared 0.5 GB `lm_head`) costs little next to
a 5.4 GB target but a lot next to a 1.7 GB one. And verifying k tokens in Q4 switches llama.cpp from its fast
one-row kernel to a batched one, which is less efficient on Turing.

Why acceptance should depend on the prompt: code and JSON are predictable (high acceptance), open chat is not.

**Exactness:** speculative decoding is "exact" in theory, but the verify pass runs k rows through batched kernels while
plain decode runs 1 row, and FP16 sums in a different order can flip a near-tied token. This is the same issue as this
repo's exactness rules (see `CLAUDE.md`). The `same text` column shows how often it happens.

## Results format

Each run writes one folder `results/<YYYYMMDD-HHMM>_<gpu>[_tag]/`:

| file | contents |
|---|---|
| `env.json` | GPU, driver, CUDA, llama.cpp commit, model files and sizes, all arguments |
| `configs.json` | every config, its exact `llama-server` command, load time, status |
| `requests.jsonl` | one row per request: config, prompt id, category, rep, llama-server `timings` (`prompt_n/ms`, `predicted_n/ms`, `draft_n`, `draft_n_accepted`), output text |
| `llama_bench.jsonl` | llama-bench rows per target |
| `summary.csv` | one row per config (decode tok/s, speedup, acceptance, tokens per step, identical outputs) |
| `report.md` | the readable report |
| `logs/` | server and llama-bench logs |

`python analyze.py results/<run>` rebuilds `summary.csv` and `report.md` from the raw files at any time.

## Metrics

- **decode tok/s** = generated tokens ÷ generation time (after the first token). The headline number.
- **speedup** = decode tok/s ÷ the same target's `base` decode tok/s.
- **acceptance** = accepted drafted tokens ÷ drafted tokens.
- **tokens/step** = tokens per target forward pass. Each verify step yields its accepted drafts plus one target token, so
  steps = generated − accepted. Liquid reports 4.8 for this model on H100.
- **same text** = outputs identical to the `base` run of the same target and prompt.

## Files

- `lfm2_dspark_t4.ipynb` — the Colab notebook (runs everything below)
- `setup_llamacpp.sh` — builds llama.cpp with CUDA for sm_75 (`LLAMA_CPP_REF=<commit>` pins a version)
- `run_bench.py` — downloads the GGUFs and runs the benchmark (`--help` for options)
- `analyze.py` — builds `summary.csv` and `report.md`
- `prompts.jsonl` — the 12 prompts
