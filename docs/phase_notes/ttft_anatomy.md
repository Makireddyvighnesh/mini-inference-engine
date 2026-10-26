# Q1: What happens between a request arriving and its first token?

**Time to first token (TTFT)** is how long a user waits before the model starts answering.

## The short answer

- **One request alone:** almost all of the wait is the GPU reading the prompt (the "prefill"). That takes 44 ms for a 128-token prompt and 1.4 s for an 8,192-token prompt. Everything else is tiny.
- **Many requests at once:** the GPU reads one prompt at a time. Each request waits for the prompts ahead of it, so the wait is mostly *other people's prompts*.
- **Too many long requests at once:** the GPU runs out of memory for them. New requests must wait until earlier ones finish. This can be the longest wait of all.

Setup: Qwen3-4B FP8 on one NVIDIA L4, with the engine's recommended settings. We sent 1 to 32 requests at the same moment, with prompts of 128 to 8,192 tokens.

## The steps, in order

| Step | What happens | Time |
|---|---|---|
| 1. Tokenize | Turn the text into tokens | 0.2–10 ms |
| 2. Wait for memory | Wait for GPU memory to hold this request | 0 if there is room |
| 3. Wait for other prompts | Wait while the GPU works on other prompts | 0 if alone |
| 4. Prefill | The GPU reads the whole prompt | 44 ms – 1.4 s |
| 5. First token | Picked at the end of the prefill | ~0 |
| 6. Detokenize | Turn the token back into text | ~0.005 ms |

When a request is alone, step 4 is more than 99% of the wait.

## How long is the wait?

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/ttft_anatomy/ttft-heatmap-dark.svg">
  <img alt="Median time to first token" src="../assets/ttft_anatomy/ttft-heatmap-light.svg">
</picture>

Typical (median) wait:

| Requests at once | 128 tokens | 512 | 2,048 | 8,192 |
|---|---|---|---|---|
| 1 | 44 ms | 56 ms | 0.27 s | 1.4 s |
| 8 | 97 ms | 0.35 s | 1.3 s | 7.8 s |
| 32 | 0.37 s | 1.2 s | 5.0 s | 29 s |

Two simple patterns:
- **Double the requests, roughly double the wait.**
- **Longer prompts mean longer waits.**

## Where does the wait come from?

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/ttft_anatomy/ttft-anatomy-dark.svg">
  <img alt="What the wait is made of" src="../assets/ttft_anatomy/ttft-anatomy-light.svg">
</picture>

- **Blue (its own prompt) stays about the same.** A 2,048-token prompt needs about 0.35 s of GPU work, whether 1 or 32 requests arrive.
- **Green (waiting for other prompts) grows with every extra request.** With 32 requests of 2,048 tokens, the average request waits 4.7 s for others and only 0.35 s for itself.
- **Yellow (waiting for memory) appears only with many long prompts.** The GPU has room for about 84,000 tokens of requests, roughly 10 prompts of 8,192 tokens. With 32 of them, 22 must wait for others to finish: an extra 15 s on average.

## Requests are served one after another

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/ttft_anatomy/ttft-timeline-dark.svg">
  <img alt="Each request from arrival to first token" src="../assets/ttft_anatomy/ttft-timeline-light.svg">
</picture>

- **Left (8 × 2,048 tokens):** a staircase. Each request waits (green) for the ones before it, then gets its turn (blue).
- **Right (32 × 8,192 tokens):** the top requests first wait for memory (yellow), then for their turn.

## Simple rule of thumb

The GPU reads about **6,500 prompt tokens per second** when it is busy. So:

> **wait for the last request ≈ total prompt tokens of everyone ahead ÷ 6,500**

Example: 32 requests × 2,048 tokens = 65,536 tokens. 65,536 ÷ 6,500 ≈ 10 s. Measured: 10.0 s.

## Three details worth knowing

1. **Short prompts can share the GPU.** The GPU can read several 128-token prompts in one pass. So 32 short requests wait about half as long as they would one by one.
2. **Long prompts are cut into pieces when others are already answering.** This keeps other users' answers flowing smoothly, but adds 0.4–0.7 s to a long prompt's wait.
3. **The work outside the GPU is tiny.** Tokenizing takes about 1 ms per 1,000 tokens.

## What would make it faster

- **For one user:** a faster prefill (faster matrix math on the GPU).
- **For many users:**
  - more prompt tokens per second (faster prefill, or more GPUs);
  - reuse work for shared prompts (prefix caching);
  - serve short prompts first;
  - turn away extra requests early instead of making them wait a long time.
- **For many long prompts:** use less memory per token (an 8-bit KV cache would fit about twice as many requests).

## How it was measured

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_ttft_anatomy --output-dir minillm_l4/results/ttft_anatomy_<date>
.conda-env/bin/python minillm_l4/scripts/make_ttft_anatomy_charts.py minillm_l4/results/ttft_anatomy_<date>/ttft_anatomy.json minillm_l4/docs/assets/ttft_anatomy
```

- **Requests and runs:** all requests arrive at the same moment, each prompt has the same length, and each request generates 16 tokens. Every case ran once to warm up and 3 times for real (about 20 minutes in total).
- **Timing source:** the engine records a timestamp at each step: arrival, memory granted, each piece of the prompt, first token. The colored parts in the charts come straight from those timestamps and add up to the full wait.
- **Engine settings:** mixed batching, adaptive chunk sizes, CUDA Graphs, fused kernels, merged projections, no prefix cache.
- **GPU memory for requests:** whatever was free after loading the model, minus 4 GiB kept spare.
- **Limits:** all requests arriving at once is a worst case; requests spread over time wait less (see Q15). There is no web server in this test, which would add a little more time.
