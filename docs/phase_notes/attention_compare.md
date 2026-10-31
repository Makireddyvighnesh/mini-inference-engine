# Q6: Why is FlashAttention faster when the math is the same?

All four methods compute the same thing: `softmax(Q·Kᵀ / √d) · V`. The outputs match (see the last section). The difference is **how often the data travels between GPU memory and the compute units**.

| Method | What it is |
|---|---|
| **Naive** | Plain PyTorch ops, as in Transformers' "eager" attention: Q·Kᵀ, mask, softmax, ·V |
| **PyTorch SDPA** | `scaled_dot_product_attention` with its standard (math) kernel |
| **FlashAttention** | the same SDPA call with its FlashAttention kernel (what the engine uses for prefill) |
| **Our Triton kernel** | our own kernels: packed prefill, and the paged decode kernel the engine uses |

## The short answer

- **Naive and standard SDPA write a giant score table to memory.** That table has one number for every pair of tokens, for every head. It used 6.1 GB at 4k tokens and grows 4× each time the prompt doubles, so naive runs out of memory at 8k and SDPA's standard kernel at 16k.
- **FlashAttention never writes the table.** It works on small tiles that stay in the fast on-chip memory and keeps a running softmax. It needs only room for the output, and it spends its time on math, not on moving data.
- **Prefill at 4k tokens:** FlashAttention takes 1.9 ms. Naive is 41× slower, standard SDPA 60× and our Triton prefill kernel 115×.
- **Decode:** FlashAttention and our kernel are equally fast (5.0 ms at 32k). Naive is 9× slower and standard SDPA 30× slower, because both copy the whole KV cache before using it.

## Prefill: speed

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/attention/attention-prefill-dark.svg">
  <img alt="Prefill attention time vs prompt length, and times slower than FlashAttention at 4k" src="../assets/attention/attention-prefill-light.svg">
</picture>

One prompt, one layer:

| Prompt length | Naive | PyTorch SDPA | **FlashAttention** | Our Triton kernel |
|---|---|---|---|---|
| 512 | 0.68 ms | 1.5 ms | **0.085 ms** | 3.6 ms |
| 2k | 19 ms | 29 ms | **0.52 ms** | 56 ms |
| 4k | 79 ms | 118 ms | **1.9 ms** | 223 ms |
| 8k | out of memory | 496 ms | **9.0 ms** | 0.9 s |
| 32k | out of memory | out of memory | **148 ms** | 16.8 s |

- **FlashAttention runs at 59–71 TFLOPS.** That's the L4's BF16 matrix-multiply limit (about 64 TF under its power cap, see `gemm_peak.md`), so attention is as fast as a plain matrix multiply.
- **Naive and standard SDPA run at only 1.1–1.8 TFLOPS.** The GPU mostly waits for the score table to move in and out of memory.
- **Our Triton prefill kernel is the slowest (0.6 TFLOPS),** even though it never writes the table. See "The two tricks" below.
- **Every method takes 4× longer when the prompt doubles,** because the work grows with length × length. The method decides how big that time is.

## Prefill: memory

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/attention/attention-memory-dark.svg">
  <img alt="Extra memory vs prompt length against the L4's memory, and at 4k" src="../assets/attention/attention-memory-light.svg">
</picture>

| Prompt length | Naive | PyTorch SDPA | **FlashAttention** | Our Triton kernel |
|---|---|---|---|---|
| 512 | 104 MB | 109 MB | **4 MB** | 4 MB |
| 4k | 6.1 GB | 4.8 GB | **33 MB** | 32 MB |
| 8k | out of memory (needs ~24 GB) | 18.8 GB | **65 MB** | 64 MB |
| 32k | out of memory | out of memory | **260 MB** | 256 MB |

- **The score table is the whole problem.** It grows 4× per doubling and hits the L4's 22 GB at about 8k (naive) to 16k (standard SDPA) tokens.
- **FlashAttention and our kernel need only the output,** which grows 2× per doubling: 260 MB even at 32k.

## The two tricks inside FlashAttention

1. **Running softmax (no score table).** Softmax needs the maximum and the sum of a whole row. FlashAttention keeps a running max and sum and rescales as it goes, so the full row never has to be stored. This saves the **memory**.
2. **Tiles that reuse K and V.** It loads a block of queries (e.g. 128) into fast on-chip memory and streams K/V past it. Each piece of K/V read from GPU memory serves 128 queries, not 1. This saves the **time**.

Our Triton prefill kernel has trick 1 but not trick 2. It handles one query row at a time, so it reads all of K and V again for every row. That's why it uses as little memory as Flash but runs about 100× slower. The engine therefore uses FlashAttention for prefill, and this kernel isn't in the serving path.

## Decode: speed

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/attention/attention-decode-dark.svg">
  <img alt="Decode attention time vs context length, and times slower than FlashAttention at 32k" src="../assets/attention/attention-decode-light.svg">
</picture>

8 requests, each adding one token, one layer:

| Context | Naive | PyTorch SDPA | **FlashAttention** | Our Triton kernel (engine) |
|---|---|---|---|---|
| 512 | 0.61 ms | 2.0 ms | **0.049 ms** | 0.055 ms |
| 2k | 2.6 ms | 8.3 ms | **0.27 ms** | 0.31 ms |
| 8k | 10 ms | 33 ms | **1.0 ms** | 1.2 ms |
| 32k | 44 ms | 149 ms | **5.0 ms** | **5.0 ms** |

- **With one new token there is no big table.** Decode just reads the KV cache, so the best possible time is "KV size ÷ memory speed". At 32k the 8 requests have 1.07 GB of K/V per layer: 5.0 ms means 214 GB/s, close to the L4's real copy speed (about 231 GB/s).
- **Our kernel matches FlashAttention** at long contexts and is within 10–20% at short ones. It also reads K/V straight from the engine's paged cache, which SDPA can't.
- **Naive and standard SDPA copy the cache first.** They expand K/V from 8 heads to 32 (standard SDPA also converts to 32-bit), so they read and write it 4–8 times. At 32k standard SDPA needs 14 GB of extra memory for one layer.
- **Decode only gets faster with less data.** No kernel can beat memory speed; a smaller KV cache (8-bit) would.
- **Across 36 layers at 32k,** attention alone costs about 180 ms per decode step, so long contexts make decode slow.

## Does the math really match?

Yes. Every output was compared with an FP32 reference:

- **FlashAttention, standard SDPA and our kernels:** max error 0.008 (prefill) and 0.0001–0.001 (decode). That's the normal rounding of a bf16 result.
- **Naive:** max error about 0.02. It rounds the score table to bf16 before softmax, which makes it the *least* accurate.

## Two more kernels in the data

- **PyTorch's memory-efficient SDPA kernel** (prefill only) is in the JSON. It uses as little memory as Flash but is about 1.6× slower (3.1 ms at 4k).
- **Our split-KV decode** (8 parts) is about 1.8× slower at batch 8 (9.0 ms at 32k). Splitting helps only when too few requests keep the GPU busy.

## How it was measured

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_attention_compare --output-dir minillm_l4/results/attention_compare_<date>
.conda-env/bin/python minillm_l4/scripts/make_attention_charts.py minillm_l4/results/attention_compare_<date>/attention_compare.json minillm_l4/docs/assets/attention
```

- **Setup:** one attention layer at Qwen3-4B shapes (32 query heads, 8 KV heads, head size 128, bf16), random inputs, no model loaded. The whole L4 was free (22 GB). NVIDIA L4, PyTorch 2.13, 2026-10-30.
- **Timing:** median of 3–7 runs after a warm-up, with CUDA events. Short decode cases repeat the call many times inside each run.
- **Extra memory:** peak GPU memory during the call minus memory in use before it (inputs excluded). In the charts, an X marks out of memory, drawn at the would-be value (4× the previous size).
- **Methods:**
  - The memory-efficient SDPA kernel has no grouped-KV support, so it got K/V already copied to 32 heads (copy not timed).
  - Our decode kernel uses the engine's settings: SDPA-compatible mode, one KV split.
  - The decode kernel's optional "GQA reuse" variant was 10–25× slower in a side test and is not used by the engine.
