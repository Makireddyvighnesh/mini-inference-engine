# Hugging Face vs vLLM vs SGLang vs MiniLLM-L4 (2026-11-03)

Same L4, same Qwen3-4B FP8 checkpoint, same prompt token IDs, greedy decoding, EOS ignored, prefix caching off. Prompts of 512, 1,024 and 2,048 tokens, 128 output tokens, 1 or 8 requests sent at once. Each engine ran in 2 rounds of 1 warm-up + 3 measured runs per case, so 6 measured runs per case.

Command: `benchmarks/commands/run_engine_compare.py`. Raw data: `results/engine_compare_20261103/`.

## Short answer

- **All three serving engines are about 3x faster than Hugging Face** at 1 request and about 2-2.8x at 8 requests.
- **At 1 request, MiniLLM has the fastest token rate.** It beats SGLang by 6% at 512 tokens and ties it at 2,048. It beats vLLM by 5-10%.
- **At 8 requests, throughput is close.** SGLang leads at 1,024 and 2,048 tokens. MiniLLM ties SGLang at 512 and is 4% behind at 2,048.
- **MiniLLM has by far the smallest pauses while other prompts prefill**: 140-155 ms, against 236-269 ms for vLLM and 432-1,690 ms for SGLang.
- **The cost of those small pauses:** at 8 requests, MiniLLM's slowest first tokens (p95) are 10-17% later than vLLM's.
- **Outputs match:** all 27 compared requests were identical to Hugging Face, token for token, on all three engines.

## 1 request

| Prompt tokens | Metric | HF generate | vLLM | SGLang | MiniLLM |
|---|---|---:|---:|---:|---:|
| 512 | Time to first token (ms) | 95.6 | 64.0 | 61.9 | **55.6** |
| 1,024 | Time to first token (ms) | 181.4 | 123.1 | 128.6 | 119.7 |
| 2,048 | Time to first token (ms) | 402.6 | **255.3** | 269.1 | 265.3 |
| 512 | Time per token (ms) | 71.7 | 22.6 | 21.7 | **20.4** |
| 1,024 | Time per token (ms) | 71.5 | 23.1 | 22.1 | **21.1** |
| 2,048 | Time per token (ms) | 70.8 | 23.9 | 22.8 | 22.6 |
| 512 | Output tok/s | 13.9 | 43.6 | 45.5 | **48.3** |
| 1,024 | Output tok/s | 13.8 | 41.9 | 43.7 | **45.7** |
| 2,048 | Output tok/s | 13.6 | 38.9 | 40.5 | 40.8 |

Bold marks a win larger than the noise. The rest are too close to call.

## 8 requests at once

| Prompt tokens | Metric | HF generate | vLLM | SGLang | MiniLLM |
|---|---|---:|---:|---:|---:|
| 512 | First token, p50 (ms) | 868.8 | 354.0 | 503.2 | 343.2 |
| 1,024 | First token, p50 (ms) | 1,882.8 | 729.6 | 783.1 | 632.0 |
| 2,048 | First token, p50 (ms) | 4,012.7 | 1,422.4 | 1,375.4 | 1,312.6 |
| 512 | First token, p95 (ms) | 870.5 | 489.1 | 514.2 | 570.6 |
| 1,024 | First token, p95 (ms) | 1,897.7 | 996.3 | 1,047.6 | 1,118.1 |
| 2,048 | First token, p95 (ms) | 4,059.8 | 2,131.0 | 2,219.1 | 2,348.1 |
| 512 | Worst pause (ms) | 74.3 | 236.4 | 431.5 | **139.8** |
| 1,024 | Worst pause (ms) | 73.7 | 246.2 | 870.0 | **149.5** |
| 2,048 | Worst pause (ms) | 74.5 | 268.9 | 1,689.9 | **155.2** |
| 512 | Output tok/s | 103.5 | 279.7 | 286.4 | 286.5 |
| 1,024 | Output tok/s | 93.7 | 228.5 | 233.1 | 228.8 |
| 2,048 | Output tok/s | 78.4 | 164.8 | **167.1** | 160.6 |

Hugging Face's worst pause is low only because all 8 prompts prefill together in one forward, before any request gets its first token. No token leaves for about 4 s at 2,048 tokens. Bold in this table compares only the three serving engines.

SGLang's first-token p50 is not a good summary at 8 requests. Its requests finish in groups, so the median can fall between two groups or move between runs. Use p95 and tok/s for SGLang here.

## What explains the differences

- **Hugging Face is slow per token because decode runs eagerly.** About 71 ms per step at both batch sizes, matching the host-bound eager step already measured (`CLAUDE.md`, Measurement pitfalls). The others use CUDA graphs.
- **SGLang's long pauses fit it finishing waiting prompts before it decodes again.** In one 2,048-token run the first request got its first token at 272 ms and then paused 1,938 ms. The last request's first token came at 2,176 ms, so the pause matches that wait. This comes from token timing; we did not read its scheduler trace.
- **MiniLLM's short pauses fit its adaptive chunks.** Its busy step limit is 150 ms, which matches the 140-155 ms measured. vLLM's 236-269 ms fit its 2,048-token prefill step. Both are inferred from settings and timing, not from a step trace.

## MiniLLM's first token time varies with adaptive chunking

At 1,024 tokens, 1 request, MiniLLM's first token ranged from 105 to 142 ms across the 6 runs. vLLM stayed within 122-125 ms. A follow-up test on the same prompts, alternating the two modes (8 runs each), found the source:

| Prompt tokens | Adaptive: p50 (min-max) ms | Whole prompt: p50 (min-max) ms |
|---|---|---|
| 512 | 58.4 (52.2-60.2) | 65.8 (62.2-66.1) |
| 1,024 | 118.3 (116.9-138.7) | 120.5 (119.7-122.3) |
| 2,048 | 275.2 (265.4-301.9) | 257.5 (254.4-259.5) |

- The spread is from adaptive scheduling, not the kernels. Whole-prompt runs are tight.
- With whole-prompt prefill at 2,048 tokens, MiniLLM (257.5 ms) is level with vLLM (255.3 ms). This came from a separate session, so it is not a strict same-run comparison.
- Adaptive is faster at 512 tokens and slower at 2,048. Time per token did not change.
- Not shown yet: why adaptive varies. The likely cause is the step-time model refitting from measured steps, which would change chunk sizes between runs, but these result files do not keep chunk events.

## How much to trust this

- **Noise.** The largest gap between rounds was at most 3.2% for time per token and tok/s on every engine. For first token it was 1.5% (HF), 1.3% (vLLM), 7.4% (SGLang) and 13.4% (MiniLLM); the last two come from the behaviour described above.
- **Run order.** Round 1 ran HF, vLLM, SGLang, MiniLLM; round 2 ran the reverse. Engines cannot alternate replays because each needs its own process. SGLang's round 2 started at 52 C rather than 50 C, with no visible effect.
- **Timing boundary.** HF and MiniLLM time a token when it reaches the host in the same process. vLLM and SGLang time it at the caller, after engine IPC. This was not measured separately, and it could account for some of the few-millisecond first-token gaps. It does not change the time-per-token picture.
- **Memory.** Peak use was HF 9.1 GiB, vLLM 20.4 GiB, SGLang 17.0 GiB and MiniLLM 8.0 GiB. This is not memory each engine needs: vLLM and SGLang reserve a large pool up front, and MiniLLM's pool was sized to each workload.
- **Software differs.** HF and MiniLLM: torch 2.13, Transformers 5.14.1. vLLM 0.26.0: torch 2.11. SGLang 0.5.23: torch 2.14.1, FlashInfer 0.7.2.
- **vLLM and SGLang are not tuned for the L4.** Both log that they have no tuned FP8 kernel config for this GPU and use a default. MiniLLM's config is tuned on this L4, so tuned configs could narrow its lead in projections.
- **MiniLLM ran from a dirty tree.** Commit `0f93b17` plus uncommitted scheduler changes (`manifest.json` lists the files). Their split-KV decode and sampling features were off in this run, but the changed scheduler code did run, so these numbers are not strictly for that commit.
- **Workload limits.** Prompts are synthetic and repetitive, sent all at once. This is not a test of arrivals over time, and 128 outputs is short. The older `vllm_comparison.md` covers arrivals over time for vLLM only.

## Settings

| Engine | Settings |
|---|---|
| HF generate | Stock Transformers FP8 path (`fp8_kernel_path="auto"`), SDPA, one static batch, `logits_to_keep=1` |
| vLLM | Defaults: torch.compile, full and piecewise CUDA graphs, chunked prefill at 2,048 tokens per step, Flash Attention |
| SGLang | Defaults: FlashInfer attention, overlap schedule, CUDA graphs, chunked prefill at 4,096, `max_prefill_tokens=16384`; radix cache off |
| MiniLLM | sm89 FP8, fused and merged kernels, mixed batching, adaptive chunks, CUDA Graph decode |

All used at most 32 requests in flight, except HF, which is one static batch. Context limit is 8,448 tokens for vLLM and SGLang.
