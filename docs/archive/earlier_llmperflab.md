# Earlier LLMPerfLab measurements

Saved workspace-level measurements from before the MiniLLM phase series. Source paths here are relative to the parent `LLMPerfLab/` workspace, outside this Git repository. These use different models, precision paths, workload shapes, or model-only timing boundaries from the MiniLLM serving phases.

## Hardware and compute probes

Hardware snapshot: 2026-07-24, NVIDIA L4/SM89, 23,034 MiB; PyTorch 2.13.0+cu130, CUDA build 13.0. Source: `results/hardware_metadata.json`.

| Saved result | Date (UTC) | 4096² FP16 matmul median (ms) | Achieved TFLOP/s | Measured samples |
| --- | --- | --- | --- | --- |
| `matmul_float16_4096.json` | 2026-07-24 | 2.298 | 59.81 | 30 |
| `compute_stability_float16_4096.json` | 2026-07-24 | 2.296 | 59.85 | 30 |
| `compute_telemetry_float16_4096.json` | 2026-07-24 | 2.607 | 52.71 | 30 |

## Sustained device-memory probes

GB/s is decimal effective traffic bandwidth from the recorded read/write traffic model.

| Saved result | Date (UTC) | Operation | Buffer MiB | Median pass (ms) | Effective GB/s |
| --- | --- | --- | --- | --- | --- |
| `memory_roof_copy_float32_256mib.json` | 2026-07-30 | copy | 256 | 2.330 | 230.40 |
| `memory_roof_copy_float32_1024mib.json` | 2026-07-30 | copy | 1024 | 9.298 | 230.96 |
| `memory_roof_add_float32_1024mib.json` | 2026-07-30 | add | 1024 | 9.272 | 231.62 |

## Model-only precision comparison

Saved 2026-08-09; Qwen3-4B Instruct-2507, prompt 128 / batch 1 / output 32. TTFT is model-only wall time and decode-step latency is CUDA event time; these are not arrival-based serving TTFT/TPOT. These synthetic runs do not establish exact-token equivalence across precisions.

| Precision path | Prefill GPU P50 (ms) | Model TTFT P50 (ms) | Decode GPU step P50 (ms) | Decode aggregate TPS | E2E aggregate TPS | Peak allocated GiB |
| --- | --- | --- | --- | --- | --- | --- |
| FP16 | 47.81 | 47.86 | 37.73 | 26.41 | 26.19 | 7.53 |
| Native FP8 / Transformers Triton | 73.88 | 73.92 | 69.80 | 14.28 | 14.25 | 4.15 |
| Native FP8 / custom SM89 | 56.70 | 56.74 | 53.59 | 18.62 | 18.59 | 4.15 |
| bitsandbytes NF4 | 56.15 | 56.19 | 45.96 | 21.70 | 21.55 | 2.74 |

Sources: `results/verification/live_l4/{fp16,fp8_triton,fp8_custom,bnb_nf4}_p128_b1/`.

## Fused-FP8 large-batch result

Prompt 128 / batch 64 / output 128; eager; three warm-ups and ten measured repetitions. ITL is device-side decode-step time.

| Precision | Prefill P50 (ms) | Decode ITL P50 (ms) | Decode aggregate TPS | E2E aggregate TPS | Peak decode GiB |
| --- | --- | --- | --- | --- | --- |
| FP16 | 1,775.97 | 64.45 | 989.62 | 819.81 | 9.78 |
| Fused native FP8 | 2,029.32 | 53.58 | 1,182.47 | 920.01 | 6.40 |

Source: `results/verification/fp16_fp8_optimization/optimization_summary.json`. Fused FP8 has higher decode/E2E throughput at this shape and slower prefill.

## First manual-generation comparison

Qwen3-0.6B FP16, 15-token prompt, 16 output tokens, three repetitions; model-only timing.

| Runtime | Mean wall time (ms) | Mean output TPS | Measured repetitions | Exact outputs |
| --- | --- | --- | --- | --- |
| llmperflab_manual | 409.56 | 39.07 | 3 | pass |
| transformers_generate | 418.18 | 38.26 | 3 | pass |

Source: `results/runtime_comparison/small_qwen3_0.6b.json`. Token digest: `67d895def5359e8ce2eddb4e7d8b9f58ae241523234d7742de68207803eb0a63`.

Additional local history is under `results/synthetic_inference/`, `results/verification/day1/`, `results/verification/fp16_fp8_crossover/`, `results/verification_20260808/`, `results/runtime_comparison/`, and `profiles/`. Those intermediate runs are not substituted for the final results above.
