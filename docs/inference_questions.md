# LLM Inference Questions

Questions to answer with measurements from MiniLLM-L4: Qwen3-4B-Instruct-2507-FP8 on one NVIDIA L4. There is one GPU, so prefill and decode always share it. Disaggregation is discussed, not measured.

Each question has a goal, the experiment, the plot that answers it, and a status. Solve them one at a time and tick the box when done.

Status: `[ ]` not started · `[~]` partial data exists · `[x]` done

---

## A. Where the time goes

### Q1. What happens between a request arriving and the first token leaving? `[ ]`
- **Experiment:** split TTFT into queue wait, tokenization, admission, prefill forward pass, sampling and detokenization. Use short (128-token) and long (8k-token) prompts, on an idle and a busy engine.
- **Plot:** a stacked bar of TTFT components for each case.

### Q2. Is prefill compute-bound and decode memory-bound? `[~]`
- **Done (GEMM level):** `run_gemm_peak` covers all precisions against the datasheet, plus the projection roofline. See `docs/phase_notes/gemm_peak.md`. Still to do: the full prefill and decode steps (attention included) on the roofline.
- **Experiment:** measure achieved TFLOPS and GB/s for prefill (128 to 16k tokens) and decode (batch 1 to 64).
- **Plot:** a roofline with the L4 compute and bandwidth limits, and one point per case.

### Q3. How do prefill latency and decode speed scale? `[~]`
- **Experiment:** prompt length vs prefill ms; decode batch size vs tok/s and ms per token.
- **Plot:** two line charts. Mark where decode saturates.

### Q4. Profile a step: compute, memory bandwidth, or kernel-launch overhead? `[~]`
- **Experiment:** use Nsight / torch profiler to split time into GEMM, attention, norm/RoPE/SiLU, sampling, Python overhead and GPU idle. Do this for prefill, decode and mixed steps.
- **Plot:** stacked bars per step type.

### Q5. The GPU sits at ~40% utilization under load. Debug it. `[~]`
- **Experiment:** decode GPU-busy % with eager execution vs CUDA Graphs. Count syncs and launches per step.
- **Plot:** GPU busy % and ms per token, before and after each fix.

---

## B. Kernels

### Q6. Why is FlashAttention faster when the math is the same? `[ ]`
- **Experiment:** compare naive attention, PyTorch SDPA and our Triton/Flash path. Sweep context length from 512 to 32k, for prefill and decode. Check that outputs match.
- **Plot:** latency vs context length and peak memory vs context length.

### Q7. Fused RMSNorm (and SiLU-mul, RoPE) vs PyTorch. `[~]`
- **Experiment:** benchmark each fused kernel against the PyTorch ops across row counts. Confirm bitwise-equal output.
- **Plot:** speedup vs rows, plus the end-to-end gain per step.

### Q8. Quantization: BF16 vs FP8. What do we gain and what do we lose? `[~]`
- **Experiment:** compare speed (prefill and decode), weight memory, and free KV capacity. Measure quality as perplexity and token agreement on a fixed prompt set. INT8/INT4 are optional extras.
- **Plot:** grouped bars for speed and memory, and a table for quality.

### Q9. What do CUDA Graphs do for decode? `[~]`
- **Experiment:** decode ms per token with and without graphs, at batch sizes 1 to 64.
- **Plot:** a line chart with two series.

---

## C. Memory and KV cache

### Q10. Size the KV cache: formula, checked against measured numbers. `[ ]`
- **Experiment:** derive KV bytes/token from layers × KV heads × head_dim × 2 × dtype bytes, accounting for GQA. Compare with measured Qwen3-4B KV usage. Then work the paper example: a Llama-style 70B model, 32K context, 100 users, in FP16 vs FP8.
- **Plot:** a table, plus max concurrent requests vs context length on the L4.

### Q11. PagedAttention: what problem does it solve? `[~]`
- **Experiment:** contiguous vs paged KV. Measure max concurrent requests, memory wasted to fragmentation and reservation, and throughput at the same memory budget.
- **Plot:** bars for capacity and waste.

### Q12. Prefix caching: how much prefill does it save? `[~]`
- **Experiment:** sweep the shared-prefix fraction from 0 to 90%. Measure hit rate, TTFT and throughput.
- **Plot:** shared-prefix % vs TTFT and hit rate.

### Q13. What happens when KV memory runs out? `[~]`
- **Experiment:** overload the engine. Track KV usage, queue depth, admissions and any preemption over time.
- **Plot:** a timeline of KV % and queue depth.

---

## D. Batching and scheduling (one GPU, colocated prefill and decode)

### Q14. Continuous batching vs static batching. `[~]`
- **Experiment:** the same arrival trace under both. Measure throughput, P50/P95 latency and GPU busy %.
- **Plot:** throughput vs P95 latency, one point per policy.

### Q15. How do TTFT and ITL move as load increases? `[~]`
- **Experiment:** sweep the arrival rate up to saturation.
- **Plot:** load-latency curves for P50/P95 TTFT and ITL.

### Q16. A long prompt stalls short chats. Fix TTFT with chunked prefill and scheduling. `[~]`
- **Experiment:** one 8k–16k prompt arrives among short chats. Compare prefill-first, decode-first, fixed 2048-token chunks (vLLM style) and our adaptive planner. Measure chat TTFT and worst decode pause.
- **Plot:** a timeline of decode stalls, plus a TTFT vs worst-pause scatter per policy.

### Q17. Cut inter-token latency without tanking throughput. `[ ]`
- **Experiment:** sweep chunk size / step time limit and max batch.
- **Plot:** a Pareto curve of P95 ITL vs tok/s.

### Q18. How many tokens does each step process? `[~]`
- **Experiment:** log step tokens (decode rows + prefill rows) over a serving run.
- **Plot:** a histogram and timeline of step tokens against step ms.

### Q19. Disconnects: cancel in-flight generation. `[~]`
- **Experiment:** cancel requests mid-prefill and mid-decode. Measure the time until KV pages are freed, and the effect on other requests.
- **Plot:** a timeline of KV pages freed after cancel, and a table.

### Q20. P99 latency is 10× P50. Diagnose it. `[ ]`
- **Experiment:** collect the latency distribution under load, then tie each tail request to counters: queue wait, the big prefill in the same step, preemption.
- **Plot:** a latency CDF, plus attribution of the tail requests.

---

## E. Observability and comparison

### Q21. Observability: which metrics should the engine report? `[~]`
- **Experiment:** confirm the summary reports TTFT, TPOT/ITL, queue depth, KV utilization, prefix-cache hit rate and GPU busy %. Add an estimated cost per 1M tokens (L4 $/hr ÷ throughput).
- **Plot:** a dashboard-style panel from one serving run.

### Q22. How close are we to vLLM on the same GPU, model and workload? `[x]`
- **Experiment:** `benchmarks/commands/run_vllm_compare.py`, covering prefill, decode, serving, longmix and heavy loads.
- **Plot:** grouped bars per workload.

### Q23. What is the end-to-end gain from the first naive engine to today? `[ ]`
- **Experiment:** apply the optimizations one at a time: paged KV, continuous batching, Triton attention, CUDA Graphs, fused kernels, chunked prefill, split FP8 GEMM.
- **Plot:** a waterfall chart of throughput and TTFT.

---

## Discussed only (not measurable on one L4)

- **Disaggregated prefill/decode:** why it helps, KV transfer cost, and how Q16 is the single-GPU alternative.
- **Speculative decoding:** see `speculative_decoding_lab/`.
- **Cache-aware routing across replicas:** needs multiple GPUs.
- **Other design-only topics:** GPU selection (H100/H200/B200/L40S), MoE parallelism, autoscaling, multi-tenant fairness, safe rollout, and capacity planning for 1M DAU.
