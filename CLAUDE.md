# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MiniLLM-L4: an LLM inference engine for `Qwen/Qwen3-4B-Instruct-2507-FP8` (pinned snapshot `8591804019c8b22094c3b5b4454e0edc05dffc98`, loaded from the HF cache, never copied here) on one NVIDIA L4 (SM89). It grew step by step from HF `generate()` into a serving engine: explicit KV cache, static and continuous batching, paged KV with Triton kernels, packed prefill, CUDA Graph decode, prefix caching, chunked prefill, mixed prefill/decode batching, and adaptive chunk sizing. The README shows one row per technique with its measured gain.

## Commands

This package imports itself as `minillm_l4`, so run Python from the **parent** directory (`/home/ubuntu/STT/LLMPerfLab`) with the workspace conda env:

```bash
.conda-env/bin/python -m pytest -q minillm_l4/tests                       # CPU suite (~18 s)
.conda-env/bin/python -m pytest -q minillm_l4/tests/test_chunked_prefill.py::test_name
MINILLM_RUN_MODEL_TESTS=1 .conda-env/bin/python -m pytest -q minillm_l4/tests  # + pinned-model GPU tests
.conda-env/bin/python -m minillm_l4.benchmarks.commands.<cmd> --help
```

- Tests run on CPU with a tiny `Qwen3Config`; GPU/Triton tests skip without CUDA (a sandbox may hide the driver: `torch.cuda.is_available() == False`). `test_phase_sweep` reads the git-ignored reference corpus under `results/`, so it fails in a fresh checkout.
- Benchmarks live only in `benchmarks/commands/` and write to dated, git-ignored `results/<name>_<date>/` directories (commands refuse non-empty output dirs). Most take `--config configs/workloads/<file>.yaml`; `run_showcase` (README benchmark set, ~30 min), `run_cuda_graphs`, `run_prefill_ttft`, `run_profile_decode`, and `run_kv_capacity` do not. `scripts/make_showcase_charts.py <showcase.json> docs/assets/showcase` regenerates the README charts.
- `scripts/stream_generate.py` streams tokens with live TTFT/TPOT (`--mixed`, `--adaptive`, `--chunk-size`, `--arrival-interval-ms`, several `--prompt-length` values cycled across requests).
- `scripts/check_policy_equivalence.py` (GPU) is the no-HF correctness gate for scheduling changes: whole-prompt, chunked, mixed, adaptive, and their CUDA Graph variants must emit identical tokens.
- `run_cuda_graphs` compares eager vs graph decode (decode, prefill, serving, overhead sections); `scripts/make_cuda_graph_charts.py <cuda_graphs.json> docs/assets/cuda_graphs` renders its charts.

## Architecture

**We do not own the model code.** The engine loads HF `Qwen3ForCausalLM` (`benchmarks/runners/huggingface_baseline.py::load_qwen_fp8`) and extends it in two ways:

1. `engine/kv_cache/qwen3_paged.py::install_paged_qwen3_attention` swaps every layer's `self_attn` for `PagedQwen3Attention`. Its `forward` picks a path by keyword: none → the original HF attention (dense fallback); `paged_kv_cache` → rectangular decode/prefill through `_paged_attend` (Triton paged kernels); `paged_packed_metadata` → flattened prefill of several fresh prompts; `paged_mixed_metadata` → a mixed step.
2. `engine/model_runner/qwen3_packed.py` and `qwen3_mixed.py` drive the decoder layers directly over a flat `[total_tokens, hidden]` batch (one forward for many requests, no padding), passing that metadata down.

**FP8.** `load_qwen_fp8(fp8_kernel_path="sm89")` installs `engine/kernels/sm89_fp8.py` in place of Transformers' `fp8_linear` and pre-tunes it. Its Triton autotune key is the power-of-two bucket of the row count (`M_BUCKET`), cached on disk; an exact-`M` key re-tuned for seconds inside timed requests whenever batching produced a new token count.

**Serving engine.** `benchmarks/runners/chunked_prefill.py::ChunkedPrefillPagedRunner` is the current scheduler (it inherits `ContinuousPrefixPagedRunner` → `PrefixCachedPagedRunner`, which owns the `PagedKvAllocator`/`PagedKvCache` pool and prefix registry). Requests are admitted whenever slots and KV pages for their full length are free; they queue only when pages run out. Policies, chosen by constructor flags:
- whole-prompt (`prefill_chunk_size=None`): waiting fresh prompts go through packed prefill, FIFO up to `packed_prefill_token_limit` tokens per forward, then one batched decode step;
- chunked (`prefill_chunk_size=N`): `engine/generation/chunked_prefill.py::ChunkedPrefillSession` runs HF dense forwards per chunk, then copies K/V into pages;
- `mixed_batch=True`: each iteration is one forward with every decode token plus prompt chunks (shortest remaining first) under `max_prefill_tokens` total tokens;
- `adaptive_chunking=True` (needs `mixed_batch`): `engine/step_planner.py` sizes chunks from a time limit (150 ms when busy, 400 ms when idle) using a step-time model refit online from measured steps, with aging and a free-memory token cap. This is the recommended default.
- `cuda_graphs=True` (`benchmarks/runners/decode_graph.py`): decode-only iterations replay a graph captured lazily per batch-size bucket (`graph_batch_sizes=(1,2,4,8,16,32)`). Static token/position/length/page-table buffers change contents, never shape; padding rows write only their own scratch KV page (extra pages added to the pool, so request capacity is unchanged). Iterations that carry prompt chunks stay one eager mixed forward unless `graph_mixed_decode=True` splits them into graph decode plus an eager prompt forward. CPU, prefix reuse, non-Triton decode, and batches above the largest bucket fall back to eager; the run summary reports the mode used, replay counts, capture time, and graph memory. The adaptive planner's cost model only learns from eager mixed steps (graph and split steps are a different cost regime).
Prefix reuse is not supported in mixed mode. Design, capture-safety audit, and results: `docs/phase_notes/cuda_graphs.md`.

**Fused kernels.** `engine/kernels/fused.py::install_fused_kernels(model)` routes Qwen3's RMSNorms, MLP SiLU-multiply, post-attention residual add + norm, and RoPE through bit-exact Triton kernels (reversible with `uninstall_fused_kernels`; RoPE is patched process-wide in Transformers' Qwen3 module and `qwen3_paged`). Measured: prefill -10 to -23%, eager decode -20%, graph decode -6 to -8%; see `docs/phase_notes/fused_kernels.md`.

**Benchmark harness.** `benchmarks/core/` defines request/workload schemas and records per-request events (arrival, admission, prefill, every token, completion) against one run clock; `metrics.py` derives TTFT/TPOT/ITL/E2E from those events (definitions in `docs/metric_definitions.md`). Each backend in `benchmarks/runners/` emits the same events, so any two runners are directly comparable. `docs/phase_notes/` holds per-step design, measurements, and known limits.

## Exactness rules

Greedy outputs are validated token for token; one bf16 rounding difference in attention can flip a near-tied token after 36 layers. The mixed/chunked paths stay exact because of these measured invariants. Keep them, or re-verify bitwise on the L4 before changing them:
- The SM89 FP8 linear, RMSNorm, and `lm_head` are row-independent (a row's bits do not depend on how many rows share the call). Prefill logits use one-row `lm_head` calls (`row_logits=True`); decode logits use one batched call, as a decode-only step does.
- Attention must reproduce the exact SDPA call a whole unpadded prompt makes in Transformers 5.14 (no mask, `is_causal`, `enable_gqa`). `aligned_prefill_sdpa` uses lower-right causal FlashAttention for chunks of ≥129 rows (bitwise equal) and zero-padded queries for smaller chunks (FlashAttention splits across keys at ≤128 rows on the L4). The packed SDPA backend delegates to `transformers.integrations.sdpa_attention.sdpa_attention_forward`.
- Decode rows inside a mixed step reuse the decode path's writer and kernel (`_paged_attend`).
- Fused kernels must round where PyTorch rounds: use integer round-to-nearest-even (`_round_bf16`), never `.to(bf16).to(fp32)`, which Triton can fold into an FMA; use `libdevice.mul_rn` where PyTorch has a separate multiply kernel. `rms_norm` reproduces `ATen/native/cuda/Reduce.cuh`'s order for a last-dim fp32 `mean` (4-wide vectors, four accumulators, pairwise tree, threads per row chosen from the row count); a plain `tl.sum` differs on ~1 row in 1,000. Re-run `tests/test_fused_kernels.py` after any PyTorch upgrade.
- On long batched outputs, batch shape alone can flip a near-tied token (HF `generate()` does too), so batched paths are compared against the reference on the validation workloads, not on arbitrary long traces.
- Graph buckets pad the row count of projections and the decode `lm_head`; this is token-identical on the L4 (policy gate), but re-run the gate after changing bucket sizes, kernels, or `lm_head` dispatch. The graph-only `paged_decode_graph` attention flag skips host-side allocator lookups inside the captured region; eager attention keeps its checks and kernels.
New benchmark commands (`run_showcase`, `run_cuda_graphs`, `run_prefill_ttft`) intentionally run no HF reference check; older `run_*` commands verify against the canonical corpus `results/phase1/references_baseline/` (git-ignored; `verify_or_write_reference` *creates* a reference if the directory is empty, which silently self-certifies).

## Measurement pitfalls

- Eager decode through Transformers is host-bound: ~56 ms/step on the `sm89` path (~64-67 ms inside the serving scheduler; ~70 ms with `fp8_kernel_path: auto`) regardless of batch 1–32; GPU-side savings only appear once launch overhead is gone (CUDA Graphs: ~23 ms at batch 1, ~38 ms at batch 32). Measure on the L4 before keeping a change, and compare against HEAD in the same session (noise is about ±3%).
- Keep blocking work (e.g. `nvidia-smi` snapshots) out of the timed window of arrival-scheduled runs; one 25 ms snapshot once changed batch composition. A finished `torch.profiler` session adds ~10 ms per later decode step in the same process.
- Runners hold their KV pool: when running many cases in one process, `close()` the runner, `del` it, `gc.collect()`, and `torch.cuda.empty_cache()` before the next, or memory leaks across cases. Report `torch.cuda.memory_stats()["num_alloc_retries"]`; non-zero means the run was under memory pressure and its timing is suspect.
- Result JSON records `environment.git_commit_sha` and `git_worktree_dirty`.

## Git and repository hygiene

Pushes go to `origin` (`github.com/Makireddyvighnesh/mini-inference-engine`, `main`). Stage specific paths, never `git add -A`. Git-ignored and never committed: `results/`, `MINILLM_L4_PLAN.md` and `README.local.md` (local plan and working status), weights and profiler outputs. The pushed `README.md` is a stable benchmark overview; in-progress status belongs in `README.local.md`, and dated before/after evidence goes in the relevant `docs/phase_notes/` file. `speculative_decoding_lab/` here is a divergent copy of a separate workspace project; confirm which copy is meant before editing it.
