# Benchmark results

Markdown archive of saved benchmark measurements through **2026-10-05**. Phases 0–7 were reconstructed from existing JSON results without rerunning benchmarks. Phase 8 adds new matched L4 measurements and retains its failed exploratory smoke runs.

## Completed phases

| Phase | Results | Saved run selected |
| --- | --- | --- |
| 0 | [Benchmark harness](phase_00_benchmark_harness.md) | Original fixture and corrected 2026-09-26 fixture |
| 1 | [Hugging Face baseline](phase_01_hf_baseline.md) | Corrected full matrix, 2026-09-26 |
| 2 | [Manual decode](phase_02_manual_decode.md) | Full matched matrix, 2026-09-26 |
| 3 | [Explicit KV cache](phase_03_kv_cache.md) | Full cached/control matrix, 2026-09-26; earlier long-context control |
| 4 | [Static batching](phase_04_static_batching.md) | Corrected arrival trace, 2026-09-26; historical stress run |
| 5 | [Continuous batching](phase_05_continuous_batching.md) | Corrected matrix, 2026-09-26 |
| 6 | [Paged KV and CUDA Graph decode](phase_06_paged_kv_and_cuda_graphs.md) | Final 31/31 passing points, 2026-09-30; allocator capacity |
| 7 | [Prefix caching](phase_07_prefix_cache.md) | Sequential pairs and continuous trace, 2026-09-29; strict-reference exception recorded |
| 8 | [Chunked prefill](phase_08_focused_20261004.md) | Focused confirmation: 6/6 cases and 90/90 measured outputs pass, 2026-10-04; [earlier checkpoint](phase_08_chunked_prefill_20261003.md); 64-token chunks remain experimental |

## Additional saved experiments

- [Interpretation and numerical replay of the extended sweep](prefill_decode_analysis_20261005.md): passing prefill/decode/batching comparisons, workload-dependent chunking gains and regressions, and reproduced batch/chunk-sensitive token differences.
- [Extended prefill, isolated decode, static, continuous, and chunked batching sweep](prefill_decode_sweep_20261003.md): completed 2026-10-05; 448 configurations accounted for, with 199 passing, 237 failing correctness, and 12 skipped by the memory guard. Failed configurations do not support performance claims.
- [Earlier LLMPerfLab hardware, memory, precision, and generation measurements](earlier_llmperflab.md).
- [Continuous-batching capacity stress](capacity_stress.md).
- [Corrected decode-step profile](decode_profile.md).
- [Historical long-context vLLM comparison and tuned decode controls](vllm_reference.md).

CUDA Graph and custom-kernel work is covered in Phase 6 and the supplemental experiments. The full Phase 9 and Phase 10 serving milestones are not claimed complete here. Phase 8 is validated for the pinned corpus and default 128-/256-token chunk sizes; its 64-token exploratory case fails the exact-token gate and is not a valid speedup result.

The expanded synthetic corpus adds a chunk-256 failure at 512 prompt tokens and active limits 1/2/4, plus long-output failures across all mixed policies. The original focused checkpoint remains valid for its stated corpus; it does not establish correctness for these expanded settings. The analysis page records fresh HF batch controls and the stronger chunk-specific divergence without changing references.

## Shared environment and metrics

MiniLLM model results use one NVIDIA L4 (SM89, 23,034 MiB) and `Qwen/Qwen3-4B-Instruct-2507-FP8`, revision `8591804019c8b22094c3b5b4454e0edc05dffc98`. Recorded software: Python 3.12.13, PyTorch 2.13.0+cu130, Transformers 5.14.1, Triton 3.7.1. Individual pages state workload, repetitions, numerical policy, and exceptions.

TTFT and E2E are request latencies measured from arrival; TPOT is the average post-first-token interval. P50/P95 are saved percentiles over the measured observations, excluding warm-up. Aggregate TPS is completed output tokens divided by measured run duration. Per-request rates, GPU event times, and model-only TTFT are explicitly labeled when used. Memory in GiB uses 2³⁰ bytes; unavailable measurements use an em dash.

See [metric definitions](../docs/metric_definitions.md). Avoid treating different workloads, request counts, precision paths, or timing boundaries as a single matched speedup series.

## Where the raw results are saved

Paths below are relative to the local `minillm_l4/` checkout. `results/` is ignored by Git; these Markdown files are intentionally outside it and can be committed.

| Evidence | Local saved location |
| --- | --- |
| Initial phase outputs and reference token corpus | `results/phase0/`, `results/phase1/`, `results/phase1/references_baseline/` |
| Corrected harness and HF matrix | `results/phase0_20260926_fixed/`, `results/phase1_20260926/` |
| Manual loop and KV matrix | `results/manual_decode_20260926/`, `results/kv_cache_20260926/` |
| Corrected static and continuous traces | `results/concurrent_repeated_20260926/`, `results/continuous_20260926_mod/`, `results/continuous_20260926_final/` |
| Latest Phase 6 final validation | `results/phase6_fixes_20260930/validation/`, `results/phase6_fixes_20260930/packed_validation/` |
| Allocator experiment | `results/kv_capacity_20260927/kv_capacity.json` |
| Prefix-cache comparisons | `results/prefix_sequential_20260929/`, `results/prefix_trace_20260929/` |
| Corrected profiles | `results/profile_decode_20260927_auto_fixed/`, `results/profile_decode_20260927_sm89_fixed/` |
| Capacity probes and static stress | `results/capacity_stress_*/`, `results/capacity_limit_*/`, `results/stress_batch_final/` |
| Historical vLLM comparison | `results/long_context_batch248_20260920/`, `results/vllm_long_context_batch248_20260920/`, `results/tuned_attention_20260920/` |
| Final chunked-prefill matrix and reproducible sources | `results/phase8_chunked_prefill_final_20261003/`, including `source_snapshot/` and `run_provenance.json` |
| Focused Phase 8 confirmation and independent event/scheduler audit | `results/phase8_focused_20261004/`, including `source_snapshot/`, `run_provenance.json`, and `independent_validation.json` |
| Extended prompt/output/batch matrix, completed with failures and skips | `results/prefill_decode_sweep_20261003/`, including raw JSON/JSONL, phase timings, `hf_references.json`, source snapshot, manifest, and tmux log |

The initial 2026-09-30 archive used a local MiniLLM result store containing 921 JSON files, 469 JSONL files, and 3 CSV files, including manifests, workloads, reference corpora, profiles, and intermediate trials. Each page identifies the result files used for its tables. Raw paths are shown as code because those files are not published on GitHub.

Earlier workspace-level artifacts live in `../results/` (98 JSON files) and `../profiles/`, outside this Git repository. Separate speculative-decoding artifacts also exist in `../speculative_decoding_lab/results/`; the copy inside this repository has `speculative_decoding_lab/results/repo_validation_20260925/`. Those belong to a separate experiment, not the MiniLLM phase sequence.

The 2026-09-26 harness correction moved blocking telemetry out of the timed window. Main Phase 0–5 tables use corrected runs; historical stress/capacity/vLLM tables retain their dates and caveats. The Phase 6 final default-policy validation supersedes the older hybrid failures. Phase 7 paired cached/uncached equality is recorded separately from its historical dense-reference tie failure.

## Phase 0–6 validation and comparison limits

Audited against the saved records on 2026-10-02. All 77 selected primary points have one warm-up and three measured runs. Their TTFT, TPOT, E2E, queue-delay, and prefill percentiles and aggregate output throughput were independently recomputed from raw events and run durations and agreed with the saved summaries. All 831 measured model outputs in Phases 1–6 exactly matched the existing Phase 1 reference corpus. Phase 0 is a CPU fixture and has no model-reference gate. The primary Markdown table values also agree with the saved summaries.

The phase files are in the correct learning order, with the following limits on performance claims:

- Phase 1 versus Phase 2 uses a matched matrix. The primary Phase 4 mixed trace has three requests, while Phase 5 has six; those two tables cannot establish a static-versus-continuous speedup on identical traffic.
- Phases 1–5 use the Transformers `auto` FP8 projection path. The final Phase 6 matrix uses the project's `sm89` path, so a cross-phase latency change cannot be attributed solely to paging. Its graph runs also use a smaller request subset at batches 1 and 2; the page records this population difference. GPU utilization and peak VRAM were not sampled in that final matrix.
- Three measured runs of a small, fixed corpus support development checkpoints. They do not establish production tail latency or performance across independent sessions, workload seeds, and traffic loads. Historical capacity probes additionally include startup costs and are labeled separately.
- Every selected primary run records a dirty worktree. The saved parent commit and dirty flag identify provenance, but the result directories contain no corresponding code patch or source snapshot. These fields alone do not establish the exact code used; reproduction needs the saved changes or evidence that they did not affect the benchmark path. Raw JSON/JSONL is also Git-ignored; committing these summaries alone does not back up the raw evidence.

No GPU benchmarks were rerun for this audit. The capacity page was clarified to state its actual warm-up/repetition protocol and explain the first case's startup and queueing costs.
