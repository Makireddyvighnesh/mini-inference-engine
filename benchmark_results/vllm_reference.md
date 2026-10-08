# Historical long-context vLLM comparison

Saved 2026-09-20; this is a completed static-shape reference experiment, not sign-off for the full Phase 10 serving comparison. Same pinned FP8 checkpoint; MiniLLM CUDA Graph path, block size 32, 49,152-slot capacity; one warm-up and two measured repetitions. Output lengths are 64 tokens at 3,072 prompt tokens and 32 at 4,608.

Rates below are **per-request** decode/E2E tokens/s, not aggregate batch throughput. Measurements predate the final Phase 6 numerical-policy fixes and the Phase 0 timing correction; model-only run boundaries differ from a general serving benchmark.

The original MiniLLM batch-8 runs created the extended reference corpora rather than checking an independent reference. Batch-2/4 MiniLLM runs and all six vLLM cases pass against those saved corpora. The tuned batch-8 controls also pass against those corpora. This establishes agreement with saved extended-corpus outputs; it does not establish an independent HF-reference gate for the original batch-8 runs.

| Prompt | Batch | Mini TTFT P50 (ms) | vLLM TTFT P50 (ms) | Mini TPOT P50 (ms) | vLLM TPOT P50 (ms) | Mini decode TPS | vLLM decode TPS | Mini E2E TPS | vLLM E2E TPS | Mini reference gate | vLLM saved-reference gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3,072 | 2 | 1,435.71 | 633.78 | 30.35 | 29.09 | 33.47 | 35.35 | 19.12 | 25.94 | pass | pass |
| 3,072 | 4 | 2,899.25 | 1,704.91 | 37.25 | 29.67 | 27.27 | 34.24 | 12.20 | 17.91 | pass | pass |
| 3,072 | 8 | 5,805.63 | 3,361.22 | 51.85 | 37.10 | 19.59 | 27.38 | 7.05 | 11.24 | reference created | pass |
| 4,608 | 2 | 2,266.26 | 1,007.35 | 34.10 | 38.04 | 30.27 | 29.31 | 9.63 | 14.62 | pass | pass |
| 4,608 | 4 | 4,626.84 | 2,633.31 | 46.28 | 33.25 | 22.30 | 31.05 | 5.28 | 8.73 | pass | pass |
| 4,608 | 8 | 9,231.71 | 5,210.98 | 69.58 | 44.30 | 14.83 | 23.30 | 2.81 | 4.86 | reference created | pass |

## Saved tuned batch-8 controls

These later same-day controls use the tuned paged-attention tile/split policy. They are separate from the main matrix.

| Prompt / batch | Before TPOT P50 (ms) | Tuned TPOT P50 (ms) | vLLM TPOT P50 (ms) | Saved-reference gate |
| --- | --- | --- | --- | --- |
| 3,072 / 8 | 51.85 | 39.48 | 37.10 | pass |
| 4,608 / 8 | 69.58 | 47.87 | 44.30 | pass |

Notes: [paged KV](../docs/phase_notes/paged_kv.md).

## Saved sources

Paths are relative to the local `results/` directory. Raw JSON/JSONL stays local. The commit is the recorded parent revision; a dirty flag means the run included local changes.

| Result JSON | Captured (UTC) | Parent commit | Dirty |
| --- | --- | --- | --- |
| `long_context_batch248_20260920/xlong/paged_graph_s32_xlong_b2.json` | 2026-09-20 | not recorded | not recorded |
| `vllm_long_context_batch248_20260920/xlong/xlong_b2.json` | not recorded | not recorded | not recorded |
| `long_context_batch248_20260920/xlong/paged_graph_s32_xlong_b4.json` | 2026-09-20 | not recorded | not recorded |
| `vllm_long_context_batch248_20260920/xlong/xlong_b4.json` | not recorded | not recorded | not recorded |
| `long_context_batch248_20260920/xlong/paged_graph_s32_xlong_b8.json` | 2026-09-20 | not recorded | not recorded |
| `vllm_long_context_batch248_20260920/xlong/xlong_b8.json` | not recorded | not recorded | not recorded |
| `long_context_batch248_20260920/xxlong/paged_graph_s32_xxlong_b2.json` | 2026-09-20 | not recorded | not recorded |
| `vllm_long_context_batch248_20260920/xxlong/xxlong_b2.json` | not recorded | not recorded | not recorded |
| `long_context_batch248_20260920/xxlong/paged_graph_s32_xxlong_b4.json` | 2026-09-20 | not recorded | not recorded |
| `vllm_long_context_batch248_20260920/xxlong/xxlong_b4.json` | not recorded | not recorded | not recorded |
| `long_context_batch248_20260920/xxlong/paged_graph_s32_xxlong_b8.json` | 2026-09-20 | not recorded | not recorded |
| `vllm_long_context_batch248_20260920/xxlong/xxlong_b8.json` | not recorded | not recorded | not recorded |
| `tuned_attention_20260920/paged_graph_s32_xlong_b8.json` | 2026-09-20 | not recorded | not recorded |
| `tuned_attention_20260920/paged_graph_s32_xxlong_b8.json` | 2026-09-20 | not recorded | not recorded |
