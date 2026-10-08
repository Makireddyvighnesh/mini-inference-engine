# Decode-step profile

Saved 2026-09-27. Corrected two-pass command: all unprofiled timings were taken before any profiler session. Random-token prompts; five warm-up steps and twenty timed steps per shape. These runs measure execution time and do not include exact-token correctness checks.

| FP8 path | Prompt | Batch | Unprofiled step P50 (ms) | Profiled GPU busy P50 (ms) | Busy / unprofiled step (%) | GPU operations/step |
| --- | --- | --- | --- | --- | --- | --- |
| Transformers auto | 128 | 1 | 69.97 | 22.66 | 32.4 | 2,081 |
| Transformers auto | 2048 | 1 | 69.75 | 25.37 | 36.4 | 2,081 |
| Transformers auto | 2048 | 4 | 69.53 | 33.21 | 47.8 | 2,081 |
| Project sm89 | 128 | 1 | 55.81 | 22.67 | 40.6 | 2,081 |
| Project sm89 | 2048 | 1 | 55.60 | 25.36 | 45.6 | 2,081 |
| Project sm89 | 2048 | 4 | 55.53 | 33.09 | 59.6 | 2,081 |

GPU busy time comes from profiling; the step denominator comes from the separate unprofiled pass. The 4.11 GiB weight payload has a modeled 14.71 ms bandwidth floor at 300 GB/s.

Saved sources: `results/profile_decode_20260927_auto_fixed/summary.json` and `results/profile_decode_20260927_sm89_fixed/summary.json`. Earlier directories without `_fixed` have profiler-induced timing inflation on later shapes. Notes: [decode profile](../docs/phase_notes/decode_profile.md).
