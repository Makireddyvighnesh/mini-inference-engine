# Prefill, decode, static batching, and continuous batching sweep

Status: **completed_with_failures_or_skips**. Created 2026-10-06T14:11:07.453623+00:00. Model `Qwen/Qwen3-4B-Instruct-2507-FP8` at `8591804019c8b22094c3b5b4454e0edc05dffc98`.

Dedicated pure prefill generates only the first token and has no cached decode. Isolated static generation separates prefill from decode after KV setup and the first token; 1048 outputs contain 1047 cached decode steps. These isolation rows use the native Transformers DynamicCache, while the matched mixed policies all use paged KV. Differences between the isolated and serving rows therefore include the cache backend. CUDA forward elapsed time excludes token sampling and page-copy code, includes stream launch gaps, and is not kernel-busy time. Request wall timings include scheduling and output synchronization.

Static and continuous mixed rows use identical request lists, arrival times, output limits, FP8 projections, paged KV storage, and decode numerics. The mixed profile has a long decoding anchor and later requests capped at 128 outputs. All row-generation limits stay at or below the reported cap. Prefix reuse and CUDA Graph replay are disabled.

Repeated prompts within an isolated batch are intentional shape controls, not a diverse serving corpus. Case order is fixed. Raw records preserve P50/P95/P99, repeat variance, forward phase records, model outputs, telemetry, references, and source snapshots. Failed or contended rows are not accepted performance claims.

## Isolated prefill and decode

| Prompt | Batch | Output tokens | Mode | Prefill wall P50 (ms) | Prefill forward CUDA (ms) | Prefill input tokens/s | Decode wall (ms) | Decode tokens/s | TPOT P50 (ms) | Exact HF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Matched mixed traces

| Largest prompt | Batch limit | Generation cap | Mode | TTFT P50 / P95 (ms) | ITL P95 / P99 (ms) | E2E P95 (ms) | Output tokens/s | GPU P50 (%) | Peak allocated GiB | Exact HF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 1 | 128 | static | 3,989.72 / 8,102.66 | 64.45 / 65.05 | 16,116.47 | 15.81 | 38.00 | 4.35 | pass |
| 512 | 1 | 128 | continuous | 98.59 / 132.07 | 64.62 / 64.97 | 8,201.12 | 30.98 | 39.00 | 4.35 | pass |
| 512 | 1 | 128 | chunked_256 | 149.34 / 233.11 | 64.44 / 65.35 | 8,221.96 | 30.89 | 39.00 | 4.34 | pass |
| 512 | 1 | 128 | chunked_512 | 97.04 / 130.77 | 64.47 / 65.22 | 8,189.42 | 31.22 | 39.00 | 4.35 | pass |
| 512 | 2 | 128 | static | 7,985.63 / 8,109.30 | 64.03 / 64.72 | 16,019.61 | 23.84 | 39.00 | 4.53 | pass |
| 512 | 2 | 128 | continuous | 130.77 / 158.49 | 65.24 / 66.41 | 8,294.58 | 45.71 | 39.00 | 4.39 | pass |
| 512 | 2 | 128 | chunked_256 | 231.74 / 257.26 | 64.86 / 98.47 | 8,357.01 | 45.15 | 39.00 | 4.37 | pass |
| 512 | 2 | 128 | chunked_512 | 131.41 / 158.97 | 65.08 / 66.65 | 8,322.93 | 45.48 | 39.00 | 4.39 | pass |
| 512 | 4 | 128 | static | 8,100.52 / 8,312.06 | 64.52 / 65.48 | 16,273.44 | 39.14 | 39.00 | 4.84 | pass |
| 512 | 4 | 128 | continuous | 155.89 / 221.06 | 64.82 / 123.95 | 8,443.32 | 74.17 | 40.00 | 4.51 | pass |
| 512 | 4 | 128 | chunked_256 | 252.59 / 535.41 | 64.67 / 132.54 | 8,594.82 | 72.17 | 40.00 | 4.49 | pass |
| 512 | 4 | 128 | chunked_512 | 156.01 / 247.66 | 64.91 / 124.44 | 8,428.41 | 73.79 | 40.00 | 4.51 | pass |
| 512 | 8 | 128 | static | 8,455.20 / 8,852.75 | 64.61 / 65.20 | 16,880.58 | 67.79 | 40.00 | 5.52 | pass |
| 512 | 8 | 128 | continuous | 206.08 / 272.25 | 64.89 / 226.45 | 8,722.63 | 127.38 | 42.00 | 4.76 | pass |
| 512 | 8 | 128 | chunked_256 | 421.01 / 1,100.84 | 65.58 / 189.31 | 9,264.37 | 117.82 | 41.00 | 4.79 | pass |
| 512 | 8 | 128 | chunked_512 | 242.06 / 401.60 | 64.76 / 192.94 | 8,800.90 | 125.70 | 42.00 | 4.76 | pass |
| 2048 | 1 | 128 | static | 4,187.63 / 8,362.86 | 63.58 / 64.29 | 16,250.01 | 15.67 | 40.00 | 4.89 | pass |
| 2048 | 1 | 128 | continuous | 294.03 / 524.77 | 63.94 / 64.40 | 8,491.13 | 29.94 | 42.00 | 4.89 | pass |
| 2048 | 1 | 128 | chunked_256 | 555.36 / 1,053.87 | 64.99 / 136.46 | 9,068.92 | 28.08 | 43.00 | 4.81 | pass |
| 2048 | 1 | 128 | chunked_512 | 374.77 / 685.09 | 64.44 / 173.75 | 8,671.10 | 29.26 | 42.00 | 4.82 | pass |
| 2048 | 2 | 128 | static | 8,862.71 / 8,986.91 | 63.80 / 64.28 | 16,940.57 | 22.63 | 41.00 | 5.37 | pass |
| 2048 | 2 | 128 | continuous | 524.55 / 549.43 | 63.67 / 65.02 | 8,508.31 | 44.54 | 43.00 | 4.93 | pass |
| 2048 | 2 | 128 | chunked_256 | 247.84 / 1,226.30 | 65.36 / 139.58 | 9,235.13 | 41.25 | 43.00 | 4.84 | pass |
| 2048 | 2 | 128 | chunked_512 | 326.25 / 845.69 | 64.64 / 170.35 | 8,846.02 | 43.02 | 43.00 | 4.85 | pass |
| 2048 | 4 | 128 | static | 10,042.23 / 10,270.55 | 65.18 / 65.63 | 18,379.14 | 34.67 | 42.00 | 6.57 | pass |
| 2048 | 4 | 128 | continuous | 549.11 / 946.70 | 65.13 / 69.26 | 9,237.82 | 68.47 | 44.00 | 5.26 | pass |
| 2048 | 4 | 128 | chunked_256 | 502.87 / 2,237.07 | 132.18 / 151.71 | 10,366.25 | 60.42 | 44.00 | 5.38 | pass |
| 2048 | 4 | 128 | chunked_512 | 517.21 / 1,468.78 | 65.92 / 193.78 | 9,623.24 | 64.99 | 43.00 | 5.35 | pass |
| 2048 | 8 | 128 | static | 12,242.83 / 12,634.98 | 65.22 / 65.87 | 20,747.10 | 55.21 | 46.00 | 8.99 | pass |
| 2048 | 8 | 128 | continuous | 947.61 / 1,732.45 | 65.19 / 623.68 | 10,391.07 | 108.77 | 47.00 | 5.93 | pass |
| 2048 | 8 | 128 | chunked_256 | 744.43 / 4,507.72 | 141.18 / 191.10 | 12,534.81 | 88.34 | 47.00 | 6.40 | pass |
| 2048 | 8 | 128 | chunked_512 | 894.30 / 2,877.77 | 178.21 / 246.38 | 11,165.86 | 99.59 | 47.00 | 6.32 | pass |
| 4096 | 1 | 128 | static | 4,534.87 / 9,034.03 | 63.60 / 64.08 | 16,949.05 | 15.04 | 45.00 | 5.60 | pass |
| 4096 | 1 | 128 | continuous | 616.27 / 1,169.42 | 63.60 / 64.90 | 9,158.75 | 28.01 | 47.00 | 5.60 | pass |
| 4096 | 1 | 128 | chunked_256 | 1,325.38 / 2,591.04 | 131.67 / 206.38 | 10,619.14 | 23.89 | 46.00 | 5.43 | pass |
| 4096 | 1 | 128 | chunked_512 | 872.37 / 1,704.93 | 66.87 / 233.83 | 9,971.30 | 25.47 | 45.00 | 5.44 | pass |
| 4096 | 2 | 128 | static | 10,608.52 / 10,811.93 | 66.11 / 66.87 | 18,904.11 | 20.33 | 44.00 | 6.54 | pass |
| 4096 | 2 | 128 | continuous | 1,170.59 / 1,207.65 | 66.00 / 67.49 | 9,424.35 | 40.24 | 45.00 | 5.64 | pass |
| 4096 | 2 | 128 | chunked_256 | 257.22 / 2,844.01 | 148.64 / 210.47 | 10,982.68 | 34.82 | 46.00 | 5.46 | pass |
| 4096 | 2 | 128 | chunked_512 | 333.19 / 1,892.56 | 68.14 / 236.69 | 10,093.94 | 37.67 | 45.00 | 5.46 | pass |
| 4096 | 4 | 128 | static | 13,359.24 / 13,578.65 | 66.55 / 67.23 | 21,829.17 | 29.23 | 47.00 | 8.91 | pass |
| 4096 | 4 | 128 | continuous | 1,207.16 / 2,255.81 | 67.07 / 68.43 | 10,809.49 | 58.88 | 47.00 | 6.25 | pass |
| 4096 | 4 | 128 | chunked_256 | 510.66 / 5,430.55 | 189.40 / 230.57 | 13,804.98 | 45.77 | 47.00 | 6.58 | pass |
| 4096 | 4 | 128 | chunked_512 | 525.46 / 3,567.77 | 207.38 / 269.57 | 11,957.78 | 52.49 | 47.00 | 6.51 | pass |
| 4096 | 8 | 128 | static | 18,439.23 / 18,830.55 | 66.04 / 66.41 | 26,999.46 | 42.55 | 53.00 | 13.67 | pass |
| 4096 | 8 | 128 | continuous | 2,259.85 / 4,289.47 | 65.49 / 67.76 | 13,019.20 | 88.03 | 55.00 | 7.48 | pass |
| 4096 | 8 | 128 | chunked_256 | 765.52 / 10,936.38 | 213.70 / 322.59 | 19,101.97 | 58.66 | 55.00 | 8.68 | pass |
| 4096 | 8 | 128 | chunked_512 | 903.67 / 7,003.28 | 242.30 / 403.79 | 15,441.95 | 72.79 | 54.00 | 8.45 | pass |
| 8192 | 1 | 128 | static | 5,233.87 / 10,516.79 | 64.28 / 64.88 | 18,463.58 | 13.79 | 53.00 | 7.03 | pass |
| 8192 | 1 | 128 | continuous | 1,323.99 / 2,588.11 | 64.77 / 65.57 | 10,678.06 | 23.90 | 54.00 | 7.03 | pass |
| 8192 | 1 | 128 | chunked_256 | 4,266.73 / 8,507.86 | 285.77 / 461.35 | 16,749.44 | 15.22 | 53.00 | 6.68 | pass |
| 8192 | 1 | 128 | chunked_512 | 2,594.98 / 5,124.01 | 203.02 / 472.41 | 13,376.95 | 19.04 | 53.00 | 6.69 | pass |
| 8192 | 2 | 128 | static | 15,150.60 / 15,259.29 | 65.80 / 66.32 | 23,464.62 | 16.30 | 52.00 | 8.92 | pass |
| 8192 | 2 | 128 | continuous | 2,608.44 / 2,664.47 | 66.60 / 71.01 | 10,992.22 | 34.53 | 52.50 | 7.07 | pass |
| 8192 | 2 | 128 | chunked_256 | 261.39 / 8,791.42 | 340.52 / 476.72 | 16,868.76 | 22.67 | 55.00 | 6.71 | pass |
| 8192 | 2 | 128 | chunked_512 | 335.05 / 5,493.36 | 274.56 / 492.16 | 13,657.84 | 28.02 | 53.50 | 6.71 | pass |
| 8192 | 4 | 128 | static | 22,026.19 / 22,233.73 | 65.83 / 66.30 | 30,423.02 | 20.98 | 57.00 | 13.69 | pass |
| 8192 | 4 | 128 | continuous | 2,631.02 / 5,113.06 | 66.16 / 68.78 | 13,582.45 | 46.65 | 58.00 | 8.24 | pass |
| 8192 | 4 | 128 | chunked_256 | 522.62 / 17,431.09 | 413.77 / 505.63 | 25,679.75 | 24.76 | 67.00 | 8.95 | pass |
| 8192 | 4 | 128 | chunked_512 | 534.46 / 10,819.21 | 401.64 / 539.49 | 19,113.86 | 33.09 | 61.50 | 8.88 | pass |
| 8192 | 8 | 128 | static | 70.19 / 70.34 | 64.87 / 65.39 | 8,140.33 | 0.00 | 38.00 | 19.06 | fail |
| 8192 | 8 | 128 | continuous | 5,162.31 / 10,077.32 | 67.04 / 69.53 | 19,026.03 | 60.21 | 68.00 | 10.60 | pass |
| 8192 | 8 | 128 | chunked_256 | 762.78 / 34,963.38 | 468.78 / 513.29 | 43,387.48 | 26.30 | 81.00 | 13.31 | pass |
| 8192 | 8 | 128 | chunked_512 | 926.55 / 21,578.00 | 489.73 / 935.80 | 30,143.12 | 37.67 | 79.00 | 13.16 | pass |

## Availability and failures

- `p8192_g128_b8_static`: fail; 

Raw evidence: `minillm_l4/results/resource_admission_sweep_20261006`.
