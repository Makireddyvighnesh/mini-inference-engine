# Continuous-batching capacity stress

Historical L4 capacity probes saved 2026-09-15, before the 2026-09-26 harness timing fix. TTFT includes the earlier telemetry overhead; throughput is not a corrected serving benchmark. These burst probes use varying queue depths and are not a steady-state throughput sweep. Capacity and admission counts remain useful.

The original capacity probes use **no warm-up and one measured run per case**. Their run-level TPS P50 is therefore the rate from that single run, and startup costs are included. Only the separately labeled `interleave_warmed` controls use one warm-up and three measured runs.

## Active-request capacity

| Case | Requests | Active limit | Prefill budget | Peak active | Max prefill tokens | Prefills during decode | Historical TTFT P50 (ms) | Historical TPS P50 | Exact prefix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `capacity_stress_concurrency/concurrency_b1` | 16 | 1 | 2048 | 1 | 128 | 0 | 20,294.74 | 8.52 | pass |
| `capacity_stress_concurrency/concurrency_b2` | 16 | 2 | 2048 | 2 | 256 | 0 | 4,218.84 | 26.94 | pass |
| `capacity_stress_concurrency/concurrency_b4` | 16 | 4 | 2048 | 4 | 512 | 0 | 1,896.88 | 52.80 | pass |
| `capacity_stress_concurrency/concurrency_b8` | 16 | 8 | 2048 | 8 | 1024 | 0 | 755.84 | 97.27 | pass |
| `capacity_stress_concurrency/concurrency_b16` | 16 | 16 | 2048 | 16 | 2048 | 0 | 285.00 | 147.69 | pass |
| `capacity_limit_concurrency/concurrency_b24` | 32 | 24 | 8192 | 24 | 3072 | 0 | 5,308.01 | 9.40 | pass |
| `capacity_limit_concurrency/concurrency_b32` | 32 | 32 | 8192 | 32 | 4096 | 0 | 450.63 | 93.06 | pass |
| `capacity_limit_concurrency/concurrency_b48` | 32 | 48 | 8192 | 32 | 4096 | 0 | 448.47 | 174.24 | pass |
| `capacity_limit_concurrency/concurrency_b64` | 32 | 64 | 8192 | 32 | 4096 | 0 | 447.66 | 174.93 | pass |
| `capacity_limit_concurrency_64req/concurrency_b24` | 64 | 24 | 8192 | 24 | 3072 | 0 | 12,587.95 | 24.43 | pass |
| `capacity_limit_concurrency_64req/concurrency_b32` | 64 | 32 | 8192 | 32 | 4096 | 0 | 1,169.82 | 174.74 | pass |
| `capacity_limit_concurrency_64req/concurrency_b48` | 64 | 48 | 8192 | 48 | 6144 | 0 | 1,375.79 | 26.29 | pass |
| `capacity_limit_concurrency_64req/concurrency_b64` | 64 | 64 | 8192 | 64 | 8192 | 0 | 934.52 | 201.79 | pass |
| `capacity_limit_128/concurrency_b96` | 128 | 96 | 16384 | 96 | 12288 | 0 | 8,517.25 | 22.21 | pass |
| `capacity_limit_128/concurrency_b128` | 128 | 128 | 16384 | 128 | 16384 | 0 | 1,904.18 | 122.13 | pass |

For example, `concurrency_b1` submits 16 requests together, each with a 128-token prompt and 16 output tokens, but allows only one active request. The recorded first request takes 12.72 seconds to finish, including 5.12 seconds of prefill and a 6.52-second first decode interval. Later requests wait in the queue and take roughly 1.15 seconds each. Its 20.29-second median TTFT includes this waiting and startup delay; its 8.52 TPS is 256 output tokens divided by the 30.05-second run. This is a capacity/correctness checkpoint, not a warmed-up latency result.

## New-prefill token budget

| Case | Requests | Active limit | Prefill budget | Peak active | Max prefill tokens | Prefills during decode | Historical TTFT P50 (ms) | Historical TPS P50 | Exact prefix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `capacity_stress_input_budget/input_budget_t2048` | 8 | 8 | 2048 | 8 | 2048 | 7 | 12,964.59 | 7.44 | pass |
| `capacity_stress_input_budget/input_budget_t4096` | 8 | 8 | 4096 | 8 | 4096 | 3 | 1,959.73 | 20.39 | pass |
| `capacity_stress_input_budget/input_budget_t8192` | 8 | 8 | 8192 | 8 | 8192 | 1 | 1,974.25 | 20.18 | pass |
| `capacity_stress_input_budget/input_budget_t16384` | 8 | 8 | 16384 | 8 | 16384 | 0 | 1,971.87 | 23.71 | pass |
| `capacity_limit_input/input_budget_t16384` | 16 | 16 | 16384 | 16 | 16384 | 1 | 14,360.78 | 2.97 | pass |
| `capacity_limit_input/input_budget_t32768` | 16 | 16 | 32768 | 16 | 32768 | 0 | 3,975.55 | 7.59 | pass |

## Prefill while decode is active

| Case | Requests | Active limit | Prefill budget | Peak active | Max prefill tokens | Prefills during decode | Historical TTFT P50 (ms) | Historical TPS P50 | Exact prefix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `capacity_stress_interleave/interleave_b4` | 12 | 4 | 4096 | 4 | 3200 | 1 | 16,574.81 | 8.50 | pass |
| `capacity_stress_interleave/interleave_b8` | 12 | 8 | 4096 | 8 | 3328 | 3 | 3,764.25 | 18.36 | pass |
| `interleave_warmed/interleave_b4` | 12 | 4 | 4096 | 4 | 3200 | 1 | 4,741.86 | 17.88 | pass |
| `interleave_warmed/interleave_b8` | 12 | 8 | 4096 | 8 | 3328 | 3 | 3,796.51 | 18.24 | pass |

Measured capacity reached 128 active requests for the short-prompt/four-output-token probe, and 32,768 input tokens in one prefill across 16 long requests. These are tested capacities, not hardware maxima.

Notes: [capacity stress](../docs/phase_notes/capacity_stress.md).

Saved manifests:

- `results/capacity_stress_concurrency/capacity_stress_manifest.json`
- `results/capacity_limit_concurrency/capacity_stress_manifest.json`
- `results/capacity_limit_concurrency_64req/capacity_stress_manifest.json`
- `results/capacity_limit_128/capacity_stress_manifest.json`
- `results/capacity_stress_input_budget/capacity_stress_manifest.json`
- `results/capacity_limit_input/capacity_stress_manifest.json`
- `results/capacity_stress_interleave/capacity_stress_manifest.json`
- `results/interleave_warmed/capacity_stress_manifest.json`
