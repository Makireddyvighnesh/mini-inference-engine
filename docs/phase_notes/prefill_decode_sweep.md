# Extended prefill, decode, and batching sweep

The requested prompt lengths are 128, 256, 512, 1024, 2048, 4096, and 8192.
Output lengths are **128, 512, and 1048**, with 1048 preserved literally
rather than changed to 1024. Batch sizes are **1, 2, 4, and 8**. The configuration is
[`qwen3_fp8_phase_sweep.yaml`](../../configs/workloads/qwen3_fp8_phase_sweep.yaml).

## Timing boundaries

Dedicated `prefill` cases run only the initial prompt forward and first-token
selection. The `isolated` cases run one real static batch with dense prefill
followed by cached decoding. The two phases execute separately; no new prompt
arrives during that decode window. These isolated cases use the native
Transformers DynamicCache. All matched mixed policies use paged KV storage.
Differences between the isolation and serving tables therefore also reflect
the cache backend and should not be attributed solely to scheduling.

- Prefill request wall time ends at the first token and includes preparing
  model inputs, model execution, and token selection. Paged mixed cases also
  include page materialization.
- Prefill forward CUDA elapsed time covers only the model forward, excluding
  input packing, page-copy code, and sampling. Input tokens/s uses real prompt
  tokens divided by this forward elapsed time.
- Pure decode wall time begins after the first token and KV setup. At 1048
  outputs, there are 1047 cached decode steps per sequence. Decode tokens/s
  counts those 1047 tokens, excluding the first token made by prefill.
- Per-forward records also retain CUDA elapsed time and host enqueue time.
  CUDA elapsed time includes stream launch gaps; it is not summed kernel-busy
  time. No profiler is used, avoiding profiler contamination of later runs.
- Decode windows in mixed traces include prefill interruptions and queueing;
  they are explicitly marked as interleaved and are not called pure decode.

## Matched mixed traffic

Each mixed case starts a 128-token prompt with the selected output limit.
Later arrivals alternate between the selected larger prompt length and a
128-token prompt. Their output limits are capped at 128 so finished rows and
new admissions occur while the anchor continues decoding.

The default mixed trace has nine requests, independent of batch limit. Prompt
tokens, output limits, arrivals, model revision, FP8 kernel, paged storage,
decode policy, and page-pool budget are matched between policies. Static
groups cannot admit new arrivals until the group finishes; completed rows
can leave the group's decode work. Static mixed prefill is left-padded, with
padding recorded. Continuous policies admit work between decode steps.

Policies are static FIFO, continuous whole-prompt prefill, and continuous
128-/256-token chunked prefill. The whole-prompt continuous control retains
the existing 256-token admission budget: oversized prompts wait while decode
is active. Both chunked policies obey that same budget per iteration.
Prefix reuse and CUDA Graph replay are disabled.

The isolated batches deliberately repeat the same prompt within a length to
control shape effects. These are synthetic workload studies, not diverse
production traffic or production tail-latency guarantees.

## Execution and evidence

From the parent `LLMPerfLab/` workspace:

```bash
.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_prefill_decode_sweep --dry-run

.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_prefill_decode_sweep \
  --prepare-only --output-dir minillm_l4/results/prefill_decode_sweep_20261003 \
  --markdown-output minillm_l4/benchmark_results/prefill_decode_sweep_20261003.md

.conda-env/bin/python -m minillm_l4.benchmarks.commands.run_prefill_decode_sweep \
  --resume --config minillm_l4/results/prefill_decode_sweep_20261003/configuration.yaml \
  --output-dir minillm_l4/results/prefill_decode_sweep_20261003 \
  --wait-for-idle-gpu
```

The matrix has 448 cases: 28 dedicated prefill cases and 420 generation/serving
cases. It uses one warm-up and three measured runs per case. Full 1048-token
streams and the complete grid make this a long, multi-hour study on the L4.
CLI overrides can select a smaller subset without changing saved runs.

Preparation saves exact workloads, source files and hashes, resolved YAML,
and a Markdown report explicitly stating that no timings were collected.
GPU execution requires exclusive GPU access. It never terminates other jobs.
A conservative memory guard records unavailable large configurations rather
than presenting skipped cases as measured zeros.

Expanded references are generated independently through HF `model.generate`
and saved in this run's `hf_references.json`. The older Phase 1 corpora are
preserved. Every measured output must match its exact reference prefix and
remain stable across repetitions. Failures are retained and labeled.

Every completed case saves JSON, raw event JSONL, forward-phase records,
latency distributions, throughput, GPU/memory telemetry, correctness status,
and scheduler metadata. The manifest and Markdown report checkpoint after
each case. Resume requires the identical configuration and source; it skips
already recorded cases without overwriting their evidence.
