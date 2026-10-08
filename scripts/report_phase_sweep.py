"""Render the audited phase sweep into the requested GitHub Markdown format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--markdown', type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.analysis.read_text())
    diagnostic = json.loads(args.diagnostic.read_text())
    assert len(diagnostic['probes']) == 8 and diagnostic['gpu_exclusive_at_end']
    cases = {c['key']: c for c in audit['cases']}
    lengths = [128, 256, 512, 1024, 2048, 4096, 8192]
    batches = [1, 2, 4, 8]
    fmt = lambda x: f'{x:,.2f}'

    def point(p, g, b, mode):
        return cases[f'p{p}_g{g}_b{b}_{mode}']

    def valid(p, g, b, mode):
        c = point(p, g, b, mode)
        assert c['status'] == 'pass', c['key']
        return c

    def table(headers, rows):
        return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows], '']

    lines = ['# Prefill, decode, and batching — interpretation of the completed sweep', '',
             'Reviewed 2026-10-05. Model: `Qwen/Qwen3-4B-Instruct-2507-FP8`, revision '
             '`8591804019c8b22094c3b5b4454e0edc05dffc98`, one NVIDIA L4. '
             '[Full measurement tables](prefill_decode_sweep_20261003.md).', '',
             'Chunked prefill improves throughput and admission latency for several batch-4/8 mixed workloads, '
             'but it increases occasional decode gaps and can regress at low concurrency. '
             'The broad matrix has unresolved correctness failures, so only passing cases below support performance comparisons. '
             'The focused Phase 8 checkpoint remains scoped to its original corpus.', '',
             '## What is validated', '',
             'All 448 configurations are accounted for: 199 pass, 237 fail exact-token correctness, and 12 are skipped '
             'by the memory guard. All 436 executed cases finish successfully at the requested output lengths and '
             'produce identical tokens across their three measured repetitions. These are numerical-output mismatches, '
             'rather than execution crashes or early truncation.', '',
             f"The raw audit checked {audit['validation']['measured_runs_checked']:,} measured runs and "
             f"{audit['validation']['measured_outputs_checked']:,} outputs. Of those outputs, "
             f"{audit['validation']['hf_matching_outputs']:,} match HF exactly. A case fails if any output differs. "
             'Event-derived latency percentiles, aggregate throughput, isolated phase windows, and CUDA forward sums '
             f"were independently recomputed for every executed case. All {audit['snapshot_hashes_checked']} "
             'snapshot hashes and saved prompt/reference hashes match.', '',
             '## Pure prefill', '',
             'Each row performs one prompt forward plus first-token selection, with no cached decode. '
             'Numbers are request prefill wall milliseconds, median across the three measured runs. '
             'All 28 points pass exact HF checks.', '']
    lines += table(['Prompt tokens', 'Batch 1 (ms)', 'Batch 2 (ms)', 'Batch 4 (ms)', 'Batch 8 (ms)'],
                   [[p, *[fmt(valid(p, 1, b, 'prefill')['phase_summary']['prefill_request_wall_ms_p50']) for b in batches]] for p in lengths])
    lines += ['Small prompts benefit from batching overhead amortization. At 128 prompt tokens, input throughput rises '
              'from about 2,206 tokens/s at batch 1 to 6,000 at batch 4. At 8192 tokens, batch 1 and batch 8 '
              'deliver about 3,810 and 3,698 input tokens/s respectively, with GPU utilization at 100%. '
              'Long-prompt prefill already fills the GPU, so more batch rows mainly add work.', '',
              '## Isolated cached decode', '',
              'All points below pass, use native Transformers DynamicCache, and generate 128 outputs per row. '
              'Decode throughput counts only the 127 cached steps after the first token and excludes prompt setup. '
              'This backend differs from the paged backend in the mixed-policy tables.', '']
    lines += table(['Prompt tokens', 'Batch 1 decode TPS', 'Batch 2 decode TPS', 'Batch 4 decode TPS', 'Batch 8 decode TPS'],
                   [[p, *[fmt(valid(p, 128, b, 'isolated')['phase_summary']['decode_window_tokens_per_second_p50']) for b in batches]] for p in lengths])
    lines += ['At 512 prompt tokens, batch 8 reaches 142.58 decode tokens/s versus 17.95 at batch 1. '
              'At 8192 tokens, batch 4 and batch 8 reach 46.61 and 53.55 decode tokens/s; GPU utilization is '
              '99–100%, and per-sequence TPOT rises from 85.81 to 149.40 ms. Batching still raises aggregate throughput, '
              'but long contexts limit the gain and increase each stream’s token latency.', '',
              'The 128-token/batch-4 point is slower than nearby shapes, and its aggregate end-to-end rates vary '
              'from 48.23 to 53.16 tokens/s across repeats. The matrix spans several sessions and dates with fixed '
              'case order; it does not establish a smooth universal scaling curve or statistically significant small deltas. '
              'For short contexts, host enqueue time is close to CUDA forward elapsed time, consistent with substantial '
              'host dispatch overhead. That ratio is not a percentage of runtime spent on CPU: CUDA intervals include launch gaps.', '',
              '## Matched mixed policies', '',
              'Each comparison uses the same nine scheduled requests, 100 ms apart, with the same prompt lengths, '
              'output caps, FP8 projection policy, paged KV, and active limit. In the valid rows below the generation '
              'cap is 128, so every request emits 128 outputs. The continuous whole-prompt control uses the existing '
              '256-token admission budget: an oversized prompt waits while decode is active. Its throughput therefore '
              'depends on that budget policy.', '']
    mixed_rows = []
    for p, b in [(512, 8), (2048, 4), (2048, 8), (4096, 8), (8192, 2), (8192, 4)]:
        selected = [valid(p, 128, b, mode) for mode in ('static', 'continuous', 'chunked_128', 'chunked_256')]
        mixed_rows.append([p, b, *[fmt(c['output_tps_p50']) for c in selected],
                           f"{100 * (selected[3]['output_tps_p50'] / selected[1]['output_tps_p50'] - 1):+.1f}%"])
    lines += table(['Largest prompt', 'Active limit', 'Static TPS', 'Continuous TPS', 'Chunk 128 TPS', 'Chunk 256 TPS', 'Chunk 256 vs continuous'], mixed_rows)
    lines += ['At 4096 tokens and active limit 8, chunk 256 is 105.6% faster than the continuous control and '
              '24.4% faster than static batching. At 8192 tokens and active limit 2, chunk 128 is 21.3% slower '
              'than the continuous control, while chunk 256 is 5.5% slower. The implementation makes many more small '
              'prefill forwards; their added overhead is a plausible contributor to the low-concurrency regressions. The evidence '
              'supports workload-dependent chunking rather than one universal preset.', '',
              '### Latency trade-off: 2048-token mixed trace, active limit 4', '']
    rows = []
    for mode in ('static', 'continuous', 'chunked_128', 'chunked_256'):
        c = valid(2048, 128, 4, mode)
        rows.append([mode, fmt(c['output_tps_p50']), fmt(c['metrics']['ttft_ms']['p50'] / 1000),
                     fmt(c['metrics']['itl_ms']['p95']), fmt(c['metrics']['e2e_latency_ms']['p95'] / 1000)])
    lines += table(['Policy', 'Output TPS', 'TTFT P50 (s)', 'Token gap P95 (ms)', 'E2E P95 (s)'], rows)
    lines += ['Chunk 256 improves throughput from 27.63 to 43.10 tokens/s against the continuous control, '
              'reduces TTFT P50 from 16.52 to 8.94 seconds, and reduces E2E P95 from 41.04 to 25.81 seconds. '
              'P95 token gaps rise from 63.80 to 133.96 ms. Relative to static batching, its throughput gain is '
              '7.8%, with a similar token-gap trade-off. The earlier focused Phase 8 trace uses different request '
              'counts, output lengths, and arrivals; its larger speedup must not be substituted for this comparison.', '',
              '## Correctness failure patterns', '',
              'The 237 failed configurations separate into three groups:', '',
              '- 216 mixed configurations with generation caps 512/1048 fail on the common 128-token decoding anchor. '
              'All executed mixed cases at these caps fail, including static and whole-prompt controls.',
              '- 18 native isolated configurations fail: prompt lengths 128/256/512, output caps 512/1048, '
              'and batches 2/4/8. Native batch 1 passes those lengths.',
              '- Three additional 128-output configurations fail: the 512-token mixed trace with chunk 256 and '
              'active limits 1/2/4. The same chunk-specific failure also appears in the already-failing longer-cap cases.', '',
              'The 1,008 differing measured request outputs have only five first-divergence signatures. '
              'Indices below are zero-based; this is a grouping of observations, not proof of five independent root causes.', '']
    mismatch_table = []
    for signature, count in audit['first_mismatch_patterns'].items():
        # The audit serializes only integer tuples generated by the reader.
        import ast
        index, actual, expected = ast.literal_eval(signature)
        mismatch_table.append([index, actual, expected, count])
    lines += table(['First output index', 'Actual token ID', 'HF token ID', 'Measured output occurrences'], mismatch_table)
    lines += ['### Targeted L4 replay', '',
              'Eight fresh probes ran from the original frozen source and original saved prompt/reference inputs. '
              'Logit hooks synchronize and add work; none of these probe durations are performance measurements.', '']
    rows = []
    for probe in diagnostic['probes']:
        diff = probe['first_difference']
        rows.append([probe['name'], probe['output_cap'], 'pass' if probe['matches_single_request_saved_hf'] else 'FAIL',
                     '—' if diff is None else diff['output_index'], '—' if diff is None else f"{diff['top2_gap']:.4f}"])
    lines += table(['Probe', 'Outputs per row', 'Exact saved batch-1 HF', 'First output difference', 'Winning logit margin at difference'], rows)
    lines += ['Fresh HF batch 1 reproduces the saved reference prefixes through the tested limits. HF batch 2 itself reproduces the 128-token prompt’s '
              'difference at output index 365, changing token 334 to 5501. Before that point the output prefixes match. '
              'At index 365, HF batch 1 chooses token 334 with a 0.0625 BF16-logit margin, while HF batch 2 chooses '
              '5501 with margin 0.125. This directly demonstrates batch-dependent model numerics for this example; '
              'it does not establish that every paged failure has the same cause.', '',
              'Whole-prompt paged decoding reproduces that long-output difference at index 365. For the 512-token '
              'prompt, HF batches 1/2, whole-prompt paged decode, and chunk 128 all match the saved prefix through '
              'the tested limits. Chunk 256 alone diverges at index 106: token 9608 wins over expected token 15003 '
              'by 3.5 logit units. The full-vector maximum difference from HF at this point is 5.03125. '
              'This is not an exact tie and is not waived as harmless rounding. Its precise source in prefill, '
              'stored KV, attention numerics, or their interaction remains unresolved.', '',
              'Exact output equality also does not prove logit equality: the passing chunk-128 probe differs from HF '
              'by up to 2.75 logit units at output index 106. The whole-prompt paged probe also retains the expected '
              'winner there while its winning logit is 33.75 versus 22.5 in HF. The present benchmark gate verifies tokens; a '
              'teacher-forced, layer/cache comparison is the appropriate next diagnostic for numerical equivalence.', '',
              '## Memory and scope', '',
              'All 12 skips are the mixed 8192-token, active-limit-8 configurations, across the four policies and '
              'three output caps. The conservative guard estimates paged pool capacity, retained dense prefill KV, '
              'and workspace. These are predicted memory skips, not observed OOM failures or measured zeros. '
              'The isolated native batch-8 cases fit because they do not duplicate the cache in a paged pool.', '',
              'The corpus uses repeated synthetic prompts and fixed arrival intervals, exact-length generation with '
              'EOS disabled, one full warm-up, and three measured repeats. It is real model inference under controlled '
              'shapes, rather than HTTP serving, diverse production traffic, or a production tail-latency study. '
              'Per-forward CUDA elapsed time includes stream launch gaps and excludes token sampling/page-copy work.', '',
              '## Implications for the next checkpoint', '',
              '1. Preserve the strict gate and the failed results. Do not publish the full matrix as an all-correct '
              'Phase 8 speedup result or change reference tokens to match the implementation.',
              '2. Diagnose the chunk-256/512-token path and long paged decoding before broad performance tuning. '
              'Use batch-matched HF controls and teacher-forced prefill/KV/attention comparisons to distinguish '
              'shape-dependent model behavior from implementation error. Keep the original batch-1 comparison visible.',
              '3. Chunk 128 is the safer tested short-output preset in this sweep, with all 27 executed '
              '128-output mixed configurations passing. Its low-concurrency regressions still argue for a '
              'workload-aware scheduling choice. Chunk 256 has useful throughput gains, but the new failing corpus '
              'prevents promoting it as a generally validated preset.',
              '4. For future grids, check representative correctness first, screen supported shapes, and spend '
              'repeat runs on meaningful comparisons. Short-context host dispatch and long-context GPU saturation '
              'need different optimization evidence; this sweep contains both.', '',
              '## Inspectable evidence', '',
              '- Source measurements: `results/prefill_decode_sweep_20261003/`, including raw request events, '
              'outputs, forward records, source snapshot, resolved configuration, and independent HF references.',
              '- Full audit: `results/phase_sweep_review_20261005/analysis.json`.',
              '- Logit replay: `results/phase_sweep_review_20261005/logit_diagnostic/diagnostic.json` and '
              '`selected_logits.pt`; original references are unchanged.',
              '- Reproducible readers: [analyze_phase_sweep.py](../scripts/analyze_phase_sweep.py), '
              '[inspect_sweep_logits.py](../scripts/inspect_sweep_logits.py), and '
              '[report_phase_sweep.py](../scripts/report_phase_sweep.py).', '']

    # Publication figure: preserve the exact gating used by the tables.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors = ['#2563eb', '#c58a0c', '#db6b35', '#61793a']
    markers = ['o', 's', '^', 'D']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Passing prefill, decode, and mixed batching measurements', fontsize=15, fontweight='bold')
    for b, color, marker in zip(batches, colors, markers, strict=True):
        axes[0].plot(lengths, [valid(p, 1, b, 'prefill')['phase_summary']['prefill_request_wall_ms_p50'] for p in lengths],
                     label=f'Batch {b}', color=color, marker=marker)
        axes[1].plot(lengths, [valid(p, 128, b, 'isolated')['phase_summary']['decode_window_tokens_per_second_p50'] for p in lengths],
                     label=f'Batch {b}', color=color, marker=marker)
    for mode, color, marker in zip(('static', 'continuous', 'chunked_128', 'chunked_256'), colors, markers, strict=True):
        selected = [point(p, 128, 4, mode) for p in lengths]
        axes[2].plot(lengths, [c['output_tps_p50'] if c['status'] == 'pass' else np.nan for c in selected],
                     label=mode.replace('_', ' '), color=color, marker=marker)
    axes[0].set(title='Pure prefill', ylabel='Request prefill wall (ms; log scale)', yscale='log')
    axes[1].set(title='Isolated decode: 128 outputs', ylabel='Cached decode output tokens/s', ylim=(0, None))
    axes[2].set(title='Mixed: active limit 4, cap 128', ylabel='Aggregate output tokens/s', ylim=(0, None))
    for ax in axes:
        ax.set_xscale('log', base=2)
        ax.set_xticks(lengths, [str(p) for p in lengths])
        ax.set_xlabel('Prompt tokens')
        ax.grid(axis='y', alpha=.2)
        ax.spines[['right', 'top']].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
        ax.tick_params(labelsize=8)
    fig.text(.02, .02, 'One L4 • 3 measured runs per case • Mixed traffic matched across policies • Failed chunk-256/512 point omitted • Separate panels use different units', fontsize=9, color='#444444')
    fig.tight_layout(rect=[0, .06, 1, .91])
    asset_dir = args.markdown.parent / 'assets'
    asset_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'svg'):
        fig.savefig(asset_dir / f'phase_sweep_review_20261005.{suffix}', dpi=160, facecolor='white')
    plt.close(fig)
    lines[7:7] = ['', '![Passing prefill, decode, and mixed-policy measurements](assets/phase_sweep_review_20261005.png)', '',
                  'Figure panels use separate units. The missing chunk-256 point at 512 tokens failed correctness; '
                  'the plotted mixed trace uses active limit 4 and cap 128. See the tables for exact values.', '']
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text('\n'.join(lines))
    print('Wrote ' + str(args.markdown))


if __name__ == '__main__':
    main()
