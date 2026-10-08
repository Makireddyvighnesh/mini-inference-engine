"""Audit saved phase-sweep outputs and derive compact comparison evidence.

This reads existing results only. It never changes reference tokens or turns
failed rows into accepted performance measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def quantile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.sweep_dir.resolve()
    manifest = json.loads((base / 'phase_sweep_manifest.json').read_text())
    references = json.loads((base / 'hf_references.json').read_text())
    rows = manifest['completed_cases']
    assert manifest['status'] == 'completed_with_failures_or_skips'
    assert {r['key'] for r in rows} == {r['key'] for r in manifest['planned_cases']}
    assert len(rows) == len({r['key'] for r in rows}) == 448
    for name, digest in manifest['provenance']['sha256'].items():
        assert hashlib.sha256((base / 'source_snapshot' / name).read_bytes()).hexdigest() == digest, name
    for name in ('prompts.json', 'hf_references.json'):
        assert hashlib.sha256((base / name).read_bytes()).hexdigest() == manifest['provenance']['sha256']['inputs/' + name]
    audit = Counter()
    mismatch_rows = []
    failures = []
    cases = []
    signatures = {}
    for index, case in enumerate(rows):
        compact = {k: case[k] for k in ('key', 'mode', 'prompt_tokens', 'generation_cap', 'batch_limit', 'status')}
        if case['status'] == 'skipped_memory_guard':
            cases.append(compact)
            continue
        data = json.loads(Path(case['result']).read_text())
        specs = {r['request_id']: r for r in data['workload']['requests']}
        assert len(data['runs']) == 3 and len(data['warmup_durations_ms']) == 1
        signature = [(s['prompt_sha256'], s['prompt_tokens'], s['max_new_tokens'], s['scheduled_arrival_ms']) for s in specs.values()]
        if case['mode'] not in ('prefill', 'isolated'):
            group = (case['prompt_tokens'], case['generation_cap'], case['batch_limit'])
            assert signatures.setdefault(group, signature) == signature
        prior = {}
        unstable = set()
        pooled = defaultdict(list)
        output_rates = []
        mismatches = []
        completion_errors = []
        phases = defaultdict(list)
        for run in data['runs']:
            assert not run['warmup']
            audit['measured_runs_checked'] += 1
            by_request = defaultdict(list)
            for event in run['events']:
                by_request[event['request_id']].append(event)
            generated = 0
            for request in run['requests']:
                audit['measured_outputs_checked'] += 1
                owner = request['request_id']
                spec, outcome = specs[owner], request['outcome']
                actual = outcome['generated_token_ids']
                expected = references[spec['prompt_sha256']]['generated_token_ids'][:spec['max_new_tokens']]
                if owner in prior and prior[owner] != actual:
                    unstable.add(owner)
                prior[owner] = actual
                if outcome['status'] != 'completed' or len(actual) != spec['max_new_tokens']:
                    completion_errors.append({'request': owner, 'status': outcome['status'], 'error': outcome.get('error')})
                else:
                    audit['completed_exact_length_outputs'] += 1
                if actual == expected:
                    audit['hf_matching_outputs'] += 1
                else:
                    mismatch = next(((i, a, e) for i, (a, e) in enumerate(zip(actual, expected)) if a != e), None)
                    row = {'case': case['key'], 'mode': case['mode'], 'batch': case['batch_limit'],
                           'selected_prompt': case['prompt_tokens'], 'generation_cap': case['generation_cap'],
                           'request': owner, 'category': spec['category'], 'request_prompt_tokens': spec['prompt_tokens'],
                           'repetition': run['repetition_index'], 'first_difference': mismatch,
                           'actual_length': len(actual), 'expected_length': len(expected)}
                    mismatches.append(row)
                    mismatch_rows.append(row)
                events = sorted(by_request[owner], key=lambda e: e['sequence'])
                times = {kind: next(e['timestamp_ns'] for e in events if e['event'] == kind)
                         for kind in ('arrival', 'admission', 'prefill_start', 'prefill_end', 'completion')}
                tokens = sorted((e for e in events if e['event'] == 'token_ready'), key=lambda e: e['token_index'])
                token_times = [e['timestamp_ns'] for e in tokens]
                assert [e['metadata']['token_id'] for e in tokens] == actual
                assert [e['token_index'] for e in tokens] == list(range(len(actual)))
                gaps = [(right - left) / 1e6 for left, right in zip(token_times, token_times[1:])]
                assert all(g >= 0 for g in gaps)
                values = {'ttft_ms': (token_times[0] - times['arrival']) / 1e6,
                          'e2e_latency_ms': (times['completion'] - times['arrival']) / 1e6,
                          'queue_delay_ms': (times['admission'] - times['arrival']) / 1e6,
                          'prefill_ms': (times['prefill_end'] - times['prefill_start']) / 1e6,
                          'decode_ms': (times['completion'] - token_times[0]) / 1e6}
                if len(token_times) > 1:
                    values['tpot_ms'] = (token_times[-1] - token_times[0]) / 1e6 / (len(token_times) - 1)
                for name, value in values.items():
                    assert math.isclose(value, request['metrics'][name], abs_tol=1e-6, rel_tol=1e-10), (case['key'], name)
                    pooled[name].append(value)
                assert gaps == request['metrics']['itl_ms']
                pooled['itl_ms'].extend(gaps)
                generated += len(actual)
            output_rates.append(generated / (run['duration_ms'] / 1000))
            all_times = [e['timestamp_ns'] for e in run['events'] if e['event'] == 'token_ready']
            window = (max(all_times) - min(all_times)) / 1e6
            phases['decode_window_wall_ms_p50'].append(window)
            decode_count = sum(max(0, len(r['outcome']['generated_token_ids']) - 1) for r in run['requests'])
            phases['decode_window_tokens_per_second_p50'].append(decode_count / (window / 1000) if window else None)
            phases['prefill_request_wall_ms_p50'].append(quantile([r['metrics']['prefill_ms'] for r in run['requests']], .5))
            forward = case['forward_runs'][run['repetition_index']]
            for phase in ('prefill', 'decode'):
                elapsed = sum(c['cuda_elapsed_ms'] for c in forward['calls'] if c['phase'] == phase) or None
                saved = forward[phase]['cuda_elapsed_total_ms']
                assert (elapsed is None and saved is None) or math.isclose(elapsed, saved, abs_tol=1e-6, rel_tol=1e-10)
                phases[phase + '_forward_cuda_total_ms_p50'].append(elapsed)
                if phase == 'prefill':
                    phases['prefill_forward_input_tokens_per_second_p50'].append(sum(s['prompt_tokens'] for s in specs.values()) / (elapsed / 1000) if elapsed else None)
        for metric, values in pooled.items():
            if not values:
                continue
            for label, q in (('p50', .5), ('p95', .95), ('p99', .99)):
                assert math.isclose(quantile(values, q), data['summary']['metrics'][metric][label], abs_tol=1e-6, rel_tol=1e-10), (case['key'], metric, label)
        for label, q in (('p50', .5), ('p95', .95), ('p99', .99)):
            assert math.isclose(quantile(output_rates, q), data['summary']['tokens_per_second'][label], abs_tol=1e-8, rel_tol=1e-10), case['key']
        for name, values in phases.items():
            available = [v for v in values if v is not None]
            expected = quantile(available, .5) if available else None
            actual = case['phase_summary'][name]
            assert (expected is None and actual is None) or math.isclose(expected, actual, abs_tol=1e-6, rel_tol=1e-10), (case['key'], name)
        audit['phase_summary_cases_checked'] += 1
        expected_status = 'fail' if mismatches or completion_errors or unstable else 'pass'
        assert case['status'] == expected_status, (case['key'], case['status'], expected_status)
        audit['executed_cases_checked'] += 1
        audit['stable_cases'] += not unstable
        if mismatches or completion_errors or unstable:
            failures.append({**compact, 'mismatch_outputs': len(mismatches), 'completion_errors': completion_errors,
                             'unstable_requests': sorted(unstable),
                             'first_differences': dict(Counter(str(m['first_difference']) for m in mismatches)),
                             'mismatch_categories': dict(Counter(m['category'] for m in mismatches))})
        summary = data['summary']
        compact['metrics'] = {metric: {q: summary['metrics'].get(metric, {}).get(q) for q in ('p50', 'p95', 'p99')}
                              for metric in ('ttft_ms', 'itl_ms', 'e2e_latency_ms', 'tpot_ms')}
        compact['output_tps_p50'] = summary['tokens_per_second']['p50']
        compact['repeat_output_tps'] = [r['summary']['tokens_per_second'] for r in data['runs']]
        compact['phase_summary'] = case['phase_summary']
        compact['gpu_p50'] = summary['gpu_utilization_percent'].get('p50')
        compact['peak_allocated_gib'] = summary['memory'].get('peak_allocated_bytes', 0) / 2**30
        compact['run_dates'] = [r['started_at_utc'] for r in data['runs']]
        compact['decode_host_to_cuda_ratios'] = [
            sum(c['host_enqueue_ms'] for c in f['calls'] if c['phase'] == 'decode') / f['decode']['cuda_elapsed_total_ms']
            for f in case['forward_runs'] if f['decode']['cuda_elapsed_total_ms']]
        cases.append(compact)
        if (index + 1) % 50 == 0:
            print(f'Audited {index + 1}/{len(rows)} cases', flush=True)
    result = {'scope': 'Saved 2026-10-03 sweep completed 2026-10-05; native isolation and matched paged mixed policies.',
              'validation': dict(audit), 'counts': dict(Counter(c['status'] for c in rows)),
              'snapshot_hashes_checked': len(manifest['provenance']['sha256']),
              'first_mismatch_patterns': dict(Counter(str(m['first_difference']) for m in mismatch_rows)),
              'mismatch_category_counts': dict(Counter(m['category'] for m in mismatch_rows)),
              'cases': cases, 'failed_cases': failures, 'mismatched_outputs': mismatch_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('cases', 'failed_cases', 'mismatched_outputs')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
