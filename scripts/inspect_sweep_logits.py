"""Replay a few sweep divergences; logit capture is not a timing benchmark."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import torch
import minillm_l4

from minillm_l4.benchmarks.commands.run_prefill_decode_sweep import gpu_occupants
from minillm_l4.benchmarks.core.harness import RequestEventRecorder
from minillm_l4.benchmarks.core.schemas import RequestSpec
from minillm_l4.benchmarks.runners.chunked_prefill import ChunkedPrefillPagedRunner
from minillm_l4.benchmarks.runners.huggingface_baseline import load_qwen_fp8
from minillm_l4.engine.generation.huggingface import transformers_greedy_generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    runtime_dir = Path(minillm_l4.__file__).resolve().parent
    assert runtime_dir == (args.sweep_dir / 'source_snapshot/minillm_l4').resolve(), runtime_dir
    assert not gpu_occupants(), 'GPU must be exclusive for diagnostic replay'
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    prompts = json.loads((args.sweep_dir / 'prompts.json').read_text())
    references = json.loads((args.sweep_dir / 'hf_references.json').read_text())
    bundle = load_qwen_fp8(fp8_kernel_path='sm89', local_files_only=True)
    model = bundle.model
    report = {'purpose': 'Numerical diagnostic only; hooks and synchronization invalidate performance timing.',
              'model': bundle.metadata(), 'source_sweep': str(args.sweep_dir),
              'runtime_source_dir': str(runtime_dir), 'probes': []}
    shutil.copyfile(__file__, args.output_dir / 'inspect_sweep_logits.py')
    report['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for filename in ('prompts.json', 'hf_references.json'):
        report[filename + '_sha256'] = hashlib.sha256((args.sweep_dir / filename).read_bytes()).hexdigest()
    # Native HF batched generation is an independent control for shape effects.
    probes = [('hf_p128_b1', 128, 366, 1, 'hf', None),
              ('hf_p128_b2', 128, 366, 2, 'hf', None),
              ('hf_p512_b1', 512, 128, 1, 'hf', None),
              ('hf_p512_b2', 512, 128, 2, 'hf', None),
              ('paged_p128_whole', 128, 366, 1, 'paged', None),
              ('paged_p512_whole', 512, 107, 1, 'paged', None),
              ('paged_p512_chunk128', 512, 107, 1, 'paged', 128),
              ('paged_p512_chunk256', 512, 107, 1, 'paged', 256)]
    vectors = {}
    for name, p, g, batch, mode, chunk in probes:
        print('Probe ' + name + ' starting', flush=True)
        request = RequestSpec(name, tuple(prompts[str(p)]), g)
        expected = references[request.prompt_sha256]['generated_token_ids'][:g]
        expected_ids = sorted(set(expected))
        expected_tensor = torch.tensor(expected_ids, device='cuda', dtype=torch.long)
        captured = []
        own_vectors = {}
        prefill_calls = math.ceil(p / chunk) if chunk else 1

        def capture(module, inputs, kwargs, output):
            call = len(captured)
            output_index = call - (prefill_calls - 1)
            logits = output.logits[:, -1, :]
            top = logits[0].float().topk(8)
            values = top.values.cpu().tolist()
            ids = top.indices.cpu().tolist()
            expected_values = logits[0].index_select(0, expected_tensor).float().cpu().tolist()
            captured.append({'output_index': output_index if output_index >= 0 else None,
                             'input_shape': list(kwargs['input_ids'].shape),
                             'logits_dtype': str(logits.dtype), 'winner': int(logits[0].argmax().item()),
                             'top8_ids': ids, 'top8_values': values, 'top2_gap': values[0] - values[1],
                             'reference_values': dict(zip(expected_ids, expected_values, strict=True))})
            if output_index in (106, 365):
                own_vectors[output_index] = logits[0].detach().cpu()

        handle = model.register_forward_hook(capture, with_kwargs=True)
        backend = None
        try:
            if mode == 'hf':
                inputs = torch.tensor([request.prompt_token_ids] * batch, device='cuda', dtype=torch.long)
                generation = transformers_greedy_generate(model, {'input_ids': inputs, 'attention_mask': torch.ones_like(inputs)}, output_tokens=g)
                outputs = generation.token_ids.cpu().tolist()
            else:
                backend = ChunkedPrefillPagedRunner(model, block_size=16, num_blocks=math.ceil((p + g - 1) / 16),
                    max_batch_size=1, max_prefill_tokens=256, device='cuda', enable_prefix=False,
                    decode_sdpa_compat=True, prefill_chunk_size=chunk)
                recorder = RequestEventRecorder(name, run_started_ns=time.perf_counter_ns())
                outcome = backend((request,), (recorder,))[0]
                assert outcome.status == 'completed' and outcome.metadata['resources_released']
                outputs = [list(outcome.generated_token_ids)]
        finally:
            handle.remove()
            if backend:
                backend.close()
                del backend
            gc.collect()
            torch.cuda.empty_cache()
        emitted = [c for c in captured if c['output_index'] is not None]
        assert len(emitted) == g and len(outputs[0]) == g
        assert [c['winner'] for c in emitted] == outputs[0]
        difference = next((i for i, (a, e) in enumerate(zip(outputs[0], expected)) if a != e), None)
        diagnosis = None
        if difference is not None:
            c = emitted[difference]
            wanted = expected[difference]
            diagnosis = {k: c[k] for k in ('input_shape', 'logits_dtype', 'winner', 'top8_ids', 'top8_values', 'top2_gap')}
            diagnosis.update(output_index=difference, reference_token=wanted,
                             reference_token_value=c['reference_values'][wanted],
                             prefix_before_difference_matches=True)
        vectors[name] = own_vectors
        report['probes'].append({'name': name, 'prompt': p, 'output_cap': g, 'batch': batch, 'mode': mode, 'chunk': chunk,
                                 'matches_single_request_saved_hf': outputs[0] == expected,
                                 'all_batch_rows_identical': all(row == outputs[0] for row in outputs),
                                 'first_difference': diagnosis, 'generated_token_ids': outputs,
                                 'selected_logits': [{k: v for k, v in c.items() if k != 'reference_values'}
                                                     for c in emitted if c['output_index'] in (106, 365)]})
        (args.output_dir / 'diagnostic.json').write_text(json.dumps(report, indent=2) + '\n')
        print(name + ': ' + json.dumps(diagnosis) if diagnosis else name + ': exact saved-HF match', flush=True)
    comparisons = []
    for actual, reference, index in [('hf_p128_b2', 'hf_p128_b1', 365), ('paged_p128_whole', 'hf_p128_b1', 365),
                                     ('paged_p512_chunk256', 'hf_p512_b1', 106), ('paged_p512_chunk128', 'hf_p512_b1', 106)]:
        delta = (vectors[actual][index].float() - vectors[reference][index].float()).abs()
        comparisons.append({'actual': actual, 'reference': reference, 'output_index': index,
                            'max_absolute_logit_difference': float(delta.max()), 'mean_absolute_logit_difference': float(delta.mean())})
    report['selected_vector_comparisons'] = comparisons
    report['gpu_exclusive_at_end'] = not gpu_occupants()
    torch.save(vectors, args.output_dir / 'selected_logits.pt')
    (args.output_dir / 'diagnostic.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Diagnostic complete: ' + str(args.output_dir), flush=True)


if __name__ == '__main__':
    main()
