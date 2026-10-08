"""Phase 8 gate without Hugging Face: every scheduling policy emits the same tokens.

Runs one staggered-arrival trace (prompts 128..8192 plus odd lengths that leave
tiny final chunks) through whole-prompt prefill and each chunked / mixed /
adaptive policy, with real decode traffic, and compares every generated token
against the whole-prompt run.  Exit code 1 on any mismatch.

  .conda-env/bin/python minillm_l4/scripts/check_policy_equivalence.py
"""

from __future__ import annotations

import gc
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from minillm_l4.benchmarks.core.harness import RequestEventRecorder  # noqa: E402
from minillm_l4.benchmarks.core.schemas import RequestSpec  # noqa: E402
from minillm_l4.benchmarks.core.synthetic import SyntheticSample, exact_token_ids  # noqa: E402
from minillm_l4.benchmarks.runners.chunked_prefill import ChunkedPrefillPagedRunner  # noqa: E402
from minillm_l4.benchmarks.runners.huggingface_baseline import load_qwen_fp8  # noqa: E402

LENGTHS = (128, 8192, 600, 2048, 4097, 1000, 3000, 256)
OUTPUT_TOKENS = 64
POLICIES = {
    "whole": dict(prefill_chunk_size=None),
    "chunked_256": dict(prefill_chunk_size=256, max_prefill_tokens=256),
    "mixed_512": dict(prefill_chunk_size=None, mixed_batch=True, max_prefill_tokens=512),
    "mixed_2048": dict(prefill_chunk_size=None, mixed_batch=True, max_prefill_tokens=2048),
    "adaptive": dict(prefill_chunk_size=None, mixed_batch=True, adaptive_chunking=True),
    "whole_graph": dict(prefill_chunk_size=None, cuda_graphs=True),
    "mixed_512_graph": dict(prefill_chunk_size=None, mixed_batch=True, max_prefill_tokens=512, cuda_graphs=True),
    "adaptive_graph": dict(prefill_chunk_size=None, mixed_batch=True, adaptive_chunking=True, cuda_graphs=True),
    "adaptive_graph_split": dict(prefill_chunk_size=None, mixed_batch=True, adaptive_chunking=True,
                                 cuda_graphs=True, graph_mixed_decode=True),
    "adaptive_graph_split_prefill_first": dict(prefill_chunk_size=None, mixed_batch=True, adaptive_chunking=True,
                                             cuda_graphs=True, graph_mixed_decode=True,
                                             graph_mixed_decode_order="prefill_first"),
}


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("policy equivalence gate requires CUDA; CPU graph fallback is not graph validation")
    bundle = load_qwen_fp8(fp8_kernel_path="sm89", local_files_only=True)
    requests = tuple(
        RequestSpec(f"r{i}", tuple(exact_token_ids(bundle.tokenizer, SyntheticSample(
            sample_id=f"equivalence-{n}", category="gate", target_prompt_tokens=n,
            target_output_tokens=OUTPUT_TOKENS,
            seed_text="Explain how prefill, cached decoding, and batching affect inference."))),
            OUTPUT_TOKENS, scheduled_arrival_ms=150.0 * i)
        for i, n in enumerate(LENGTHS))
    blocks = sum(math.ceil((r.prompt_tokens + OUTPUT_TOKENS) / 16) for r in requests)
    outputs = {}
    for name, options in POLICIES.items():
        runner = ChunkedPrefillPagedRunner(
            bundle.model, block_size=16, num_blocks=blocks, max_batch_size=len(requests),
            device="cuda", decode_backend="auto", decode_sdpa_compat=True, enable_prefix=False,
            **{"max_prefill_tokens": 256, **options})
        start = time.perf_counter_ns()
        recorders = tuple(RequestEventRecorder(r.request_id, run_started_ns=start) for r in requests)
        try:
            result = runner(requests, recorders)
            if any(outcome.status != "completed" or len(outcome.generated_token_ids) != OUTPUT_TOKENS
                   for outcome in result):
                raise RuntimeError(f"{name}: incomplete request outputs")
            summary = runner.last_summary
            if options.get("cuda_graphs") and not summary["graph_replays"]:
                raise RuntimeError(f"{name}: no graph replays: {summary['graph_fallback_reason']}")
            chunks = [r["computed_tokens"] for r in runner.last_summary["prefill_records"]]
        finally:
            runner.close()
            del runner
            gc.collect()
            torch.cuda.empty_cache()
        outputs[name] = [list(outcome.generated_token_ids) for outcome in result]
        print(f"{name:34s} {(time.perf_counter_ns() - start) / 1e9:6.1f} s, {len(chunks)} prefill chunks, "
              f"smallest {min(chunks)} tokens, mode={summary['decode_mode_used']}, "
              f"replays={summary['graph_replays']}, eager={summary['eager_decode_steps']}, "
              f"buckets={summary['captured_buckets']}", flush=True)
    failures = 0
    for name, rows in outputs.items():
        if name == "whole":
            continue
        for request, row, reference in zip(requests, rows, outputs["whole"], strict=True):
            if row != reference:
                failures += 1
                first = next((i for i, (a, b) in enumerate(zip(row, reference)) if a != b), min(len(row), len(reference)))
                print(f"MISMATCH {name} {request.request_id} ({request.prompt_tokens} tokens) at output {first}")
    total = len(requests) * (len(POLICIES) - 1)
    print(f"{total - failures}/{total} request outputs identical to whole-prompt prefill "
          f"({len(requests)} requests x {len(POLICIES) - 1} policies, {OUTPUT_TOKENS} tokens each)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
