"""Opt-in pinned-model regression: MINILLM_RUN_MODEL_TESTS=1 pytest ..."""

from __future__ import annotations

import math
import os
from dataclasses import replace
from pathlib import Path

import pytest

from minillm_l4.benchmarks.core.harness import BenchmarkHarness
from minillm_l4.benchmarks.core.schemas import HarnessConfig
from minillm_l4.benchmarks.runners.huggingface_baseline import build_hf_workload, load_qwen_fp8
from minillm_l4.benchmarks.runners.kv_cache import KvCacheBatchRunner
from minillm_l4.benchmarks.runners.paged_kv import PagedHybridBatchRunner
from minillm_l4.benchmarks.runners.paged_cuda_graph import PagedCudaGraphBatchRunner


pytestmark = pytest.mark.skipif(
    os.environ.get("MINILLM_RUN_MODEL_TESTS") != "1",
    reason="opt-in test requires the pinned FP8 model and an L4 GPU",
)


def _harness():
    return BenchmarkHarness(HarnessConfig(
        warmup_repetitions=0, repetitions=1, collect_gpu=False,
        collect_system_telemetry=False,
    ))


def _tokens(result):
    return {
        record["request_id"]: tuple(record["outcome"]["generated_token_ids"])
        for record in result.runs[0]["requests"]
    }


@pytest.fixture(scope="module")
def real_model():
    bundle = load_qwen_fp8(fp8_kernel_path="sm89")
    workload = build_hf_workload(
        bundle.tokenizer, Path(__file__).resolve().parents[1] / "data/synthetic/workloads_v1.jsonl",
        bucket_name="short", prompt_tokens=128, output_tokens=32, count=4, seed=17,
    )
    reference = _harness().run_batched(
        workload, batch_size=1,
        runner=KvCacheBatchRunner(bundle.model, mode="contiguous", device="cuda:0"),
    )
    expected = _tokens(reference)
    # Anchor the original failure; never substitute paged tokens as a reference.
    assert expected["baseline-short-003"][16] == 1378  # " two"
    return bundle.model, workload, expected


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("block_size", [8, 16, 32, 64])
def test_hybrid_real_model_matches_unchanged_dense_reference(real_model, batch_size, block_size):
    model, workload, expected = real_model
    runner = PagedHybridBatchRunner(
        model, device="cuda:0", block_size=block_size,
        num_blocks=4 * math.ceil(160 / block_size),
    )
    actual = _harness().run_batched(workload, batch_size=batch_size, runner=runner)
    assert _tokens(actual) == expected
    assert runner.last_cache_snapshot["resources_released"] is True


@pytest.mark.parametrize("batch_size", [1, 4])
def test_dense_prefill_graph_matches_reference_including_original_failure(real_model, batch_size):
    model, workload, expected = real_model
    # Batch one must include request 003, not just the easier first request.
    requests = (workload.requests[3],) if batch_size == 1 else workload.requests
    selected = replace(workload, requests=requests)
    runner = PagedCudaGraphBatchRunner(
        model, device="cuda:0", block_size=16,
        num_blocks=4 * math.ceil(160 / 16), prefill_backend="dense",
    )
    result = BenchmarkHarness(HarnessConfig(
        warmup_repetitions=1, repetitions=1, collect_gpu=False,
        collect_system_telemetry=False,
    )).run_batched(selected, batch_size=batch_size, runner=runner)
    assert _tokens(result) == {request.request_id: expected[request.request_id] for request in requests}
    assert runner.last_cache_snapshot["decode_sdpa_compat"] is True
