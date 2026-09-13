from __future__ import annotations

from copy import deepcopy

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from minillm_l4.benchmarks.core.harness import BenchmarkHarness
from minillm_l4.benchmarks.core.schemas import HarnessConfig, RequestSpec, WorkloadSpec
from minillm_l4.benchmarks.runners.paged_kv import (
    PagedAttentionBatchRunner,
    PagedHybridBatchRunner,
)
from minillm_l4.engine.generation.manual import manual_greedy_generate


def tiny_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        use_sliding_window=False,
        sliding_window=None,
    )


def test_direct_paged_qwen_attention_matches_trusted_manual_generation() -> None:
    torch.manual_seed(23)
    reference_model = Qwen3ForCausalLM(tiny_config()).eval()
    direct_model = deepcopy(reference_model).eval()
    requests = (
        RequestSpec("a", (1, 2, 3), 3),
        RequestSpec("b", (4, 5, 6), 3),
    )
    workload = WorkloadSpec(
        name="direct-paged-qwen",
        seed=23,
        requests=requests,
        device="cpu",
    )
    runner = PagedAttentionBatchRunner(
        direct_model,
        block_size=2,
        num_blocks=8,
        device="cpu",
    )
    result = BenchmarkHarness(
        HarnessConfig(
            warmup_repetitions=0,
            repetitions=1,
            collect_gpu=False,
            collect_system_telemetry=False,
        )
    ).run_batched(workload, batch_size=2, runner=runner)

    expected: list[tuple[int, ...]] = []
    for request in requests:
        output = manual_greedy_generate(
            reference_model,
            {
                "input_ids": torch.tensor([request.prompt_token_ids]),
                "attention_mask": torch.ones(
                    (1, request.prompt_tokens),
                    dtype=torch.long,
                ),
            },
            output_tokens=request.max_new_tokens,
        )
        expected.append(tuple(int(value) for value in output.row(0).tolist()))

    actual = tuple(
        tuple(record["outcome"]["generated_token_ids"])
        for record in result.runs[0]["requests"]
    )
    assert actual == tuple(expected)
    assert runner.last_cache_snapshot is not None
    assert runner.last_cache_snapshot["gather_path"] is False
    assert runner.last_cache_snapshot["direct_paged_attention"] is True
    assert (
        runner.last_cache_snapshot["attention_backend"]
        == "paged_reference_prefill_torch_blockwise_reference"
    )
    assert runner.last_cache_snapshot["resources_released"] is True


def test_hybrid_paged_qwen_attention_matches_trusted_manual_generation() -> None:
    torch.manual_seed(29)
    reference_model = Qwen3ForCausalLM(tiny_config()).eval()
    hybrid_model = deepcopy(reference_model).eval()
    requests = (
        RequestSpec("a", (1, 2, 3), 3),
        RequestSpec("b", (4, 5, 6), 3),
    )
    workload = WorkloadSpec(
        name="hybrid-paged-qwen",
        seed=29,
        requests=requests,
        device="cpu",
    )
    runner = PagedHybridBatchRunner(
        hybrid_model,
        block_size=2,
        num_blocks=8,
        device="cpu",
    )
    result = BenchmarkHarness(
        HarnessConfig(
            warmup_repetitions=0,
            repetitions=1,
            collect_gpu=False,
            collect_system_telemetry=False,
        )
    ).run_batched(workload, batch_size=2, runner=runner)

    expected = []
    for request in requests:
        output = manual_greedy_generate(
            reference_model,
            {
                "input_ids": torch.tensor([request.prompt_token_ids]),
                "attention_mask": torch.ones(
                    (1, request.prompt_tokens),
                    dtype=torch.long,
                ),
            },
            output_tokens=request.max_new_tokens,
        )
        expected.append(tuple(int(value) for value in output.row(0).tolist()))

    actual = tuple(
        tuple(record["outcome"]["generated_token_ids"])
        for record in result.runs[0]["requests"]
    )
    assert actual == tuple(expected)
    assert runner.last_cache_snapshot is not None
    snapshot = runner.last_cache_snapshot
    assert snapshot["gather_path"] is False
    assert snapshot["direct_paged_attention"] is True
    assert snapshot["prefill_backend"] == "sdpa"
    assert snapshot["decode_backend"] == "torch_blockwise_reference"
    assert snapshot["page_visits"]["prefill_all_layers"] == 0
    assert snapshot["page_visits"]["decode_all_layers"] == 20
    assert snapshot["resources_released"] is True
