from __future__ import annotations

from copy import deepcopy

import torch
import pytest
from transformers import Qwen3Config, Qwen3ForCausalLM

from minillm_l4.benchmarks.core.harness import BenchmarkHarness
from minillm_l4.benchmarks.core.schemas import HarnessConfig, RequestSpec, WorkloadSpec
from minillm_l4.benchmarks.runners.packed_paged import (
    PackedPagedPrefillBatchRunner,
)
from minillm_l4.engine.generation.manual import manual_greedy_generate
from minillm_l4.engine.kv_cache import PackedSequenceMetadata


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


def test_packed_metadata_has_request_boundaries_without_padding() -> None:
    metadata = PackedSequenceMetadata.from_lengths(
        ("a", "b", "c"),
        (3, 5, 2),
    )
    assert metadata.total_tokens == 10
    assert metadata.cu_seqlens.tolist() == [0, 3, 8, 10]
    assert metadata.token_to_sequence.tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 2, 2]
    assert metadata.token_positions.tolist() == [0, 1, 2, 0, 1, 2, 3, 4, 0, 1]
    assert metadata.last_token_indices.tolist() == [2, 7, 9]
    assert metadata.to_dict()["padding_tokens_avoided"] == 5


@pytest.mark.parametrize("prefill_backend", ("torch", "sdpa"))
def test_packed_paged_prefill_matches_independent_greedy_generation(
    prefill_backend: str,
) -> None:
    torch.manual_seed(47)
    reference_model = Qwen3ForCausalLM(tiny_config()).eval()
    packed_model = deepcopy(reference_model).eval()
    requests = (
        RequestSpec("short", (1, 2, 3), 3),
        RequestSpec("long", (4, 5, 6, 7, 8), 3),
        RequestSpec("medium", (9, 10), 3),
    )
    workload = WorkloadSpec(
        name="packed-prefill",
        seed=47,
        requests=requests,
        device="cpu",
    )
    runner = PackedPagedPrefillBatchRunner(
        packed_model,
        block_size=2,
        num_blocks=16,
        device="cpu",
        prefill_backend=prefill_backend,
        decode_backend="torch",
    )
    result = BenchmarkHarness(
        HarnessConfig(
            warmup_repetitions=0,
            repetitions=1,
            collect_gpu=False,
            collect_system_telemetry=False,
        )
    ).run_batched(workload, batch_size=3, runner=runner)

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
    snapshot = runner.last_cache_snapshot
    assert snapshot["packed_prefill"] is True
    assert snapshot["prefill_prompt_lengths"] == [3, 5, 2]
    assert snapshot["prefill_input_tokens"] == 10
    assert snapshot["prefill_padded_equivalent_tokens"] == 15
    assert snapshot["prefill_padding_tokens"] == 5
    assert snapshot["prefill_metadata"]["cu_seqlens"] == [0, 3, 8, 10]
    assert snapshot["resources_released"] is True
