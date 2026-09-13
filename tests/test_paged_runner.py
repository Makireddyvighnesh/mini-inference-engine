from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

from minillm_l4.benchmarks.core.harness import BenchmarkHarness
from minillm_l4.benchmarks.core.schemas import HarnessConfig, RequestSpec, WorkloadSpec
from minillm_l4.benchmarks.runners.paged_kv import PagedKvBatchRunner
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


class PagedFakeModel:
    def __init__(self) -> None:
        self.config = tiny_config()
        self.config._name_or_path = "paged-fake-model"

    def __call__(self, **kwargs):
        input_ids = kwargs["input_ids"]
        batch_size, input_tokens = input_ids.shape
        cache = kwargs.get("past_key_values")
        if cache is None:
            cache = DynamicCache(config=self.config)
        for layer_index in range(self.config.num_hidden_layers):
            keys = torch.full(
                (
                    batch_size,
                    self.config.num_key_value_heads,
                    input_tokens,
                    self.config.head_dim,
                ),
                float(layer_index + 1),
                dtype=torch.float32,
            )
            cache.update(keys, keys + 0.5, layer_index)

        next_ids = (input_ids[:, -1] + 1) % self.config.vocab_size
        logits = torch.zeros(
            (batch_size, 1, self.config.vocab_size),
            dtype=torch.float32,
        )
        logits.scatter_(2, next_ids[:, None, None], 1.0)
        return SimpleNamespace(logits=logits, past_key_values=cache)


def test_paged_runner_matches_independent_manual_generation() -> None:
    requests = (
        RequestSpec("a", (1, 2, 3), 3),
        RequestSpec("b", (4, 5, 6), 3),
    )
    workload = WorkloadSpec(
        name="paged-runner",
        seed=17,
        requests=requests,
        device="cpu",
    )
    runner = PagedKvBatchRunner(
        PagedFakeModel(),
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

    actual = [
        record["outcome"]["generated_token_ids"]
        for record in result.runs[0]["requests"]
    ]
    expected = []
    for request in requests:
        reference = manual_greedy_generate(
            PagedFakeModel(),
            {
                "input_ids": torch.tensor([request.prompt_token_ids]),
                "attention_mask": torch.ones((1, request.prompt_tokens), dtype=torch.long),
            },
            output_tokens=request.max_new_tokens,
        )
        expected.append([int(value) for value in reference.row(0).tolist()])
    assert actual == expected

    assert runner.last_cache_snapshot is not None
    assert runner.last_cache_snapshot["gather_path"] is True
    assert runner.last_cache_snapshot["paged_attention_kernel"] is False
    assert runner.last_cache_snapshot["used_token_slots"] == 10
    assert runner.last_cache_snapshot["reserved_token_slots"] == 12
    assert runner.last_cache_snapshot["wasted_token_slots"] == 2
    assert runner.last_cache_snapshot["active_sequence_count"] == 2
    assert runner.last_cache_snapshot["resources_released"] is True
    assert runner.last_cache_snapshot["active_sequence_count_after_release"] == 0


def test_paged_runner_matches_tiny_real_qwen_model() -> None:
    model = Qwen3ForCausalLM(tiny_config()).eval()
    requests = (
        RequestSpec("a", (1, 2, 3), 3),
        RequestSpec("b", (4, 5, 6), 3),
    )
    workload = WorkloadSpec(
        name="tiny-qwen-paged",
        seed=17,
        requests=requests,
        device="cpu",
    )
    runner = PagedKvBatchRunner(
        model,
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
        reference = manual_greedy_generate(
            model,
            {
                "input_ids": torch.tensor([request.prompt_token_ids]),
                "attention_mask": torch.ones((1, request.prompt_tokens), dtype=torch.long),
            },
            output_tokens=request.max_new_tokens,
        )
        expected.append(tuple(int(value) for value in reference.row(0).tolist()))
    actual = tuple(
        tuple(record["outcome"]["generated_token_ids"])
        for record in result.runs[0]["requests"]
    )
    assert actual == tuple(expected)
