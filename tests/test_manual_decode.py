from types import SimpleNamespace

import torch
import pytest

from minillm_l4.benchmarks.core.harness import RequestEventRecorder
from minillm_l4.benchmarks.core.schemas import RequestSpec
from minillm_l4.benchmarks.runners.manual_decode import ManualGreedyBatchRunner
from minillm_l4.engine.generation.manual import manual_greedy_generate


class FakeForwardModel:
    """A forward-only model whose next token is determined by call number."""

    def __init__(self, *, eos_on_call: int | None = None) -> None:
        self.calls: list[dict] = []
        self.eos_on_call = eos_on_call
        self.config = SimpleNamespace(_name_or_path="fake-manual")

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        batch_size = kwargs["input_ids"].shape[0]
        call_index = len(self.calls) - 1
        token_id = 2 if call_index == self.eos_on_call else 10 + call_index
        logits = torch.zeros((batch_size, 1, 32), dtype=torch.float32)
        logits[:, :, token_id] = 1.0
        return SimpleNamespace(
            logits=logits,
            past_key_values=(call_index,),
        )

    def generate(self, **kwargs):
        del kwargs
        raise AssertionError("manual generation must not call model.generate")


def sample_inputs() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[3, 4, 5], [6, 7, 8]], dtype=torch.long),
        "attention_mask": torch.ones((2, 3), dtype=torch.long),
    }


def test_manual_generation_uses_prefill_then_cached_decode_steps() -> None:
    model = FakeForwardModel()
    prefill_tokens: list[list[int]] = []
    streamed_tokens: list[tuple[int, list[int]]] = []

    result = manual_greedy_generate(
        model,
        sample_inputs(),
        output_tokens=4,
        on_prefill_end=lambda token_ids: prefill_tokens.append(
            [int(value) for value in token_ids[:, 0].tolist()]
        ),
        on_token=lambda index, token_ids: streamed_tokens.append(
            (index, [int(value) for value in token_ids[:, 0].tolist()])
        ),
    )

    assert result.runtime == "manual_eager"
    assert result.token_ids.tolist() == [
        [10, 11, 12, 13],
        [10, 11, 12, 13],
    ]
    assert result.sequence_lengths == (4, 4)
    assert result.aggregate_output_tokens == 8
    assert prefill_tokens == [[10, 10]]
    assert streamed_tokens == [
        (0, [10, 10]),
        (1, [11, 11]),
        (2, [12, 12]),
        (3, [13, 13]),
    ]
    assert len(model.calls) == 4
    assert model.calls[0]["input_ids"].shape == (2, 3)
    assert [call["input_ids"].shape for call in model.calls[1:]] == [
        (2, 1),
        (2, 1),
        (2, 1),
    ]
    assert [call["attention_mask"].shape[1] for call in model.calls] == [3, 4, 5, 6]
    assert "past_key_values" not in model.calls[0]
    assert all("past_key_values" in call for call in model.calls[1:])
    assert all(call["use_cache"] is True for call in model.calls)
    assert all(call["return_dict"] is True for call in model.calls)
    assert all(call["logits_to_keep"] == 1 for call in model.calls)


def test_manual_generation_supports_eos_and_tracks_true_lengths() -> None:
    model = FakeForwardModel(eos_on_call=1)

    result = manual_greedy_generate(
        model,
        {
            "input_ids": torch.tensor([[3, 4]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        },
        output_tokens=5,
        eos_token_id=2,
        pad_token_id=0,
    )

    assert result.token_ids.tolist() == [[10, 2, 0, 0, 0]]
    assert result.sequence_lengths == (2,)
    assert result.row(0).tolist() == [10, 2]
    assert result.aggregate_output_tokens == 2
    assert len(model.calls) == 2


def test_manual_generation_supports_per_row_limits_and_position_ids() -> None:
    model = FakeForwardModel()
    inputs = sample_inputs()
    inputs["position_ids"] = torch.tensor(
        [[0, 1, 2], [0, 1, 2]], dtype=torch.long
    )

    result = manual_greedy_generate(
        model,
        inputs,
        output_tokens=4,
        sequence_output_limits=(2, 4),
    )

    assert result.sequence_lengths == (2, 4)
    assert result.row(0).tolist() == [10, 11]
    assert result.row(1).tolist() == [10, 11, 12, 13]
    assert [call["position_ids"][:, 0].tolist() for call in model.calls[1:]] == [
        [3, 3],
        [4, 4],
        [5, 5],
    ]


def test_manual_runner_records_token_events_and_returns_request_outcomes() -> None:
    model = FakeForwardModel()
    runner = ManualGreedyBatchRunner(model, device="cpu")
    requests = tuple(
        RequestSpec(
            request_id=f"request-{index}",
            prompt_token_ids=(1, 2, 3),
            max_new_tokens=3,
        )
        for index in range(2)
    )
    recorders = [
        RequestEventRecorder(request.request_id, run_started_ns=0)
        for request in requests
    ]

    outcomes = runner(requests, recorders)

    assert [outcome.generated_token_ids for outcome in outcomes] == [
        (10, 11, 12),
        (10, 11, 12),
    ]
    assert all(outcome.metadata["runner"] == "manual_eager" for outcome in outcomes)
    for recorder in recorders:
        events = [event.event for event in recorder.events]
        assert events.count("prefill_start") == 1
        assert events.count("prefill_end") == 1
        assert events.count("token_ready") == 3
        assert events.count("token_sent") == 3
        assert events.count("completion") == 1


def test_manual_static_runner_rejects_mixed_prompt_lengths() -> None:
    runner = ManualGreedyBatchRunner(FakeForwardModel(), device="cpu")
    requests = (
        RequestSpec(request_id="first", prompt_token_ids=(1, 2), max_new_tokens=2),
        RequestSpec(
            request_id="second",
            prompt_token_ids=(1, 2, 3),
            max_new_tokens=2,
        ),
    )
    recorders = [
        RequestEventRecorder(request.request_id, run_started_ns=0)
        for request in requests
    ]

    with pytest.raises(ValueError, match="equal prompt lengths"):
        runner(requests, recorders)
