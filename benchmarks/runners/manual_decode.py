"""Benchmark runner for the explicit manual generation backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..core.harness import BenchmarkResult, RequestEventRecorder
from ..core.schemas import RequestOutcome, RequestSpec
from minillm_l4.engine.generation.manual import (
    manual_greedy_generate,
    output_token_digest,
)


def _first_model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration) as error:
        raise ValueError(
            "device must be supplied when the model has no discoverable parameters"
        ) from error


def _token_values(token_ids: torch.Tensor, *, batch_size: int) -> list[int]:
    values = token_ids.detach().to(device="cpu")
    if values.ndim == 1:
        values = values.reshape(batch_size, -1)
    if values.ndim != 2 or values.shape[0] != batch_size or values.shape[1] != 1:
        raise ValueError(
            "manual generation callbacks must provide one token per batch row"
        )
    return [int(value) for value in values[:, 0].tolist()]


class ManualGreedyBatchRunner:
    """Run equal-shape static batches through the manual forward loop."""

    def __init__(
        self,
        model: Any,
        *,
        device: str | torch.device | None = None,
        logits_mode: str = "last",
        eos_token_id: int | Sequence[int] | None = None,
        pad_token_id: int = 0,
    ) -> None:
        self.model = model
        self.device = (
            torch.device(device) if device is not None else _first_model_device(model)
        )
        self.logits_mode = logits_mode
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        requests: Sequence[RequestSpec],
        recorders: Sequence[RequestEventRecorder],
    ) -> tuple[RequestOutcome, ...]:
        if not requests:
            raise ValueError("A generation batch must contain at least one request")
        if len(requests) != len(recorders):
            raise ValueError("requests and recorders must have equal lengths")
        prompt_lengths = {request.prompt_tokens for request in requests}
        output_lengths = {request.max_new_tokens for request in requests}
        if len(prompt_lengths) != 1:
            raise ValueError(
                "Manual static batching requires equal prompt lengths; "
                f"received {sorted(prompt_lengths)}"
            )
        if len(output_lengths) != 1:
            raise ValueError(
                "Manual static batching requires equal output lengths; "
                f"received {sorted(output_lengths)}"
            )

        batch_size = len(requests)
        prompt_tokens = next(iter(prompt_lengths))
        output_tokens = next(iter(output_lengths))
        for recorder in recorders:
            recorder.record(
                "prefill_start",
                metadata={
                    "runner": "manual_eager",
                    "batch_size": batch_size,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                },
            )

        input_ids = torch.tensor(
            [request.prompt_token_ids for request in requests],
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.ones_like(input_ids)

        def record_prefill_end(token_ids: torch.Tensor) -> None:
            _token_values(token_ids, batch_size=batch_size)
            timestamp_ns = max(recorder.now_ns() for recorder in recorders)
            for recorder in recorders:
                recorder.record("prefill_end", timestamp_ns=timestamp_ns)

        def record_token(token_index: int, token_ids: torch.Tensor) -> None:
            token_values = _token_values(token_ids, batch_size=batch_size)
            timestamp_ns = max(recorder.now_ns() for recorder in recorders)
            for recorder, token_id in zip(
                recorders,
                token_values,
                strict=True,
            ):
                recorder.mark_token_ready(
                    token_index,
                    token_id=token_id,
                    timestamp_ns=timestamp_ns,
                )
                recorder.mark_token_sent(
                    token_index,
                    token_id=token_id,
                    timestamp_ns=timestamp_ns,
                )

        generation = manual_greedy_generate(
            self.model,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            output_tokens=output_tokens,
            logits_mode=self.logits_mode,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            on_prefill_end=record_prefill_end,
            on_token=record_token,
        )

        outcomes: list[RequestOutcome] = []
        model_config = getattr(self.model, "config", None)
        for request, recorder, row_index in zip(
            requests,
            recorders,
            range(batch_size),
            strict=True,
        ):
            row = generation.row(row_index)
            token_row = [
                int(token)
                for token in row.detach().to(device="cpu").tolist()
            ]
            if self.eos_token_id is None and len(token_row) != output_tokens:
                raise RuntimeError(
                    "manual generation returned an unexpected continuation length"
                )
            recorder.record("completion")
            outcomes.append(
                RequestOutcome(
                    status="completed",
                    generated_token_ids=tuple(token_row),
                    metadata={
                        "runner": "manual_eager",
                        "model_id": getattr(model_config, "_name_or_path", None),
                        "batch_size": batch_size,
                        "prefill_forward_calls": 1,
                        "decode_forward_calls": max(0, len(token_row) - 1),
                        "cache": "past_key_values",
                        "output_token_sha256": output_token_digest(row[None, :]),
                    },
                )
            )
        return tuple(outcomes)


def write_manual_result(
    result: BenchmarkResult,
    path: Path,
    *,
    model_metadata: Mapping[str, Any],
    correctness: Mapping[str, Any],
) -> None:
    """Write common harness data plus manual-decoder metadata."""

    payload = result.to_dict()
    payload["manual_decode"] = {
        "model": dict(model_metadata),
        "correctness": dict(correctness),
        "cache": "explicit past_key_values handoff",
        "generation": "one prefill forward plus one forward per subsequent token",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
