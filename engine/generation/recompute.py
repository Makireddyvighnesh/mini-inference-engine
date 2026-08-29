"""Greedy reference decoding that intentionally does not use a KV cache."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from .manual import (
    ManualGenerationResult,
    PrefillCallback,
    TokenCallback,
    _is_eos,
    _logits_to_keep,
    _normalize_eos_ids,
    _select_next_token,
    _validate_inputs,
)


def recompute_greedy_generate(
    model: Any,
    inputs: dict[str, torch.Tensor],
    *,
    output_tokens: int,
    logits_mode: str = "last",
    eos_token_id: int | Sequence[int] | None = None,
    pad_token_id: int = 0,
    on_prefill_end: PrefillCallback | None = None,
    on_token: TokenCallback | None = None,
) -> ManualGenerationResult:
    """Re-run the complete prompt plus generated prefix for every token."""

    input_ids, attention_mask, batch_size, _ = _validate_inputs(
        inputs,
        output_tokens=output_tokens,
    )
    eos_ids = _normalize_eos_ids(eos_token_id)
    if pad_token_id < 0:
        raise ValueError("pad_token_id must be non-negative")
    logits_to_keep = _logits_to_keep(logits_mode)
    sequence = input_ids
    sequence_mask = attention_mask
    generated: list[torch.Tensor] = []
    sequence_lengths = torch.full(
        (batch_size,),
        output_tokens,
        dtype=torch.long,
        device=input_ids.device,
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

    with torch.inference_mode():
        for token_index in range(output_tokens):
            output = model(
                input_ids=sequence,
                attention_mask=sequence_mask,
                use_cache=False,
                return_dict=True,
                logits_to_keep=logits_to_keep,
            )
            next_token = _select_next_token(output)
            generated.append(next_token)
            eos_matches = _is_eos(next_token, eos_ids)
            if eos_ids:
                newly_finished = eos_matches & ~finished
                sequence_lengths[newly_finished] = token_index + 1
                finished |= eos_matches

            if token_index == 0 and on_prefill_end is not None:
                on_prefill_end(next_token.detach())
            if on_token is not None:
                on_token(token_index, next_token.detach())
            del output

            if token_index + 1 == output_tokens or (
                eos_ids and bool(torch.all(finished))
            ):
                break
            append_token = next_token
            if eos_ids and bool(torch.any(finished)):
                append_token = next_token.clone()
                append_token[finished, 0] = pad_token_id
            sequence = torch.cat((sequence, append_token), dim=1)
            sequence_mask = torch.cat(
                (
                    sequence_mask,
                    torch.ones(
                        (batch_size, 1),
                        dtype=sequence_mask.dtype,
                        device=sequence_mask.device,
                    ),
                ),
                dim=1,
            )

    if len(generated) < output_tokens:
        padding = torch.full(
            (batch_size, output_tokens - len(generated)),
            pad_token_id,
            dtype=generated[0].dtype,
            device=generated[0].device,
        )
        token_ids = torch.cat((*generated, padding), dim=1)
    else:
        token_ids = torch.cat(generated, dim=1)

    return ManualGenerationResult(
        token_ids=token_ids,
        sequence_lengths=tuple(int(value) for value in sequence_lengths.tolist()),
        requested_output_tokens=output_tokens,
        runtime="recompute_eager",
    )
