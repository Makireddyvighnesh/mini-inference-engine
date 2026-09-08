"""Explicit autoregressive generation for the MiniLLM-L4 engine.

This backend deliberately calls the model forward method directly. It
performs one prompt prefill, selects the first token from the prefill logits,
and then performs one-token decode steps while carrying the model cache
forward.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Sequence

import torch

if TYPE_CHECKING:
    from minillm_l4.engine.kv_cache import ContiguousKvCache


PrefillCallback = Callable[[torch.Tensor], None]
TokenCallback = Callable[[int, torch.Tensor], None]


@dataclass(frozen=True)
class ManualGenerationResult:
    """A padded batch of generated tokens and the true length of each row."""

    token_ids: torch.Tensor
    sequence_lengths: tuple[int, ...]
    requested_output_tokens: int
    runtime: str = "manual_eager"

    def __post_init__(self) -> None:
        if self.token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, output_tokens]")
        if self.requested_output_tokens < 1:
            raise ValueError("requested_output_tokens must be positive")
        if self.token_ids.shape[1] != self.requested_output_tokens:
            raise ValueError("token_ids width must equal requested_output_tokens")
        if len(self.sequence_lengths) != self.token_ids.shape[0]:
            raise ValueError("sequence_lengths must contain one value per row")
        if any(
            length < 1 or length > self.requested_output_tokens
            for length in self.sequence_lengths
        ):
            raise ValueError("sequence lengths must be within the generated range")

    @property
    def batch_size(self) -> int:
        return int(self.token_ids.shape[0])

    @property
    def output_tokens_per_sequence(self) -> int:
        """Return the configured maximum, matching the benchmark contract."""

        return self.requested_output_tokens

    @property
    def aggregate_output_tokens(self) -> int:
        return sum(self.sequence_lengths)

    def row(self, index: int) -> torch.Tensor:
        """Return one row without any EOS or padding tokens."""

        return self.token_ids[index, : self.sequence_lengths[index]]


def _validate_inputs(
    inputs: dict[str, torch.Tensor],
    *,
    output_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if output_tokens < 1:
        raise ValueError("output_tokens must be positive")
    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")
    if input_ids is None or attention_mask is None:
        raise ValueError("inputs must contain input_ids and attention_mask")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids shape")
    batch_size, prompt_tokens = input_ids.shape
    if batch_size < 1 or prompt_tokens < 1:
        raise ValueError("input_ids must contain at least one token")
    return input_ids, attention_mask, int(batch_size), int(prompt_tokens)


def _logits_to_keep(logits_mode: str) -> int:
    if logits_mode == "last":
        return 1
    if logits_mode == "all":
        return 0
    raise ValueError("logits_mode must be 'last' or 'all'")


def _preallocate_attention_mask(
    attention_mask: torch.Tensor,
    *,
    output_tokens: int,
) -> torch.Tensor:
    """Allocate the prompt-plus-decode mask once for the whole request."""

    batch_size, prompt_tokens = attention_mask.shape
    total_tokens = prompt_tokens + max(0, output_tokens - 1)
    full_mask = torch.ones(
        (batch_size, total_tokens),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    full_mask[:, :prompt_tokens].copy_(attention_mask)
    return full_mask


def _decode_attention_mask(
    full_attention_mask: torch.Tensor,
    *,
    prompt_tokens: int,
    decode_step: int,
) -> torch.Tensor:
    required_tokens = prompt_tokens + decode_step + 1
    return full_attention_mask[:, :required_tokens]


def _normalize_eos_ids(
    eos_token_id: int | Sequence[int] | None,
) -> tuple[int, ...]:
    if eos_token_id is None:
        return ()
    if isinstance(eos_token_id, int):
        values = (eos_token_id,)
    else:
        values = tuple(int(value) for value in eos_token_id)
    if not values or any(value < 0 for value in values):
        raise ValueError("eos_token_id must contain at least one non-negative ID")
    if len(set(values)) != len(values):
        raise ValueError("eos_token_id values must be unique")
    return values


def _select_next_token(output: Any) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if logits is None or not isinstance(logits, torch.Tensor):
        raise TypeError("model output must contain a tensor named logits")
    if logits.ndim != 3 or logits.shape[1] < 1:
        raise ValueError("model logits must have shape [batch, sequence, vocab]")
    return logits[:, -1, :].argmax(dim=-1, keepdim=True)


def _is_eos(token_ids: torch.Tensor, eos_ids: tuple[int, ...]) -> torch.Tensor:
    if not eos_ids:
        return torch.zeros(
            token_ids.shape[0],
            dtype=torch.bool,
            device=token_ids.device,
        )
    matches = torch.zeros(
        token_ids.shape[0],
        dtype=torch.bool,
        device=token_ids.device,
    )
    for eos_id in eos_ids:
        matches |= token_ids[:, 0] == eos_id
    return matches


def manual_greedy_generate(
    model: Any,
    inputs: dict[str, torch.Tensor],
    *,
    output_tokens: int,
    logits_mode: str = "last",
    eos_token_id: int | Sequence[int] | None = None,
    pad_token_id: int = 0,
    on_prefill_end: PrefillCallback | None = None,
    on_token: TokenCallback | None = None,
    kv_cache: ContiguousKvCache | None = None,
    sequence_output_limits: Sequence[int] | None = None,
) -> ManualGenerationResult:
    """Generate tokens with an explicit prefill and cached decode loop.

    eos_token_id is opt-in. When it is omitted, the function emits exactly
    output_tokens tokens, matching the fixed-length Phase 1 benchmark. If EOS
    is supplied, rows are padded internally and sequence_lengths identifies
    the useful prefix of each row.
    """

    input_ids, attention_mask, batch_size, prompt_tokens = _validate_inputs(
        inputs,
        output_tokens=output_tokens,
    )
    eos_ids = _normalize_eos_ids(eos_token_id)
    if sequence_output_limits is None:
        output_limits = (output_tokens,) * batch_size
    else:
        output_limits = tuple(int(value) for value in sequence_output_limits)
        if len(output_limits) != batch_size:
            raise ValueError(
                "sequence_output_limits must contain one value per batch row"
            )
        if any(value < 1 or value > output_tokens for value in output_limits):
            raise ValueError(
                "sequence output limits must be between 1 and output_tokens"
            )
    if pad_token_id < 0:
        raise ValueError("pad_token_id must be non-negative")
    logits_to_keep = _logits_to_keep(logits_mode)
    full_attention_mask = _preallocate_attention_mask(
        attention_mask,
        output_tokens=output_tokens,
    )
    prompt_inputs = {
        **inputs,
        "input_ids": input_ids,
        "attention_mask": full_attention_mask[:, :prompt_tokens],
    }
    if kv_cache is not None:
        if kv_cache.batch_size != batch_size:
            raise ValueError(
                "KV cache batch size must match generation input batch size"
            )
        kv_cache.prepare_append(prompt_tokens)
        prompt_inputs["past_key_values"] = kv_cache.backend_cache

    generated: list[torch.Tensor] = []
    sequence_lengths = torch.tensor(
        output_limits,
        dtype=torch.long,
        device=input_ids.device,
    )
    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=input_ids.device,
    )

    with torch.inference_mode():
        prefill_output = model(
            **prompt_inputs,
            use_cache=True,
            return_dict=True,
            logits_to_keep=logits_to_keep,
        )
        past_key_values = getattr(prefill_output, "past_key_values", None)
        if past_key_values is None:
            raise RuntimeError(
                "The model did not return past_key_values during prefill"
            )
        if kv_cache is not None:
            kv_cache.commit_append(prompt_tokens, past_key_values)
        next_token = _select_next_token(prefill_output)
        generated.append(next_token)
        eos_matches = _is_eos(next_token, eos_ids)
        if eos_ids:
            sequence_lengths[eos_matches] = 1
            finished |= eos_matches
        finished |= sequence_lengths <= 1
        if on_prefill_end is not None:
            on_prefill_end(next_token.detach())
        if on_token is not None:
            on_token(0, next_token.detach())
        del prefill_output

        for decode_step in range(output_tokens - 1):
            if eos_ids and bool(torch.all(finished)):
                break

            decode_input = next_token
            if eos_ids and bool(torch.any(finished)):
                decode_input = next_token.clone()
                decode_input[finished, 0] = pad_token_id
            decode_attention_mask = _decode_attention_mask(
                full_attention_mask,
                prompt_tokens=prompt_tokens,
                decode_step=decode_step,
            )
            if kv_cache is not None:
                kv_cache.prepare_append(1)
            decode_output = model(
                input_ids=decode_input,
                attention_mask=decode_attention_mask,
                past_key_values=past_key_values,
                **(
                    {
                        "position_ids": prompt_inputs["position_ids"][:, -1:]
                        + decode_step
                        + 1
                    }
                    if "position_ids" in prompt_inputs
                    else {}
                ),
                use_cache=True,
                return_dict=True,
                logits_to_keep=logits_to_keep,
            )
            past_key_values = getattr(decode_output, "past_key_values", None)
            if past_key_values is None:
                raise RuntimeError(
                    "The model did not return past_key_values during decoding"
                )
            if kv_cache is not None:
                kv_cache.commit_append(1, past_key_values)
            next_token = _select_next_token(decode_output)
            generated.append(next_token)
            eos_matches = _is_eos(next_token, eos_ids)
            if eos_ids:
                newly_finished = eos_matches & ~finished
                sequence_lengths[newly_finished] = decode_step + 2
                finished |= eos_matches
            finished |= sequence_lengths <= decode_step + 2
            if on_token is not None:
                on_token(decode_step + 1, next_token.detach())
            del decode_output

    if len(generated) < output_tokens:
        pad = torch.full(
            (batch_size, output_tokens - len(generated)),
            pad_token_id,
            dtype=generated[0].dtype,
            device=generated[0].device,
        )
        token_ids = torch.cat((*generated, pad), dim=1)
    else:
        token_ids = torch.cat(generated, dim=1)

    return ManualGenerationResult(
        token_ids=token_ids,
        sequence_lengths=tuple(int(value) for value in sequence_lengths.tolist()),
        requested_output_tokens=output_tokens,
        runtime=(
            "manual_contiguous_cache" if kv_cache is not None else "manual_eager"
        ),
    )


def output_token_digest(token_ids: torch.Tensor) -> str:
    """Return a stable digest for generated token IDs."""

    contiguous = token_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()
