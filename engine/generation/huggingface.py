"""The Hugging Face generation backend kept for the trusted baseline."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class GenerationResult:
    """Generated continuation tokens, excluding the input prompt."""

    token_ids: torch.Tensor
    runtime: str
    output_tokens_per_sequence: int

    @property
    def batch_size(self) -> int:
        return int(self.token_ids.shape[0])

    @property
    def aggregate_output_tokens(self) -> int:
        return self.batch_size * self.output_tokens_per_sequence


def _logits_to_keep(logits_mode: str) -> int:
    if logits_mode == "last":
        return 1
    if logits_mode == "all":
        return 0
    raise ValueError("logits_mode must be 'last' or 'all'")


def transformers_greedy_generate(
    model: Any,
    inputs: dict[str, torch.Tensor],
    *,
    output_tokens: int,
    logits_mode: str = "last",
    streamer: Any | None = None,
) -> GenerationResult:
    """Run exact-length greedy generation through model.generate."""

    if output_tokens < 1:
        raise ValueError("output_tokens must be positive")
    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")
    if input_ids is None or attention_mask is None:
        raise ValueError("inputs must contain input_ids and attention_mask")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids shape")

    prompt_tokens = int(input_ids.shape[1])
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.do_sample = False
    generation_config.max_new_tokens = output_tokens
    generation_config.min_new_tokens = None
    generation_config.use_cache = True
    for sampling_parameter, neutral_value in (
        ("temperature", 1.0),
        ("top_p", 1.0),
        ("top_k", 50),
    ):
        if hasattr(generation_config, sampling_parameter):
            setattr(generation_config, sampling_parameter, neutral_value)
    generation_config.eos_token_id = None

    generate_kwargs: dict[str, Any] = {
        **inputs,
        "generation_config": generation_config,
        "logits_to_keep": _logits_to_keep(logits_mode),
    }
    if streamer is not None:
        generate_kwargs["streamer"] = streamer
    with torch.inference_mode():
        sequences = model.generate(**generate_kwargs)

    continuation = sequences[:, prompt_tokens:]
    if continuation.shape[1] != output_tokens:
        raise RuntimeError(
            "Transformers generate returned an unexpected continuation length: "
            f"expected {output_tokens}, received {continuation.shape[1]}"
        )
    return GenerationResult(
        token_ids=continuation,
        runtime="transformers_generate",
        output_tokens_per_sequence=output_tokens,
    )
