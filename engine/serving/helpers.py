"""Small helpers shared by the serving runners and older benchmark runners."""

from __future__ import annotations

import time
from typing import Any

import torch

from minillm_l4.engine.events import RequestEventRecorder


def _first_model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration) as error:
        raise ValueError(
            "device must be supplied when the model has no discoverable parameters"
        ) from error


def _cache_layers(cache: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise TypeError("model cache must expose layer objects")
    output: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_index, layer in enumerate(layers):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise TypeError(f"model cache layer {layer_index} is not initialized")
        if keys.ndim != 4 or values.shape != keys.shape:
            raise TypeError("model cache layers must have matching [batch, heads, tokens, dim] tensors")
        output.append((keys, values))
    if not output:
        raise TypeError("model cache must contain at least one layer")
    return tuple(output)


def _paged_activation_dtype(model: Any) -> torch.dtype:
    """Choose a cache dtype matching the model's non-FP8 activations."""

    base_model = getattr(model, "model", None)
    candidates = (
        getattr(getattr(base_model, "embed_tokens", None), "weight", None),
        getattr(getattr(base_model, "norm", None), "weight", None),
    )
    for candidate in candidates:
        dtype = getattr(candidate, "dtype", None)
        if isinstance(dtype, torch.dtype) and not str(dtype).startswith("torch.float8"):
            return dtype
    # Qwen's FP8 projections compute into the configured fallback activation
    # type.  bfloat16 is the L4-friendly fallback when the activation parameter
    # dtype cannot be discovered from a wrapper or fixture model.
    return torch.bfloat16


def _wait_until_ms(recorder: RequestEventRecorder, target_ms: float) -> None:
    while recorder.now_ns() / 1_000_000.0 < target_ms:
        remaining_ms = target_ms - recorder.now_ns() / 1_000_000.0
        time.sleep(min(remaining_ms / 1000.0, 0.01))
