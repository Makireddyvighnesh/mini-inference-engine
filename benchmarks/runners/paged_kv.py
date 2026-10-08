"""Static generation runners backed by paged KV storage.

The module retains a dense-gather control runner and direct block-table
attention runners. CUDA one-token decode uses a fused Triton kernel while the
readable PyTorch implementation remains the correctness fallback.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import torch

from ..core.harness import BenchmarkResult, RequestEventRecorder
from ..core.schemas import RequestOutcome, RequestSpec
from ..core.tracing import ExecutionTrace
from minillm_l4.engine.generation.manual import output_token_digest
from minillm_l4.engine.kv_cache import (
    PagedKvAllocator,
    PagedKvCache,
    select_decode_block_tokens,
    select_decode_split_count,
)
from minillm_l4.engine.kv_cache.qwen3_paged import install_paged_qwen3_attention
from minillm_l4.engine.kv_cache.triton_paged_attention import triton_is_available


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


def _select_next_token(output: Any) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[1] < 1:
        raise TypeError("model output logits must have shape [batch, sequence, vocab]")
    return logits[:, -1, :].argmax(dim=-1, keepdim=True)


def _token_values(token_ids: torch.Tensor, *, batch_size: int) -> list[int]:
    values = token_ids.detach().to(device="cpu")
    if values.shape != (batch_size, 1):
        raise ValueError("token callbacks must contain one token per batch row")
    return [int(value) for value in values[:, 0].tolist()]


def _synchronize(device: torch.device) -> None:
    """Synchronize only when CUDA timings need completed device work."""

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _elapsed_ms(started: float, device: torch.device) -> float:
    _synchronize(device)
    return (perf_counter() - started) * 1000.0


def _cache_dimensions(cache: Any) -> tuple[int, int, int, torch.dtype, torch.device]:
    layers = _cache_layers(cache)
    keys = layers[0][0]
    return (
        len(layers),
        int(keys.shape[1]),
        int(keys.shape[3]),
        keys.dtype,
        keys.device,
    )


def _make_paged_cache(
    dense_cache: Any,
    *,
    allocator: PagedKvAllocator,
    owner_ids: Sequence[str],
) -> PagedKvCache:
    layers = _cache_layers(dense_cache)
    num_layers, num_kv_heads, head_dim, dtype, device = _cache_dimensions(dense_cache)
    batch_size = len(owner_ids)
    if int(layers[0][0].shape[0]) != batch_size:
        raise ValueError("prefill cache batch does not match request count")
    cache = PagedKvCache(
        allocator,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    for owner_id in owner_ids:
        allocator.allocate(owner_id)
    for row, owner_id in enumerate(owner_ids):
        cache.append(
            owner_id,
            tuple(
                (keys[row : row + 1], values[row : row + 1])
                for keys, values in layers
            ),
        )
    return cache


class PagedKvBatchRunner:
    """Run equal-shape static batches through the paged gather path.

    Equal prompt and output lengths keep this runner focused on the allocator
    comparison.  Variable-length scheduling remains covered by the Phase 5
    runner; this path deliberately isolates page allocation and correctness.
    """

    def __init__(
        self,
        model: Any,
        *,
        block_size: int,
        num_blocks: int,
        device: str | torch.device | None = None,
        logits_mode: str = "last",
        pad_token_id: int = 0,
        trace_enabled: bool = False,
    ) -> None:
        if int(block_size) < 1:
            raise ValueError("block_size must be positive")
        if int(num_blocks) < 1:
            raise ValueError("num_blocks must be positive")
        if logits_mode not in {"last", "all"}:
            raise ValueError("logits_mode must be 'last' or 'all'")
        self.model = model
        self.block_size = int(block_size)
        self.num_blocks = int(num_blocks)
        self.device = (
            torch.device(device) if device is not None else _first_model_device(model)
        )
        self.logits_mode = logits_mode
        self.pad_token_id = int(pad_token_id)
        self.trace_enabled = bool(trace_enabled)
        self.last_cache_snapshot: dict[str, Any] | None = None
        self.last_execution_trace: dict[str, Any] | None = None

    @property
    def runner_name(self) -> str:
        return "paged_kv_gather"

    def __call__(
        self,
        requests: Sequence[RequestSpec],
        recorders: Sequence[RequestEventRecorder],
    ) -> tuple[RequestOutcome, ...]:
        if not requests:
            raise ValueError("a generation batch must contain at least one request")
        if len(requests) != len(recorders):
            raise ValueError("requests and recorders must have equal lengths")
        prompt_lengths = {request.prompt_tokens for request in requests}
        output_lengths = {request.max_new_tokens for request in requests}
        if len(prompt_lengths) != 1 or len(output_lengths) != 1:
            raise ValueError(
                "paged static batching requires equal prompt and output lengths"
            )

        batch_size = len(requests)
        prompt_tokens = next(iter(prompt_lengths))
        output_tokens = next(iter(output_lengths))
        owner_ids = tuple(request.request_id for request in requests)
        trace = ExecutionTrace(
            device=self.device,
            started_ns=recorders[0].run_started_ns,
            enabled=self.trace_enabled,
        )
        self.last_execution_trace = None
        with trace.span(
            "kv_allocator_and_physical_cache_allocation",
            category="kv_memory",
            metadata={
                "block_size": self.block_size,
                "num_blocks": self.num_blocks,
                "capacity_tokens": prompt_tokens + output_tokens - 1,
            },
        ):
            allocator = PagedKvAllocator(
                num_blocks=self.num_blocks,
                block_size=self.block_size,
                max_sequence_tokens=prompt_tokens + output_tokens - 1,
            )
        with trace.span(
            "input_rectangularization_and_device_transfer",
            category="request_preparation",
            metadata={"input_shape": [batch_size, prompt_tokens]},
        ):
            input_ids = torch.tensor(
                [request.prompt_token_ids for request in requests],
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.ones_like(input_ids)
        self.last_cache_snapshot = None

        for recorder in recorders:
            recorder.record(
                "prefill_start",
                metadata={
                    "runner": self.runner_name,
                    "block_size": self.block_size,
                    "num_blocks": self.num_blocks,
                    "batch_size": batch_size,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                },
            )

        paged_cache: PagedKvCache | None = None
        generated: list[torch.Tensor] = []
        try:
            with torch.inference_mode():
                with trace.span(
                    "prefill_model_forward",
                    category="model_execution",
                    gpu=True,
                    metadata={"input_shape": [batch_size, prompt_tokens]},
                ):
                    prefill_output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                        return_dict=True,
                        logits_to_keep=1 if self.logits_mode == "last" else 0,
                    )
                prefill_dense_cache = getattr(prefill_output, "past_key_values", None)
                if prefill_dense_cache is None:
                    raise RuntimeError("model did not return a prefill KV cache")
                with trace.span(
                    "prefill_cache_materialization",
                    category="kv_memory",
                    gpu=True,
                    metadata={"prompt_tokens": prompt_tokens},
                ):
                    paged_cache = _make_paged_cache(
                        prefill_dense_cache,
                        allocator=allocator,
                        owner_ids=owner_ids,
                    )
                with trace.span(
                    "first_token_selection",
                    category="sampling",
                    gpu=True,
                    metadata={"algorithm": "greedy_argmax"},
                ):
                    next_token = _select_next_token(prefill_output)
                generated.append(next_token)
                with trace.span(
                    "first_token_host_transfer_and_stream_emit",
                    category="streaming",
                    metadata={"token_index": 0},
                ):
                    values = _token_values(next_token, batch_size=batch_size)
                    timestamp_ns = max(recorder.now_ns() for recorder in recorders)
                    for recorder in recorders:
                        recorder.record("prefill_end", timestamp_ns=timestamp_ns)
                    for recorder, token_id in zip(recorders, values, strict=True):
                        recorder.mark_token_ready(
                            0,
                            token_id=token_id,
                            timestamp_ns=timestamp_ns,
                        )
                        recorder.mark_token_sent(
                            0,
                            token_id=token_id,
                            timestamp_ns=timestamp_ns,
                        )
                del prefill_output

                for decode_step in range(output_tokens - 1):
                    assert paged_cache is not None
                    cached_tokens = prompt_tokens + decode_step
                    with trace.span(
                        "decode_cache_gather",
                        category="kv_memory",
                        metadata={
                            "decode_step": decode_step + 1,
                            "cached_tokens": cached_tokens,
                        },
                    ):
                        dense_cache = paged_cache.as_dynamic_cache(
                            owner_ids,
                            model_config=self.model.config,
                        )
                    with trace.span(
                        "decode_model_forward",
                        category="model_execution",
                        gpu=True,
                        metadata={
                            "decode_step": decode_step + 1,
                            "input_shape": [batch_size, 1],
                            "cached_tokens": cached_tokens,
                        },
                    ):
                        decode_output = self.model(
                            input_ids=next_token,
                            attention_mask=torch.ones(
                                (batch_size, cached_tokens + 1),
                                dtype=torch.long,
                                device=self.device,
                            ),
                            past_key_values=dense_cache,
                            use_cache=True,
                            return_dict=True,
                            logits_to_keep=1 if self.logits_mode == "last" else 0,
                        )
                    updated_dense_cache = getattr(decode_output, "past_key_values", None)
                    if updated_dense_cache is None:
                        raise RuntimeError("model did not return a decode KV cache")
                    with trace.span(
                        "decode_cache_append",
                        category="kv_memory",
                        gpu=True,
                        metadata={"decode_step": decode_step + 1},
                    ):
                        paged_cache.append_from_dynamic_cache(
                            owner_ids,
                            updated_dense_cache,
                            previous_token_counts=(cached_tokens,) * batch_size,
                            appended_token_counts=1,
                        )
                    with trace.span(
                        "decode_token_selection",
                        category="sampling",
                        gpu=True,
                        metadata={
                            "decode_step": decode_step + 1,
                            "algorithm": "greedy_argmax",
                        },
                    ):
                        next_token = _select_next_token(decode_output)
                    generated.append(next_token)
                    with trace.span(
                        "decode_host_transfer_and_stream_emit",
                        category="streaming",
                        metadata={"token_index": decode_step + 1},
                    ):
                        values = _token_values(next_token, batch_size=batch_size)
                        timestamp_ns = max(recorder.now_ns() for recorder in recorders)
                        for recorder, token_id in zip(recorders, values, strict=True):
                            token_index = decode_step + 1
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
                    del decode_output

            assert paged_cache is not None
            allocator_snapshot = allocator.snapshot()
            physical_bytes = int(
                paged_cache.key_blocks.numel() * paged_cache.key_blocks.element_size()
                + paged_cache.value_blocks.numel() * paged_cache.value_blocks.element_size()
            )
            self.last_cache_snapshot = {
                **allocator_snapshot,
                "runner": self.runner_name,
                "block_size": self.block_size,
                "num_blocks": self.num_blocks,
                "num_layers": paged_cache.num_layers,
                "num_kv_heads": paged_cache.num_kv_heads,
                "head_dim": paged_cache.head_dim,
                "dtype": str(paged_cache.dtype),
                "device": str(paged_cache.device),
                "physical_kv_bytes": physical_bytes,
                "gather_path": True,
                "paged_attention_kernel": False,
            }
            with trace.span(
                "output_materialization",
                category="response_finalization",
                metadata={"batch_size": batch_size},
            ):
                token_matrix = torch.cat(generated, dim=1).detach().to("cpu")
                token_rows = tuple(
                    tuple(int(value) for value in token_matrix[row].tolist())
                    for row in range(batch_size)
                )
            for recorder in recorders:
                recorder.record("completion")
            return tuple(
                RequestOutcome(
                    status="completed",
                    generated_token_ids=row,
                    metadata={
                        "runner": self.runner_name,
                        "block_size": self.block_size,
                        "num_blocks": self.num_blocks,
                        "cache": dict(self.last_cache_snapshot),
                        "output_token_sha256": output_token_digest(
                            torch.tensor([row], dtype=torch.long)
                        ),
                    },
                )
                for row in token_rows
            )
        finally:
            with trace.span("kv_resource_release", category="kv_memory"):
                if paged_cache is not None:
                    for owner_id in owner_ids:
                        if owner_id in allocator.sequence_ids:
                            paged_cache.release(owner_id)
            trace.counter("batch_size", batch_size)
            trace.counter("prompt_tokens_per_request", prompt_tokens)
            trace.counter("output_tokens_per_request", output_tokens)
            trace.counter("cache_mode", "paged_gather")
            self.last_execution_trace = trace.to_dict()
            if self.last_cache_snapshot is not None:
                self.last_cache_snapshot["execution_trace"] = dict(
                    self.last_execution_trace
                )
                self.last_cache_snapshot["resources_released"] = True
                self.last_cache_snapshot["active_sequence_count_after_release"] = (
                    allocator.active_sequence_count
                )


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


class PagedAttentionBatchRunner:
    """Run static Qwen3 batches through direct block-table attention.

    The allocator reserves the prompt span before prefill and reserves one new
    logical position before each decode step.  Every Qwen3 attention layer
    writes its own K/V segment into those positions, then
    :func:`paged_attention` reads the physical pages block by block.  No
    ``DynamicCache`` or left-padded dense KV tensor is constructed on this
    path.

    Multi-token direct prefill uses the correctness-first PyTorch indirection
    algorithm; supported CUDA decode shapes use the fused Triton kernel. This
    runner is deliberately kept separate from
    :class:`PagedKvBatchRunner`, whose dense-gather path is the fallback and
    useful control experiment.
    """

    def __init__(
        self,
        model: Any,
        *,
        block_size: int,
        num_blocks: int,
        device: str | torch.device | None = None,
        logits_mode: str = "last",
        pad_token_id: int = 0,
        kv_dtype: torch.dtype | None = None,
        prefill_backend: str = "paged_reference",
        decode_backend: str = "auto",
        decode_sdpa_compat: bool | None = None,
        trace_enabled: bool = False,
    ) -> None:
        if int(block_size) < 1:
            raise ValueError("block_size must be positive")
        if int(num_blocks) < 1:
            raise ValueError("num_blocks must be positive")
        if logits_mode not in {"last", "all"}:
            raise ValueError("logits_mode must be 'last' or 'all'")
        if kv_dtype is not None and not isinstance(kv_dtype, torch.dtype):
            raise TypeError("kv_dtype must be a torch.dtype when supplied")
        if prefill_backend not in {"paged_reference", "sdpa"}:
            raise ValueError("prefill_backend must be 'paged_reference' or 'sdpa'")
        if decode_backend not in {"auto", "torch", "triton"}:
            raise ValueError("decode_backend must be 'auto', 'torch', or 'triton'")
        self.model = model
        self.block_size = int(block_size)
        self.num_blocks = int(num_blocks)
        self.device = (
            torch.device(device) if device is not None else _first_model_device(model)
        )
        self.logits_mode = logits_mode
        self.pad_token_id = int(pad_token_id)
        self.kv_dtype = kv_dtype
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend
        self.decode_sdpa_compat = (
            prefill_backend == "sdpa" and self.device.type == "cuda"
            and decode_backend != "torch" and self.block_size <= 128
            and _paged_activation_dtype(model) == torch.bfloat16
            and getattr(getattr(model, "config", None), "_attn_implementation", None) == "sdpa"
            if decode_sdpa_compat is None else bool(decode_sdpa_compat)
        )
        if self.decode_sdpa_compat and decode_backend == "torch":
            raise ValueError("SDPA-compatible numerics require Triton decode, not the torch reference")
        self.trace_enabled = bool(trace_enabled)
        self.last_cache_snapshot: dict[str, Any] | None = None
        self.last_execution_trace: dict[str, Any] | None = None
        self._attention_wrappers = install_paged_qwen3_attention(model)

    @property
    def runner_name(self) -> str:
        if self.prefill_backend == "sdpa":
            return "paged_kv_sdpa_prefill_direct_decode"
        return "paged_kv_direct_attention"

    def _new_allocator(
        self,
        *,
        prompt_tokens: int,
        output_tokens: int,
    ) -> PagedKvAllocator:
        return PagedKvAllocator(
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            max_sequence_tokens=prompt_tokens + output_tokens - 1,
        )

    def _new_cache(
        self,
        *,
        owner_ids: Sequence[str],
        prompt_tokens: int,
        output_tokens: int,
    ) -> tuple[PagedKvAllocator, PagedKvCache]:
        config = getattr(self.model, "config", None)
        if config is None:
            raise ValueError("direct paged attention requires model.config")
        num_layers = int(getattr(config, "num_hidden_layers"))
        num_kv_heads = int(getattr(config, "num_key_value_heads"))
        head_dim = int(
            getattr(
                config,
                "head_dim",
                int(config.hidden_size) // int(config.num_attention_heads),
            )
        )
        allocator = self._new_allocator(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )
        cache = PagedKvCache(
            allocator,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=self.kv_dtype or _paged_activation_dtype(self.model),
            device=self.device,
        )
        for owner_id in owner_ids:
            allocator.allocate(owner_id, token_count=prompt_tokens)
        return allocator, cache

    @staticmethod
    def _physical_bytes(cache: PagedKvCache) -> int:
        return int(
            cache.key_blocks.numel() * cache.key_blocks.element_size()
            + cache.value_blocks.numel() * cache.value_blocks.element_size()
        )

    def __call__(
        self,
        requests: Sequence[RequestSpec],
        recorders: Sequence[RequestEventRecorder],
    ) -> tuple[RequestOutcome, ...]:
        if not requests:
            raise ValueError("a generation batch must contain at least one request")
        if len(requests) != len(recorders):
            raise ValueError("requests and recorders must have equal lengths")
        prompt_lengths = {request.prompt_tokens for request in requests}
        output_lengths = {request.max_new_tokens for request in requests}
        if len(prompt_lengths) != 1 or len(output_lengths) != 1:
            raise ValueError(
                "direct paged static batching requires equal prompt and output lengths"
            )

        trace = ExecutionTrace(
            device=self.device,
            started_ns=recorders[0].run_started_ns,
            enabled=self.trace_enabled,
        )
        self.last_execution_trace = None
        with trace.span(
            "request_validation_and_shape_setup",
            category="request_lifecycle",
            metadata={"batch_size": len(requests)},
        ):
            batch_size = len(requests)
            prompt_tokens = next(iter(prompt_lengths))
            output_tokens = next(iter(output_lengths))
            owner_ids = tuple(request.request_id for request in requests)
            decode_max_sequence_length = prompt_tokens + output_tokens - 1
            decode_split_count = select_decode_split_count(
                decode_max_sequence_length,
                batch_size=batch_size,
            )
            if self.decode_sdpa_compat:
                decode_split_count = 1
            decode_block_tokens = select_decode_block_tokens(
                decode_max_sequence_length
            )
            if self.decode_sdpa_compat:
                decode_block_tokens = 128
        with trace.span(
            "input_rectangularization_and_device_transfer",
            category="request_preparation",
            metadata={"input_shape": [batch_size, prompt_tokens]},
        ):
            input_ids = torch.tensor(
                [request.prompt_token_ids for request in requests],
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.ones_like(input_ids)
            prefill_positions = torch.arange(
                prompt_tokens,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0).expand(batch_size, -1)
        self.last_cache_snapshot = None
        with trace.span("backend_resolution", category="runner_control"):
            uses_triton_decode = bool(
                self.decode_backend != "torch"
                and self.device.type == "cuda"
                and triton_is_available()
            )
            resolved_decode_backend = (
                "triton_paged_decode"
                if uses_triton_decode
                else "torch_blockwise_reference"
            )

        for recorder in recorders:
            recorder.record(
                "prefill_start",
                metadata={
                    "runner": self.runner_name,
                    "block_size": self.block_size,
                    "num_blocks": self.num_blocks,
                    "batch_size": batch_size,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "prefill_backend": self.prefill_backend,
                    "decode_backend": resolved_decode_backend,
                },
            )

        allocator: PagedKvAllocator
        paged_cache: PagedKvCache | None = None
        cache_setup_ms = 0.0
        prefill_model_ms = 0.0
        prefill_cache_materialization_ms = 0.0
        decode_model_ms = 0.0
        if self.prefill_backend == "paged_reference":
            with trace.span(
                "kv_allocator_and_physical_cache_allocation",
                category="kv_memory",
                metadata={
                    "block_size": self.block_size,
                    "num_blocks": self.num_blocks,
                },
            ) as measurement:
                allocator, paged_cache = self._new_cache(
                    owner_ids=owner_ids,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                )
            cache_setup_ms = measurement.wall_ms
        else:
            with trace.span(
                "kv_allocator_and_physical_cache_allocation",
                category="kv_memory",
                metadata={
                    "block_size": self.block_size,
                    "num_blocks": self.num_blocks,
                },
            ):
                allocator = self._new_allocator(
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                )
        generated: list[torch.Tensor] = []
        try:
            with torch.inference_mode():
                with trace.span(
                    "prefill_model_forward",
                    category="model_execution",
                    gpu=True,
                    metadata={
                        "input_shape": [batch_size, prompt_tokens],
                        "backend": self.prefill_backend,
                    },
                ) as measurement:
                    if self.prefill_backend == "paged_reference":
                        assert paged_cache is not None
                        prefill_output = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=prefill_positions,
                            use_cache=False,
                            return_dict=True,
                            logits_to_keep=1 if self.logits_mode == "last" else 0,
                            paged_kv_cache=paged_cache,
                            paged_sequence_ids=owner_ids,
                            paged_query_start_positions=(0,) * batch_size,
                        )
                    else:
                        prefill_output = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=prefill_positions,
                            use_cache=True,
                            return_dict=True,
                            logits_to_keep=1 if self.logits_mode == "last" else 0,
                        )
                prefill_model_ms = measurement.wall_ms

                if self.prefill_backend == "sdpa":
                    dense_cache = getattr(prefill_output, "past_key_values", None)
                    if dense_cache is None:
                        raise RuntimeError("model did not return a prefill KV cache")
                    with trace.span(
                        "prefill_cache_materialization",
                        category="kv_memory",
                        gpu=True,
                        metadata={"prompt_tokens": prompt_tokens},
                    ) as measurement:
                        paged_cache = _make_paged_cache(
                            dense_cache,
                            allocator=allocator,
                            owner_ids=owner_ids,
                        )
                    prefill_cache_materialization_ms = measurement.wall_ms
                with trace.span(
                    "first_token_selection",
                    category="sampling",
                    gpu=True,
                    metadata={"algorithm": "greedy_argmax"},
                ):
                    next_token = _select_next_token(prefill_output)
                generated.append(next_token)
                with trace.span(
                    "first_token_host_transfer_and_stream_emit",
                    category="streaming",
                    metadata={"token_index": 0},
                ):
                    values = _token_values(next_token, batch_size=batch_size)
                    timestamp_ns = max(recorder.now_ns() for recorder in recorders)
                    for recorder in recorders:
                        recorder.record("prefill_end", timestamp_ns=timestamp_ns)
                    for recorder, token_id in zip(recorders, values, strict=True):
                        recorder.mark_token_ready(0, token_id=token_id, timestamp_ns=timestamp_ns)
                        recorder.mark_token_sent(0, token_id=token_id, timestamp_ns=timestamp_ns)
                del prefill_output

                for decode_step in range(output_tokens - 1):
                    assert paged_cache is not None
                    with trace.span(
                        "decode_schedule_and_kv_reservation",
                        category="scheduler_and_kv",
                        metadata={"decode_step": decode_step + 1},
                    ):
                        old_lengths = tuple(
                            allocator.get_block_table(owner_id).token_count
                            for owner_id in owner_ids
                        )
                        paged_cache.reserve_append(owner_ids, 1)
                    with trace.span(
                        "decode_metadata_materialization",
                        category="scheduler_and_kv",
                        metadata={"decode_step": decode_step + 1},
                    ):
                        decode_positions = torch.tensor(
                            old_lengths,
                            dtype=torch.long,
                            device=self.device,
                        ).unsqueeze(-1)
                        decode_block_tables = paged_cache.block_table_tensor(
                            owner_ids,
                            device=self.device,
                        ).to(dtype=torch.int32)
                        decode_sequence_lengths = torch.tensor(
                            tuple(length + 1 for length in old_lengths),
                            dtype=torch.int32,
                            device=self.device,
                        )
                    with trace.span(
                        "decode_model_forward",
                        category="model_execution",
                        gpu=True,
                        metadata={
                            "decode_step": decode_step + 1,
                            "input_shape": [batch_size, 1],
                            "backend": resolved_decode_backend,
                        },
                    ) as measurement:
                        decode_output = self.model(
                            input_ids=next_token,
                            attention_mask=torch.ones(
                                (batch_size, 1),
                                dtype=torch.long,
                                device=self.device,
                            ),
                            position_ids=decode_positions,
                            use_cache=False,
                            return_dict=True,
                            logits_to_keep=1 if self.logits_mode == "last" else 0,
                            paged_kv_cache=paged_cache,
                            paged_sequence_ids=owner_ids,
                            paged_query_start_positions=decode_positions,
                            paged_block_tables=decode_block_tables,
                            paged_sequence_lengths=decode_sequence_lengths,
                            paged_attention_backend=self.decode_backend,
                            paged_decode_split_count=decode_split_count,
                            paged_decode_max_sequence_length=(
                                decode_max_sequence_length
                            ),
                            paged_decode_block_tokens=decode_block_tokens,
                            paged_decode_use_gqa_reuse=False,
                            paged_decode_sdpa_compat=self.decode_sdpa_compat,
                        )
                    decode_model_ms += measurement.wall_ms
                    with trace.span(
                        "decode_token_selection",
                        category="sampling",
                        gpu=True,
                        metadata={
                            "decode_step": decode_step + 1,
                            "algorithm": "greedy_argmax",
                        },
                    ):
                        next_token = _select_next_token(decode_output)
                    generated.append(next_token)
                    with trace.span(
                        "decode_host_transfer_and_stream_emit",
                        category="streaming",
                        metadata={"token_index": decode_step + 1},
                    ):
                        values = _token_values(next_token, batch_size=batch_size)
                        timestamp_ns = max(recorder.now_ns() for recorder in recorders)
                        for recorder, token_id in zip(recorders, values, strict=True):
                            token_index = decode_step + 1
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
                    del decode_output

            with trace.span(
                "output_materialization",
                category="response_finalization",
            ):
                token_matrix = torch.cat(generated, dim=1).detach().to("cpu")
            token_rows = tuple(
                tuple(int(value) for value in token_matrix[row].tolist())
                for row in range(batch_size)
            )

            assert paged_cache is not None
            allocator_snapshot = allocator.snapshot()
            prefill_page_visits_per_layer = 0
            if self.prefill_backend == "paged_reference":
                prefill_page_visits_per_layer = batch_size * sum(
                    math.ceil(token_count / self.block_size)
                    for token_count in range(1, prompt_tokens + 1)
                )
            decode_page_visits_per_layer = batch_size * sum(
                math.ceil(token_count / self.block_size)
                for token_count in range(prompt_tokens + 1, prompt_tokens + output_tokens)
            )
            self.last_cache_snapshot = {
                **allocator_snapshot,
                "runner": self.runner_name,
                "block_size": self.block_size,
                "num_blocks": self.num_blocks,
                "num_layers": paged_cache.num_layers,
                "num_kv_heads": paged_cache.num_kv_heads,
                "head_dim": paged_cache.head_dim,
                "dtype": str(paged_cache.dtype),
                "device": str(paged_cache.device),
                "physical_kv_bytes": self._physical_bytes(paged_cache),
                "gather_path": False,
                "direct_paged_attention": True,
                "prefill_backend": self.prefill_backend,
                "decode_backend": resolved_decode_backend,
                "decode_split_count": decode_split_count,
                "decode_block_tokens": decode_block_tokens,
                "decode_gqa_reuse": False,
                "decode_max_sequence_length": decode_max_sequence_length,
                "decode_sdpa_compat": self.decode_sdpa_compat,
                "attention_backend": (
                    f"sdpa_prefill_{resolved_decode_backend}"
                    if self.prefill_backend == "sdpa"
                    else f"paged_reference_prefill_{resolved_decode_backend}"
                ),
                "paged_attention_kernel": uses_triton_decode,
                "stage_timings_ms": {
                    "cache_setup": cache_setup_ms,
                    "prefill_model": prefill_model_ms,
                    "prefill_cache_materialization": prefill_cache_materialization_ms,
                    "decode_model_total": decode_model_ms,
                },
                "execution_trace": trace.to_dict(),
                "page_visits": {
                    "prefill_per_layer": prefill_page_visits_per_layer,
                    "decode_per_layer": decode_page_visits_per_layer,
                    "prefill_all_layers": (
                        prefill_page_visits_per_layer * paged_cache.num_layers
                    ),
                    "decode_all_layers": (
                        decode_page_visits_per_layer * paged_cache.num_layers
                    ),
                },
            }
            for recorder in recorders:
                recorder.record("completion")
            model_config = getattr(self.model, "config", None)
            return tuple(
                RequestOutcome(
                    status="completed",
                    generated_token_ids=row,
                    metadata={
                        "runner": self.runner_name,
                        "model_id": getattr(model_config, "_name_or_path", None),
                        "batch_size": batch_size,
                        "cache": dict(self.last_cache_snapshot),
                        "output_token_sha256": output_token_digest(
                            torch.tensor([row], dtype=torch.long)
                        ),
                    },
                )
                for row in token_rows
            )
        finally:
            with trace.span("kv_resource_release", category="kv_memory"):
                if paged_cache is not None:
                    for owner_id in owner_ids:
                        if owner_id in allocator.sequence_ids:
                            paged_cache.release(owner_id)
            trace.counter("batch_size", batch_size)
            trace.counter("prompt_tokens_per_request", prompt_tokens)
            trace.counter("output_tokens_per_request", output_tokens)
            trace.counter("prefill_backend", self.prefill_backend)
            trace.counter("resolved_decode_backend", resolved_decode_backend)
            self.last_execution_trace = trace.to_dict()
            if self.last_cache_snapshot is not None:
                self.last_cache_snapshot["execution_trace"] = dict(
                    self.last_execution_trace
                )
                self.last_cache_snapshot["resources_released"] = True
                self.last_cache_snapshot["active_sequence_count_after_release"] = (
                    allocator.active_sequence_count
                )


class PagedHybridBatchRunner(PagedAttentionBatchRunner):
    """Use optimized dense prefill, then decode directly from paged KV blocks."""

    def __init__(self, model: Any, **kwargs: Any) -> None:
        super().__init__(model, prefill_backend="sdpa", **kwargs)


def write_paged_result(
    result: BenchmarkResult,
    path: Path,
    *,
    model_metadata: Mapping[str, Any],
    correctness: Mapping[str, Any],
    block_size: int,
    num_blocks: int,
    cache_snapshot: Mapping[str, Any] | None,
) -> None:
    """Write common harness data plus paged allocation evidence."""

    payload = result.to_dict()
    payload["paged_kv"] = {
        "model": dict(model_metadata),
        "correctness": dict(correctness),
        "block_size": int(block_size),
        "num_blocks": int(num_blocks),
        "cache_snapshot": None if cache_snapshot is None else dict(cache_snapshot),
        "generation": (
            "paged physical storage read directly by a block-wise attention "
            "routine"
            if cache_snapshot is not None
            and cache_snapshot.get("direct_paged_attention")
            else "paged physical storage with dense gather before model calls; "
            "new cache suffix copied back after each decode step"
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
