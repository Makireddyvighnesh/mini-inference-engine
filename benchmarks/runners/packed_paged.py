"""Packed-prefill plus direct paged-KV generation.

This runner is the variable-length prefill path.  It flattens prompts into
one token stream, runs all decoder layers once over that stream, and uses
request-boundary metadata inside page-aware attention.  Decode remains an
iteration-level batch of one token per active request.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from ..core.harness import RequestEventRecorder
from ..core.schemas import RequestOutcome, RequestSpec
from ..core.tracing import ExecutionTrace
from minillm_l4.engine.generation.manual import output_token_digest
from minillm_l4.engine.kv_cache import (
    PackedSequenceMetadata,
    PagedKvAllocator,
    PagedKvCache,
    select_decode_block_tokens,
    select_decode_split_count,
)
from minillm_l4.engine.model_runner.qwen3_packed import qwen3_packed_prefill

from .paged_kv import (
    _elapsed_ms,
    _first_model_device,
    _paged_activation_dtype,
    _select_next_token,
    _synchronize,
    _token_values,
)
from minillm_l4.engine.kv_cache.qwen3_paged import install_paged_qwen3_attention
from minillm_l4.engine.kv_cache.triton_paged_attention import triton_is_available


class PackedPagedPrefillBatchRunner:
    """Run variable-length prompts without padding during prefill.

    The first generated token is selected from the final token of each packed
    prompt.  For the initial implementation, all requests in one benchmark
    batch use the same output limit; prompt lengths may differ.  This keeps the
    comparison focused on ragged prefill while preserving the existing fixed
    length benchmark contract.
    """

    def __init__(
        self,
        model: Any,
        *,
        block_size: int,
        num_blocks: int,
        device: str | torch.device | None = None,
        logits_mode: str = "last",
        kv_dtype: torch.dtype | None = None,
        prefill_backend: str = "auto",
        decode_backend: str = "auto",
        decode_sdpa_compat: bool | None = None,
        trace_enabled: bool = False,
    ) -> None:
        if int(block_size) < 1:
            raise ValueError("block_size must be positive")
        if int(num_blocks) < 1:
            raise ValueError("num_blocks must be positive")
        if logits_mode != "last":
            raise ValueError("packed prefill requires logits_mode='last'")
        if kv_dtype is not None and not isinstance(kv_dtype, torch.dtype):
            raise TypeError("kv_dtype must be a torch.dtype when supplied")
        if prefill_backend not in {
            "auto",
            "torch",
            "triton",
            "sdpa",
            "sdpa_math",
        }:
            raise ValueError(
                "prefill_backend must be auto, torch, triton, sdpa, or sdpa_math"
            )
        if decode_backend not in {"auto", "torch", "triton"}:
            raise ValueError("decode_backend must be 'auto', 'torch', or 'triton'")
        self.model = model
        self.block_size = int(block_size)
        self.num_blocks = int(num_blocks)
        self.device = (
            torch.device(device) if device is not None else _first_model_device(model)
        )
        self.kv_dtype = kv_dtype
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend
        self.decode_sdpa_compat = bool(decode_sdpa_compat)
        self.trace_enabled = bool(trace_enabled)
        self.last_cache_snapshot: dict[str, Any] | None = None
        self.last_execution_trace: dict[str, Any] | None = None
        install_paged_qwen3_attention(model)

    @property
    def runner_name(self) -> str:
        return "paged_kv_packed_prefill_direct_decode"

    def _new_cache(
        self,
        owner_ids: Sequence[str],
        prompt_lengths: Sequence[int],
        output_tokens: int,
    ) -> tuple[PagedKvAllocator, PagedKvCache]:
        config = getattr(self.model, "config", None)
        if config is None:
            raise ValueError("packed paged prefill requires model.config")
        normalized_lengths = tuple(int(value) for value in prompt_lengths)
        max_sequence_tokens = max(
            length + output_tokens - 1 for length in normalized_lengths
        )
        allocator = PagedKvAllocator(
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            max_sequence_tokens=max_sequence_tokens,
        )
        cache = PagedKvCache(
            allocator,
            num_layers=int(getattr(config, "num_hidden_layers")),
            num_kv_heads=int(getattr(config, "num_key_value_heads")),
            head_dim=int(
                getattr(
                    config,
                    "head_dim",
                    int(config.hidden_size) // int(config.num_attention_heads),
                )
            ),
            dtype=self.kv_dtype or _paged_activation_dtype(self.model),
            device=self.device,
        )
        for owner_id, prompt_length in zip(owner_ids, normalized_lengths, strict=True):
            allocator.allocate(owner_id, token_count=prompt_length)
        return allocator, cache

    @staticmethod
    def _physical_bytes(cache: PagedKvCache) -> int:
        return int(
            (cache.key_blocks.numel() + cache.value_blocks.numel())
            * cache.key_blocks.element_size()
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

        output_lengths = tuple(int(request.max_new_tokens) for request in requests)
        if len(set(output_lengths)) != 1:
            raise ValueError(
                "packed prefill currently requires equal output lengths within a batch"
            )
        output_tokens = output_lengths[0]
        prompt_lengths = tuple(int(request.prompt_tokens) for request in requests)
        batch_size = len(requests)
        owner_ids = tuple(request.request_id for request in requests)
        trace = ExecutionTrace(
            device=self.device,
            started_ns=recorders[0].run_started_ns,
            enabled=self.trace_enabled,
        )
        self.last_execution_trace = None
        with trace.span(
            "request_validation_and_shape_setup",
            category="request_lifecycle",
            metadata={"batch_size": batch_size},
        ):
            metadata = PackedSequenceMetadata.from_lengths(
                owner_ids,
                prompt_lengths,
                device=self.device,
            )
            decode_max_sequence_length = max(prompt_lengths) + output_tokens - 1
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
            "input_flattening_and_device_transfer",
            category="request_preparation",
            metadata={
                "input_shape": [metadata.total_tokens],
                "prompt_lengths": list(prompt_lengths),
            },
        ):
            flat_input_ids = torch.tensor(
                [token for request in requests for token in request.prompt_token_ids],
                dtype=torch.long,
                device=self.device,
            )
        self.last_cache_snapshot = None

        with trace.span("backend_resolution", category="runner_control"):
            if self.prefill_backend == "sdpa":
                resolved_prefill_backend = "sdpa"
            elif self.prefill_backend == "sdpa_math":
                resolved_prefill_backend = "sdpa_math"
            elif (
                self.prefill_backend != "torch"
                and self.device.type == "cuda"
                and triton_is_available()
            ):
                resolved_prefill_backend = "triton"
            else:
                resolved_prefill_backend = "torch"
            resolved_decode_backend = (
                "triton_paged_decode"
                if self.decode_backend != "torch"
                and self.device.type == "cuda"
                and triton_is_available()
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
                    "prompt_lengths": list(prompt_lengths),
                    "prompt_tokens_total": metadata.total_tokens,
                    "output_tokens": output_tokens,
                    "prefill_backend": resolved_prefill_backend,
                    "decode_backend": resolved_decode_backend,
                    "packed_kv_write_backend": (
                        "triton_scatter"
                        if self.device.type == "cuda" and triton_is_available()
                        else "python_page_copy"
                    ),
                    "packed": True,
                },
            )

        with trace.span(
            "kv_allocator_and_physical_cache_allocation",
            category="kv_memory",
            metadata={
                "num_blocks": self.num_blocks,
                "block_size": self.block_size,
            },
        ):
            allocator, paged_cache = self._new_cache(
                owner_ids,
                prompt_lengths,
                output_tokens,
            )
        with trace.span(
            "block_table_materialization",
            category="kv_memory",
            metadata={
                "shape": [
                    batch_size,
                    max(
                        len(allocator.get_block_table(owner_id).block_ids)
                        for owner_id in owner_ids
                    ),
                ]
            },
        ):
            block_tables = paged_cache.block_table_tensor(
                owner_ids,
                device=self.device,
            ).to(dtype=torch.int32)
        prefill_model_ms = 0.0
        decode_model_ms = 0.0
        generated: list[torch.Tensor] = []
        try:
            with torch.inference_mode():
                with trace.span(
                    "prefill_model_forward",
                    category="model_execution",
                    gpu=True,
                    metadata={
                        "input_shape": [metadata.total_tokens],
                        "logits_shape": [batch_size],
                        "backend": resolved_prefill_backend,
                    },
                ) as measurement:
                    prefill_output = qwen3_packed_prefill(
                        self.model,
                        flat_input_ids,
                        metadata,
                        paged_kv_cache=paged_cache,
                        block_tables=block_tables,
                        attention_backend=resolved_prefill_backend,
                    )
                prefill_model_ms = measurement.wall_ms
                with trace.span(
                    "first_token_selection",
                    category="sampling",
                    gpu=True,
                    metadata={"algorithm": "greedy_argmax", "batch_size": batch_size},
                ):
                    next_token = prefill_output.logits.argmax(
                        dim=-1,
                        keepdim=True,
                    )
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
                        metadata={
                            "decode_step": decode_step + 1,
                            "block_table_shape": [
                                batch_size,
                                block_tables.shape[1],
                            ],
                        },
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
                            "sequence_lengths": list(length + 1 for length in old_lengths),
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
                            logits_to_keep=1,
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
            prefill_page_visits_per_layer = sum(
                sum(
                    math.ceil(token_count / self.block_size)
                    for token_count in range(1, prompt_length + 1)
                )
                for prompt_length in prompt_lengths
            )
            decode_page_visits_per_layer = sum(
                sum(
                    math.ceil((prompt_length + step + 1) / self.block_size)
                    for step in range(output_tokens - 1)
                )
                for prompt_length in prompt_lengths
            )
            allocator_snapshot = allocator.snapshot()
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
                "packed_prefill": True,
                "prefill_backend": resolved_prefill_backend,
                "decode_backend": resolved_decode_backend,
                "decode_split_count": decode_split_count,
                "decode_block_tokens": decode_block_tokens,
                "decode_gqa_reuse": False,
                "decode_max_sequence_length": decode_max_sequence_length,
                "decode_sdpa_compat": self.decode_sdpa_compat,
                "packed_kv_write_backend": (
                    "triton_scatter"
                    if self.device.type == "cuda" and triton_is_available()
                    else "python_page_copy"
                ),
                "attention_backend": (
                    f"packed_{resolved_prefill_backend}_prefill_"
                    f"{resolved_decode_backend}"
                ),
                "prefill_prompt_lengths": list(prompt_lengths),
                "prefill_input_tokens": metadata.total_tokens,
                "prefill_padded_equivalent_tokens": (
                    batch_size * max(prompt_lengths)
                ),
                "prefill_padding_tokens": (
                    batch_size * max(prompt_lengths) - metadata.total_tokens
                ),
                "prefill_metadata": metadata.to_dict(),
                "paged_attention_kernel": resolved_decode_backend
                == "triton_paged_decode",
                "packed_prefill_kernel": resolved_prefill_backend == "triton",
                "stage_timings_ms": {
                    "cache_setup": trace.to_dict()["components"].get(
                        "kv_allocator_and_physical_cache_allocation", {}
                    ).get("wall_ms_total", 0.0),
                    "prefill_model": prefill_model_ms,
                    "prefill_cache_materialization": trace.to_dict()["components"].get(
                        "block_table_materialization", {}
                    ).get("wall_ms_total", 0.0),
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
            return tuple(
                RequestOutcome(
                    status="completed",
                    generated_token_ids=row,
                    metadata={
                        "runner": self.runner_name,
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
                for owner_id in owner_ids:
                    if owner_id in allocator.sequence_ids:
                        paged_cache.release(owner_id)
            trace.counter("generated_tokens_total", batch_size * output_tokens)
            trace.counter("prefill_tokens_total", metadata.total_tokens)
            trace.counter("batch_size", batch_size)
            trace.counter("output_tokens_per_request", output_tokens)
            trace.counter("resolved_prefill_backend", resolved_prefill_backend)
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


__all__ = ["PackedPagedPrefillBatchRunner"]
