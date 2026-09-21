"""Exact paged-KV decode using reusable CUDA Graphs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import torch

from ..core.harness import BenchmarkResult, RequestEventRecorder
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
from minillm_l4.engine.kv_cache.qwen3_paged import install_paged_qwen3_attention

from .paged_kv import _paged_activation_dtype


def _next_token(output: Any) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise TypeError("model output logits must have shape [batch, sequence, vocab]")
    return logits[:, -1, :].argmax(dim=-1, keepdim=True)


def _dense_layers(cache: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    output: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer in getattr(cache, "layers", ()):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise TypeError("prefill cache contains an uninitialized layer")
        output.append((keys, values))
    if not output:
        raise TypeError("prefill cache has no layers")
    return tuple(output)


@dataclass
class _DecodeGraph:
    graph: torch.cuda.CUDAGraph
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    output_token: torch.Tensor
    position_ids: torch.Tensor
    sequence_lengths: torch.Tensor


class PagedCudaGraphBatchRunner:
    """Capture one exact paged decode graph for a fixed active batch.

    Every decode position has the same tensor shapes while the logical
    position and sequence length change. Those values are updated in static
    device buffers before replay, and the graph's dynamic page writer uses
    them to select the physical KV slot. The graph and page tensors are
    retained across benchmark repetitions.
    """

    def __init__(
        self,
        model: Any,
        *,
        block_size: int,
        num_blocks: int,
        device: str | torch.device = "cuda:0",
        logits_mode: str = "last",
        decode_backend: str = "auto",
        prefill_backend: str = "packed",
        trace_enabled: bool = False,
    ) -> None:
        if block_size < 1 or num_blocks < 1:
            raise ValueError("block_size and num_blocks must be positive")
        if logits_mode != "last":
            raise ValueError("paged CUDA Graph decode requires logits_mode='last'")
        if decode_backend not in {"auto", "triton"}:
            raise ValueError("paged CUDA Graph decode requires Triton attention")
        if prefill_backend not in {"auto", "packed", "dense"}:
            raise ValueError(
                "graph prefill_backend must be 'auto', 'packed', or 'dense'"
            )
        self.model = model
        self.block_size = int(block_size)
        self.num_blocks = int(num_blocks)
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("paged CUDA Graph decode requires CUDA")
        self.decode_backend = decode_backend
        self.prefill_backend = prefill_backend
        self.trace_enabled = bool(trace_enabled)
        self._shape: tuple[int, int, int] | None = None
        self._owner_ids: tuple[str, ...] = ()
        self._allocator: PagedKvAllocator | None = None
        self._cache: PagedKvCache | None = None
        self._block_tables: torch.Tensor | None = None
        self._graphs: list[_DecodeGraph] = []
        self._capture_time_ms = 0.0
        self.last_cache_snapshot: dict[str, Any] | None = None
        self.last_execution_trace: dict[str, Any] | None = None
        install_paged_qwen3_attention(model)

    @property
    def runner_name(self) -> str:
        return "paged_kv_cuda_graph"

    def _initialize_cache(
        self,
        dense_cache: Any,
        owner_ids: tuple[str, ...],
        *,
        capacity: int,
    ) -> None:
        layers = _dense_layers(dense_cache)
        keys = layers[0][0]
        allocator = PagedKvAllocator(
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            max_sequence_tokens=capacity,
        )
        cache = PagedKvCache(
            allocator,
            num_layers=len(layers),
            num_kv_heads=int(keys.shape[1]),
            head_dim=int(keys.shape[-1]),
            dtype=keys.dtype,
            device=keys.device,
        )
        for owner_id in owner_ids:
            allocator.allocate(owner_id, token_count=capacity)
        self._allocator = allocator
        self._cache = cache
        self._owner_ids = owner_ids
        self._block_tables = cache.block_table_tensor(
            owner_ids, device=self.device
        ).to(torch.int32)

    def _initialize_model_cache(
        self,
        owner_ids: tuple[str, ...],
        *,
        capacity: int,
    ) -> None:
        """Allocate page storage before packed prefill writes layer K/V."""

        config = getattr(self.model, "config", None)
        if config is None:
            raise ValueError("graph prefill requires model.config")
        head_dim = int(
            getattr(
                config,
                "head_dim",
                int(config.hidden_size) // int(config.num_attention_heads),
            )
        )
        allocator = PagedKvAllocator(
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            max_sequence_tokens=capacity,
        )
        cache = PagedKvCache(
            allocator,
            num_layers=int(config.num_hidden_layers),
            num_kv_heads=int(config.num_key_value_heads),
            head_dim=head_dim,
            dtype=_paged_activation_dtype(self.model),
            device=self.device,
        )
        for owner_id in owner_ids:
            allocator.allocate(owner_id, token_count=capacity)
        self._allocator = allocator
        self._cache = cache
        self._owner_ids = owner_ids
        self._block_tables = cache.block_table_tensor(
            owner_ids, device=self.device
        ).to(torch.int32)

    def _copy_prefill(self, dense_cache: Any, prompt_tokens: int) -> None:
        assert self._cache is not None and self._block_tables is not None
        layers = _dense_layers(dense_cache)
        self._cache.write_dense_batch(
            self._owner_ids,
            tuple(
                (
                    keys[:, :, :prompt_tokens, :],
                    values[:, :, :prompt_tokens, :],
                )
                for keys, values in layers
            ),
            start_token=0,
            block_tables=self._block_tables,
        )

    def _resolve_prefill_backend(self) -> str:
        """Choose the fastest safe path for this fixed-shape graph runner."""

        if self.prefill_backend == "auto":
            # Graph batches currently require equal prompt lengths. Dense SDPA
            # therefore needs no padding and is substantially faster than the
            # educational flat-token page-walk prefill. Keep ``packed``
            # available as an explicit path for implementation comparisons.
            return "dense"
        return self.prefill_backend

    def _decode_call(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        sequence_lengths: torch.Tensor,
        starts: tuple[int, ...],
        attention_mask: torch.Tensor | None = None,
        decode_max_sequence_length: int | None = None,
    ) -> torch.Tensor:
        assert self._cache is not None and self._block_tables is not None
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
            paged_kv_cache=self._cache,
            paged_sequence_ids=self._owner_ids,
            paged_query_start_positions=starts,
            paged_block_tables=self._block_tables,
            paged_sequence_lengths=sequence_lengths,
            paged_attention_backend=self.decode_backend,
            paged_decode_split_count=(
                1
                if decode_max_sequence_length is None
                else select_decode_split_count(
                    decode_max_sequence_length,
                    batch_size=int(input_ids.shape[0]),
                )
            ),
            paged_decode_max_sequence_length=decode_max_sequence_length,
            paged_decode_block_tokens=(
                16
                if decode_max_sequence_length is None
                else select_decode_block_tokens(decode_max_sequence_length)
            ),
            paged_decode_use_gqa_reuse=False,
        )
        return _next_token(output)

    def _warm_decode(
        self,
        token: torch.Tensor,
        prompt_tokens: int,
        max_sequence_length: int,
    ) -> None:
        batch_size = int(token.shape[0])
        positions = torch.full(
            (batch_size, 1), prompt_tokens, dtype=torch.long, device=self.device
        )
        lengths = torch.full(
            (batch_size,), prompt_tokens + 1, dtype=torch.int32, device=self.device
        )
        self._decode_call(
            token,
            positions,
            lengths,
            (prompt_tokens,) * batch_size,
            decode_max_sequence_length=max_sequence_length,
        )
        torch.cuda.synchronize(self.device)

    def _capture_graph(
        self,
        first_token: torch.Tensor,
        *,
        prompt_tokens: int,
        max_sequence_length: int,
    ) -> None:
        """Capture one reusable one-token decode graph.

        Decode has a fixed tensor shape for a given active batch, but the
        logical position and sequence length change after every replay.  Those
        values live in static device buffers and are updated by the host before
        replay.  The paged KV writer and attention kernel read the current
        buffer values, so a single graph can walk the sequence without
        allocating one graph per output position.
        """
        batch_size = int(first_token.shape[0])
        self._warm_decode(first_token, prompt_tokens, max_sequence_length)
        capture_started = perf_counter()
        static_input = torch.empty_like(first_token)
        static_input.copy_(first_token)
        positions = torch.full(
            (batch_size, 1),
            prompt_tokens,
            dtype=torch.long,
            device=self.device,
        )
        lengths = torch.full(
            (batch_size,),
            prompt_tokens + 1,
            dtype=torch.int32,
            device=self.device,
        )
        attention_mask = torch.ones_like(static_input)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            output_token = self._decode_call(
                static_input,
                positions,
                lengths,
                (prompt_tokens,) * batch_size,
                attention_mask=attention_mask,
                decode_max_sequence_length=max_sequence_length,
            )
        self._graphs.append(
            _DecodeGraph(
                graph=graph,
                input_ids=static_input,
                attention_mask=attention_mask,
                output_token=output_token,
                position_ids=positions,
                sequence_lengths=lengths,
            )
        )
        torch.cuda.synchronize(self.device)
        # A graph can capture successfully before its first replay has paid
        # CUDA's lazy module/loading cost.  Replay once here, while the
        # runner is still in the warm-up request, so the first measured
        # decode step is steady state.  The replay uses the same first token
        # and logical position as the capture and is therefore idempotent for
        # the cache slot it writes.
        graph.replay()
        torch.cuda.synchronize(self.device)
        self._capture_time_ms = (perf_counter() - capture_started) * 1000.0

    def __call__(
        self,
        requests: Sequence[RequestSpec],
        recorders: Sequence[RequestEventRecorder],
    ) -> tuple[RequestOutcome, ...]:
        if not requests or len(requests) != len(recorders):
            raise ValueError("requests and recorders must be non-empty and equal")
        prompt_lengths = {request.prompt_tokens for request in requests}
        output_lengths = {request.max_new_tokens for request in requests}
        if len(prompt_lengths) != 1 or len(output_lengths) != 1:
            raise ValueError("paged graph batches require equal prompt/output lengths")
        batch_size = len(requests)
        prompt_tokens = next(iter(prompt_lengths))
        output_tokens = next(iter(output_lengths))
        resolved_prefill_backend = self._resolve_prefill_backend()
        max_sequence_length = prompt_tokens + output_tokens - 1
        decode_split_count = select_decode_split_count(
            max_sequence_length,
            batch_size=batch_size,
        )
        decode_block_tokens = select_decode_block_tokens(max_sequence_length)
        shape = (batch_size, prompt_tokens, output_tokens)
        owner_ids = tuple(request.request_id for request in requests)
        if self._shape is not None and (
            shape != self._shape or owner_ids != self._owner_ids
        ):
            raise ValueError("runner graph shape and request IDs must remain fixed")

        trace = ExecutionTrace(
            device=self.device,
            started_ns=recorders[0].run_started_ns,
            enabled=self.trace_enabled,
        )
        self.last_execution_trace = None
        for recorder in recorders:
            recorder.record("prefill_start", metadata={"runner": self.runner_name})
        with trace.span(
            "input_buffer_preparation_and_device_transfer",
            category="request_preparation",
            metadata={
                "input_shape": [batch_size, prompt_tokens],
                "compute_layout": (
                    "flat_packed_tokens"
                    if resolved_prefill_backend == "packed"
                    else "rectangular_dense"
                ),
            },
        ):
            input_ids = torch.tensor(
                [request.prompt_token_ids for request in requests],
                dtype=torch.long,
                device=self.device,
            )
            positions = torch.arange(prompt_tokens, device=self.device).unsqueeze(0).expand(
                batch_size, -1
            )
        with torch.inference_mode():
            if resolved_prefill_backend == "packed":
                if self._cache is None:
                    with trace.span(
                        "kv_allocator_and_physical_cache_allocation",
                        category="kv_memory",
                        metadata={
                            "capacity_tokens": prompt_tokens + output_tokens - 1,
                            "block_size": self.block_size,
                        },
                    ):
                        self._initialize_model_cache(
                            owner_ids,
                            capacity=prompt_tokens + output_tokens - 1,
                        )
                    self._shape = shape
                assert self._cache is not None and self._block_tables is not None
                packed_metadata = PackedSequenceMetadata.from_lengths(
                    owner_ids,
                    (prompt_tokens,) * batch_size,
                    device=self.device,
                )
                with trace.span(
                    "prefill_model_forward",
                    category="model_execution",
                    gpu=True,
                    metadata={
                        "input_shape": [packed_metadata.total_tokens],
                        "input_layout": "flat_packed_tokens",
                        "backend": "packed_triton",
                    },
                ):
                    prefill_output = qwen3_packed_prefill(
                        self.model,
                        input_ids.reshape(-1),
                        packed_metadata,
                        paged_kv_cache=self._cache,
                        block_tables=self._block_tables,
                        attention_backend="triton",
                    )
                with trace.span(
                    "first_token_selection",
                    category="sampling",
                    gpu=True,
                    metadata={"algorithm": "greedy_argmax"},
                ):
                    first_token = prefill_output.logits.argmax(
                        dim=-1,
                        keepdim=True,
                    )
            else:
                with trace.span(
                    "prefill_model_forward",
                    category="model_execution",
                    gpu=True,
                    metadata={
                        "input_shape": [batch_size, prompt_tokens],
                        "input_layout": "rectangular_dense",
                        "backend": "dense_transformers",
                    },
                ):
                    prefill_output = self.model(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        position_ids=positions,
                        use_cache=True,
                        return_dict=True,
                        logits_to_keep=1,
                    )
                dense_cache = getattr(prefill_output, "past_key_values", None)
                if dense_cache is None:
                    raise RuntimeError("model did not return prefill KV")
                if self._cache is None:
                    with trace.span(
                        "kv_allocator_and_physical_cache_allocation",
                        category="kv_memory",
                        metadata={
                            "capacity_tokens": prompt_tokens + output_tokens - 1,
                            "block_size": self.block_size,
                        },
                    ):
                        self._initialize_cache(
                            dense_cache,
                            owner_ids,
                            capacity=prompt_tokens + output_tokens - 1,
                        )
                    self._shape = shape
                assert self._cache is not None
                with trace.span(
                    "prefill_cache_materialization",
                    category="kv_memory",
                    metadata={"prompt_tokens": prompt_tokens},
                ):
                    self._copy_prefill(dense_cache, prompt_tokens)
                with trace.span(
                    "first_token_selection",
                    category="sampling",
                    gpu=True,
                    metadata={"algorithm": "greedy_argmax"},
                ):
                    first_token = _next_token(prefill_output)
            generated = [first_token.clone()]
            with trace.span(
                "first_token_host_transfer_and_stream_emit",
                category="streaming",
                metadata={"token_index": 0},
            ):
                first_values = first_token.detach().cpu()[:, 0].tolist()
                timestamp_ns = max(recorder.now_ns() for recorder in recorders)
                for recorder in recorders:
                    recorder.record("prefill_end", timestamp_ns=timestamp_ns)
                for recorder, value in zip(recorders, first_values, strict=True):
                    recorder.mark_token_ready(
                        0, token_id=int(value), timestamp_ns=timestamp_ns
                    )
                    recorder.mark_token_sent(
                        0, token_id=int(value), timestamp_ns=timestamp_ns
                    )

            if not self._graphs and output_tokens > 1:
                with trace.span(
                    "cuda_graph_warmup_and_capture",
                    category="cuda_graph",
                    metadata={"graph_count": 1},
                ):
                    self._capture_graph(
                        first_token,
                        prompt_tokens=prompt_tokens,
                        max_sequence_length=max_sequence_length,
                    )

            # A graph is captured for the first decode position.  The same
            # graph is replayed at every later position after updating its
            # static device inputs.  The first replay also handles the branch
            # that just captured the graph.
            state = self._graphs[0] if self._graphs else None
            if output_tokens > 1:
                if state is None:
                    raise RuntimeError("decode graph was not initialized")
                next_token = first_token
                for token_index in range(1, output_tokens):
                    logical_position = prompt_tokens + token_index - 1
                    with trace.span(
                        "cuda_graph_input_update",
                        category="cuda_graph",
                        metadata={"token_index": token_index},
                    ):
                        state.input_ids.copy_(next_token)
                        state.position_ids.fill_(logical_position)
                        state.sequence_lengths.fill_(logical_position + 1)
                    with trace.span(
                        "cuda_graph_replay",
                        category="model_execution",
                        gpu=True,
                        metadata={
                            "token_index": token_index,
                            "graph_count": len(self._graphs),
                            "logical_position": logical_position,
                        },
                    ):
                        state.graph.replay()
                    next_token = state.output_token
                    generated.append(next_token.clone())
                    with trace.span(
                        "replayed_token_host_transfer_and_stream_emit",
                        category="streaming",
                        metadata={"token_index": token_index},
                    ):
                        values = next_token.detach().cpu()[:, 0].tolist()
                        timestamp_ns = max(recorder.now_ns() for recorder in recorders)
                        for recorder, value in zip(recorders, values, strict=True):
                            recorder.mark_token_ready(
                                token_index,
                                token_id=int(value),
                                timestamp_ns=timestamp_ns,
                            )
                            recorder.mark_token_sent(
                                token_index,
                                token_id=int(value),
                                timestamp_ns=timestamp_ns,
                            )

        with trace.span("output_materialization", category="response_finalization"):
            token_matrix = torch.cat(generated, dim=1).cpu()
        assert self._cache is not None and self._allocator is not None
        self.last_cache_snapshot = {
            **self._allocator.snapshot(),
            "runner": self.runner_name,
            "physical_kv_bytes": int(
                (self._cache.key_blocks.numel() + self._cache.value_blocks.numel())
                * self._cache.key_blocks.element_size()
            ),
            "cuda_graph": True,
            "prefill_backend": resolved_prefill_backend,
            "prefill_backend_requested": self.prefill_backend,
            "prefill_packed": resolved_prefill_backend == "packed",
            "prefill_input_tokens": batch_size * prompt_tokens,
            "prefill_padded_equivalent_tokens": batch_size * prompt_tokens,
            "prefill_padding_tokens": 0,
            "graph_count": len(self._graphs),
            "decode_split_count": decode_split_count,
            "decode_block_tokens": decode_block_tokens,
            "decode_gqa_reuse": False,
            "decode_max_sequence_length": max_sequence_length,
            "capture_time_ms": self._capture_time_ms,
            "capture_cost_in_steady_state_metrics": False,
            "allocation_policy": "full_request_reservation_for_fixed_graph_addresses",
        }
        for recorder in recorders:
            recorder.record("completion")
        rows = tuple(
            tuple(int(value) for value in token_matrix[row].tolist())
            for row in range(batch_size)
        )
        trace.counter("batch_size", batch_size)
        trace.counter("prompt_tokens_per_request", prompt_tokens)
        trace.counter("output_tokens_per_request", output_tokens)
        trace.counter("prefill_backend", resolved_prefill_backend)
        trace.counter("prefill_backend_requested", self.prefill_backend)
        trace.counter("graph_count", len(self._graphs))
        trace.counter("decode_split_count", decode_split_count)
        trace.counter("decode_block_tokens", decode_block_tokens)
        trace.counter("decode_gqa_reuse", False)
        trace.counter("decode_max_sequence_length", max_sequence_length)
        trace.counter("capture_time_ms", self._capture_time_ms)
        trace.counter("steady_state_replay", bool(self._graphs))
        self.last_execution_trace = trace.to_dict()
        self.last_cache_snapshot["execution_trace"] = dict(self.last_execution_trace)
        return tuple(
            RequestOutcome(
                status="completed",
                generated_token_ids=row,
                metadata={
                    "runner": self.runner_name,
                    "cache": dict(self.last_cache_snapshot),
                    "output_token_sha256": output_token_digest(
                        torch.tensor([row], dtype=torch.long)
                    ),
                },
            )
            for row in rows
        )



def write_paged_cuda_graph_result(
    result: BenchmarkResult,
    path: Path,
    *,
    model_metadata: Mapping[str, Any],
    correctness: Mapping[str, Any],
    cache_snapshot: Mapping[str, Any] | None,
) -> None:
    payload = result.to_dict()
    payload["paged_cuda_graph"] = {
        "model": dict(model_metadata),
        "correctness": dict(correctness),
        "cache_snapshot": (
            None if cache_snapshot is None else dict(cache_snapshot)
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = ["PagedCudaGraphBatchRunner", "write_paged_cuda_graph_result"]
