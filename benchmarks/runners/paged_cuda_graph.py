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
from minillm_l4.engine.generation.manual import output_token_digest
from minillm_l4.engine.kv_cache import PagedKvAllocator, PagedKvCache
from minillm_l4.engine.kv_cache.qwen3_paged import install_paged_qwen3_attention


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
    output_token: torch.Tensor
    position_ids: torch.Tensor
    sequence_lengths: torch.Tensor


class PagedCudaGraphBatchRunner:
    """Capture one exact paged decode graph for each logical position.

    Every position has identical tensor shapes but writes a different physical
    KV slot. Capturing per position keeps those writes explicit and avoids
    dynamic CPU address calculations during replay. The graphs and page
    tensors are retained across benchmark repetitions.
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
    ) -> None:
        if block_size < 1 or num_blocks < 1:
            raise ValueError("block_size and num_blocks must be positive")
        if logits_mode != "last":
            raise ValueError("paged CUDA Graph decode requires logits_mode='last'")
        if decode_backend not in {"auto", "triton"}:
            raise ValueError("paged CUDA Graph decode requires Triton attention")
        self.model = model
        self.block_size = int(block_size)
        self.num_blocks = int(num_blocks)
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("paged CUDA Graph decode requires CUDA")
        self.decode_backend = decode_backend
        self._shape: tuple[int, int, int] | None = None
        self._owner_ids: tuple[str, ...] = ()
        self._allocator: PagedKvAllocator | None = None
        self._cache: PagedKvCache | None = None
        self._block_tables: torch.Tensor | None = None
        self._graphs: list[_DecodeGraph] = []
        self._capture_time_ms = 0.0
        self.last_cache_snapshot: dict[str, Any] | None = None
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

    def _copy_prefill(self, dense_cache: Any, prompt_tokens: int) -> None:
        assert self._cache is not None
        layers = _dense_layers(dense_cache)
        for row, owner_id in enumerate(self._owner_ids):
            for layer_index, (keys, values) in enumerate(layers):
                self._cache.write_layer_segment(
                    owner_id,
                    layer_index,
                    keys[row : row + 1, :, :prompt_tokens, :],
                    values[row : row + 1, :, :prompt_tokens, :],
                    start_token=0,
                )

    def _decode_call(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        sequence_lengths: torch.Tensor,
        starts: tuple[int, ...],
    ) -> torch.Tensor:
        assert self._cache is not None and self._block_tables is not None
        output = self.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
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
        )
        return _next_token(output)

    def _warm_decode(self, token: torch.Tensor, prompt_tokens: int) -> None:
        batch_size = int(token.shape[0])
        positions = torch.full(
            (batch_size, 1), prompt_tokens, dtype=torch.long, device=self.device
        )
        lengths = torch.full(
            (batch_size,), prompt_tokens + 1, dtype=torch.int32, device=self.device
        )
        self._decode_call(
            token, positions, lengths, (prompt_tokens,) * batch_size
        )
        torch.cuda.synchronize(self.device)

    def _capture_graphs(
        self,
        first_token: torch.Tensor,
        *,
        prompt_tokens: int,
        output_tokens: int,
    ) -> list[torch.Tensor]:
        batch_size = int(first_token.shape[0])
        self._warm_decode(first_token, prompt_tokens)
        generated: list[torch.Tensor] = []
        next_token = first_token
        capture_started = perf_counter()
        for step in range(output_tokens - 1):
            logical_position = prompt_tokens + step
            static_input = torch.empty_like(next_token)
            static_input.copy_(next_token)
            positions = torch.full(
                (batch_size, 1),
                logical_position,
                dtype=torch.long,
                device=self.device,
            )
            lengths = torch.full(
                (batch_size,),
                logical_position + 1,
                dtype=torch.int32,
                device=self.device,
            )
            graph = torch.cuda.CUDAGraph()
            starts = (logical_position,) * batch_size
            with torch.inference_mode(), torch.cuda.graph(graph):
                output_token = self._decode_call(
                    static_input, positions, lengths, starts
                )
            self._graphs.append(
                _DecodeGraph(
                    graph=graph,
                    input_ids=static_input,
                    output_token=output_token,
                    position_ids=positions,
                    sequence_lengths=lengths,
                )
            )
            next_token = output_token
            generated.append(output_token.clone())
        torch.cuda.synchronize(self.device)
        self._capture_time_ms = (perf_counter() - capture_started) * 1000.0
        return generated

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
        shape = (batch_size, prompt_tokens, output_tokens)
        owner_ids = tuple(request.request_id for request in requests)
        if self._shape is not None and (
            shape != self._shape or owner_ids != self._owner_ids
        ):
            raise ValueError("runner graph shape and request IDs must remain fixed")

        for recorder in recorders:
            recorder.record("prefill_start", metadata={"runner": self.runner_name})
        input_ids = torch.tensor(
            [request.prompt_token_ids for request in requests],
            dtype=torch.long,
            device=self.device,
        )
        positions = torch.arange(prompt_tokens, device=self.device).unsqueeze(0).expand(
            batch_size, -1
        )
        with torch.inference_mode():
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
                self._initialize_cache(
                    dense_cache,
                    owner_ids,
                    capacity=prompt_tokens + output_tokens - 1,
                )
                self._shape = shape
            self._copy_prefill(dense_cache, prompt_tokens)
            first_token = _next_token(prefill_output)
            generated = [first_token.clone()]
            torch.cuda.synchronize(self.device)
            timestamp_ns = max(recorder.now_ns() for recorder in recorders)
            first_values = first_token.detach().cpu()[:, 0].tolist()
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
                decode_tokens = self._capture_graphs(
                    first_token,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                )
                for token_index, token in enumerate(decode_tokens, start=1):
                    generated.append(token)
                    timestamp_ns = max(recorder.now_ns() for recorder in recorders)
                    values = token.detach().cpu()[:, 0].tolist()
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
            else:
                next_token = first_token
                for token_index, state in enumerate(self._graphs, start=1):
                    state.input_ids.copy_(next_token)
                    state.graph.replay()
                    torch.cuda.synchronize(self.device)
                    next_token = state.output_token
                    generated.append(next_token.clone())
                    timestamp_ns = max(recorder.now_ns() for recorder in recorders)
                    values = next_token.detach().cpu()[:, 0].tolist()
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
            "graph_count": len(self._graphs),
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
