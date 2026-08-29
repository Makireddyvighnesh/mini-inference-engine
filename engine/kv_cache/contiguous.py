"""Capacity-checked contiguous KV-cache lifecycle management.

The model-compatible backing tensors come from Transformers ``DynamicCache``.
MiniLLM owns when that cache is created, which request rows own it, how much
capacity it may use, its logical position, its accounting, reset, and release.
Paged physical allocation is intentionally deferred to the paged-cache work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch


class KvCacheStateError(RuntimeError):
    """Raised when cache ownership or lifecycle invariants are violated."""


class KvCacheCapacityError(RuntimeError):
    """Raised before a model call would exceed the configured capacity."""


BackendFactory = Callable[[Any, int], Any]


def _default_backend_factory(model_config: Any, capacity_tokens: int) -> Any:
    del capacity_tokens
    from transformers import DynamicCache

    return DynamicCache(config=model_config)


def _as_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().to(device="cpu").item())
    return int(value)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


@dataclass(frozen=True)
class LayerKvSnapshot:
    layer_index: int
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    dtype: str
    device: str
    allocated_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "key_shape": list(self.key_shape),
            "value_shape": list(self.value_shape),
            "dtype": self.dtype,
            "device": self.device,
            "allocated_bytes": self.allocated_bytes,
        }


class ContiguousKvCache:
    """Own one growable contiguous cache with a strict logical capacity."""

    def __init__(
        self,
        model_config: Any,
        *,
        owner_ids: Sequence[str],
        capacity_tokens: int,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        normalized_owners = tuple(str(owner_id) for owner_id in owner_ids)
        if not normalized_owners or any(not owner_id for owner_id in normalized_owners):
            raise ValueError("owner_ids must contain non-empty request IDs")
        if len(set(normalized_owners)) != len(normalized_owners):
            raise ValueError("owner_ids must be unique")
        if capacity_tokens < 1:
            raise ValueError("capacity_tokens must be positive")

        max_positions = getattr(model_config, "max_position_embeddings", None)
        if max_positions is not None and capacity_tokens > int(max_positions):
            raise KvCacheCapacityError(
                f"capacity {capacity_tokens} exceeds model context {max_positions}"
            )

        self._model_config = model_config
        self._owner_ids = normalized_owners
        self._capacity_tokens = int(capacity_tokens)
        self._backend_factory = backend_factory or _default_backend_factory
        self._backend = self._backend_factory(
            self._model_config,
            self._capacity_tokens,
        )
        self._position = 0
        self._lifecycle = "active"

    @property
    def owner_ids(self) -> tuple[str, ...]:
        return self._owner_ids

    @property
    def batch_size(self) -> int:
        return len(self._owner_ids)

    @property
    def capacity_tokens(self) -> int:
        """Maximum cached positions per batch row."""

        return self._capacity_tokens

    @property
    def position(self) -> int:
        """Number of valid cached positions in every batch row."""

        return self._position

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    @property
    def backend_cache(self) -> Any:
        self._require_active()
        return self._backend

    def _require_active(self) -> None:
        if self._lifecycle != "active" or self._backend is None:
            raise KvCacheStateError("KV cache has been released")

    def assert_owners(self, owner_ids: Sequence[str]) -> None:
        self._require_active()
        requested = tuple(str(owner_id) for owner_id in owner_ids)
        if requested != self._owner_ids:
            raise KvCacheStateError(
                "KV cache owner mismatch: "
                f"expected {self._owner_ids}, received {requested}"
            )

    def prepare_append(self, token_count: int) -> None:
        """Reject an append before it can overrun the backing allocation."""

        self._require_active()
        if token_count < 1:
            raise ValueError("token_count must be positive")
        required = self._position + int(token_count)
        if required > self._capacity_tokens:
            raise KvCacheCapacityError(
                f"KV cache requires {required} positions but capacity is "
                f"{self._capacity_tokens}"
            )

    def commit_append(self, token_count: int, returned_cache: Any) -> None:
        """Commit model-written positions and enforce backing-cache identity."""

        self._require_active()
        self.prepare_append(token_count)
        if returned_cache is not self._backend:
            raise KvCacheStateError(
                "model replaced the owned KV cache instead of updating it in place"
            )
        self._position += int(token_count)

    def validate_backend_position(self) -> None:
        """Synchronize once and verify logical versus backing-cache position."""

        self._require_active()
        getter = getattr(self._backend, "get_seq_length", None)
        if not callable(getter):
            raise KvCacheStateError("backing cache does not report sequence length")
        actual = _as_int(getter())
        if actual != self._position:
            raise KvCacheStateError(
                f"KV position mismatch: manager={self._position}, backend={actual}"
            )

    def _layer_snapshots(self) -> tuple[LayerKvSnapshot, ...]:
        self._require_active()
        snapshots: list[LayerKvSnapshot] = []
        layers = getattr(self._backend, "layers", None)
        if layers is None:
            raise KvCacheStateError("backing cache does not expose layers")
        for layer_index, layer in enumerate(layers):
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
                continue
            snapshots.append(
                LayerKvSnapshot(
                    layer_index=layer_index,
                    key_shape=tuple(int(value) for value in keys.shape),
                    value_shape=tuple(int(value) for value in values.shape),
                    dtype=str(keys.dtype),
                    device=str(keys.device),
                    allocated_bytes=_tensor_bytes(keys) + _tensor_bytes(values),
                )
            )
        return tuple(snapshots)

    def snapshot(
        self,
        *,
        free_device_bytes_before: int | None = None,
        total_device_bytes: int | None = None,
        observed_allocated_bytes_before: int | None = None,
        observed_allocated_bytes_after: int | None = None,
    ) -> dict[str, Any]:
        """Return JSON-safe layer shapes, allocation, capacity, and utilization."""

        self.validate_backend_position()
        layers = self._layer_snapshots()
        allocated_bytes = sum(layer.allocated_bytes for layer in layers)
        aggregate_capacity_slots = self.batch_size * self._capacity_tokens
        aggregate_used_slots = self.batch_size * self._position
        physical_tokens = 0
        if layers and len(layers[0].key_shape) >= 3:
            physical_tokens = layers[0].key_shape[-2]
        aggregate_physical_slots = self.batch_size * physical_tokens
        bytes_per_token = (
            allocated_bytes / aggregate_physical_slots
            if allocated_bytes and aggregate_physical_slots
            else None
        )
        estimated_additional_slots = None
        estimated_total_slots = None
        if free_device_bytes_before is not None and bytes_per_token:
            estimated_additional_slots = int(
                free_device_bytes_before // bytes_per_token
            )
            estimated_total_slots = aggregate_used_slots + estimated_additional_slots
        observed_incremental_bytes = None
        accounting_overhead_bytes = None
        if (
            observed_allocated_bytes_before is not None
            and observed_allocated_bytes_after is not None
        ):
            observed_incremental_bytes = (
                observed_allocated_bytes_after - observed_allocated_bytes_before
            )
            accounting_overhead_bytes = observed_incremental_bytes - allocated_bytes

        return {
            "backend": type(self._backend).__name__,
            "layout": "contiguous_growable",
            "allocation_policy": "append_and_reallocate",
            "lifecycle": self._lifecycle,
            "owner_ids": list(self._owner_ids),
            "batch_size": self.batch_size,
            "layer_count": len(layers),
            "position_tokens_per_sequence": self._position,
            "capacity_tokens_per_sequence": self._capacity_tokens,
            "aggregate_used_token_slots": aggregate_used_slots,
            "aggregate_capacity_token_slots": aggregate_capacity_slots,
            "physical_capacity_tokens_per_sequence": physical_tokens,
            "aggregate_physical_token_slots": aggregate_physical_slots,
            "utilization": (
                aggregate_used_slots / aggregate_capacity_slots
                if aggregate_capacity_slots
                else 0.0
            ),
            "allocated_bytes": allocated_bytes,
            "observed_allocated_bytes_before": observed_allocated_bytes_before,
            "observed_allocated_bytes_after": observed_allocated_bytes_after,
            "observed_incremental_allocated_bytes": observed_incremental_bytes,
            "observed_minus_tensor_bytes": accounting_overhead_bytes,
            "used_bytes_estimate": allocated_bytes,
            "unused_physical_capacity_bytes": max(
                0,
                int(
                    allocated_bytes
                    * (aggregate_physical_slots - aggregate_used_slots)
                    / aggregate_physical_slots
                ),
            ) if aggregate_physical_slots else 0,
            "logical_capacity_bytes_estimate": (
                int(bytes_per_token * aggregate_capacity_slots)
                if bytes_per_token
                else None
            ),
            "bytes_per_sequence_token": bytes_per_token,
            "free_device_bytes_before": free_device_bytes_before,
            "total_device_bytes": total_device_bytes,
            "estimated_additional_token_slots_from_free_memory": estimated_additional_slots,
            "estimated_total_token_slots_from_free_memory": estimated_total_slots,
            "estimate_warning": (
                "Free-memory capacity is theoretical and excludes future model "
                "workspace, allocator fragmentation, and safety margin."
                if free_device_bytes_before is not None
                else None
            ),
            "layers": [layer.to_dict() for layer in layers],
        }

    def reset(self) -> None:
        """Drop all tensors and create a fresh empty cache for the same owners."""

        self._require_active()
        self._backend = self._backend_factory(
            self._model_config,
            self._capacity_tokens,
        )
        self._position = 0
        self.validate_backend_position()

    def release(self) -> None:
        """Drop ownership and the final reference to all backing tensors."""

        if self._lifecycle == "released":
            return
        self._backend = None
        self._position = 0
        self._lifecycle = "released"
