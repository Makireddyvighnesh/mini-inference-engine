"""Fixed-block KV storage, block tables, and page-aware access helpers.

This module deliberately separates two concerns:

* :class:`PagedKvAllocator` owns physical block IDs and logical sequence
  ownership.  It never assumes that a request's blocks are contiguous.
* :class:`PagedKvCache` stores per-layer K/V tensors in those blocks and
  exposes both a dense gather adapter and direct block-table access.

The dense gather path is intentionally explicit and copy-based. Direct
attention uses the block table without materializing a dense batch cache;
supported CUDA decode shapes are handled by the fused Triton kernel selected
in :mod:`paged_attention`.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch


class PagedKvError(RuntimeError):
    """Base class for paged KV allocation failures."""


class PagedKvOutOfMemoryError(PagedKvError):
    """Raised when the block pool cannot satisfy an allocation request."""


class PagedKvCapacityError(PagedKvError):
    """Raised when a sequence would exceed its configured context capacity."""


class PagedKvStateError(PagedKvError):
    """Raised when a sequence or block-table lifecycle is invalid."""


class PagedKvShapeError(PagedKvError):
    """Raised when K/V tensors do not match the physical cache layout."""


@dataclass(frozen=True)
class PagedBlockTable:
    """Immutable view of one sequence's logical-to-physical mapping."""

    sequence_id: str
    block_ids: tuple[int, ...]
    token_count: int
    block_size: int

    @property
    def allocated_block_count(self) -> int:
        return len(self.block_ids)

    @property
    def reserved_token_slots(self) -> int:
        return self.allocated_block_count * self.block_size

    @property
    def wasted_token_slots(self) -> int:
        return self.reserved_token_slots - self.token_count

    def physical_location(self, logical_token_index: int) -> tuple[int, int]:
        """Return ``(physical_block_id, slot_in_block)`` for one token."""

        index = int(logical_token_index)
        if index < 0 or index >= self.token_count:
            raise IndexError(
                f"logical token index {index} is outside [0, {self.token_count})"
            )
        block_index, offset = divmod(index, self.block_size)
        return self.block_ids[block_index], offset

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "block_ids": list(self.block_ids),
            "token_count": self.token_count,
            "block_size": self.block_size,
            "allocated_block_count": self.allocated_block_count,
            "reserved_token_slots": self.reserved_token_slots,
            "wasted_token_slots": self.wasted_token_slots,
        }


@dataclass
class _MutableAllocation:
    sequence_id: str
    block_ids: list[int]
    token_count: int


class PagedKvAllocator:
    """Deterministic fixed-size block allocator for variable-length sequences.

    A request owns a logical sequence of tokens.  Its block table maps each
    logical block to an arbitrary physical block from the free pool.  Growing
    a request allocates only the additional blocks required by its new token
    count; releasing a request immediately returns all of its blocks.
    """

    def __init__(
        self,
        *,
        num_blocks: int,
        block_size: int,
        max_sequence_tokens: int | None = None,
    ) -> None:
        if int(num_blocks) < 1:
            raise ValueError("num_blocks must be positive")
        if int(block_size) < 1:
            raise ValueError("block_size must be positive")
        if max_sequence_tokens is not None and int(max_sequence_tokens) < 1:
            raise ValueError("max_sequence_tokens must be positive when supplied")
        if max_sequence_tokens is not None and int(max_sequence_tokens) > (
            int(num_blocks) * int(block_size)
        ):
            raise ValueError("max_sequence_tokens cannot exceed the block pool")

        self._num_blocks = int(num_blocks)
        self._block_size = int(block_size)
        self._max_sequence_tokens = (
            None if max_sequence_tokens is None else int(max_sequence_tokens)
        )
        self._free_blocks = list(range(self._num_blocks))
        heapq.heapify(self._free_blocks)
        self._allocations: dict[str, _MutableAllocation] = {}

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def max_sequence_tokens(self) -> int | None:
        return self._max_sequence_tokens

    @property
    def capacity_tokens(self) -> int:
        return self._num_blocks * self._block_size

    @property
    def free_block_count(self) -> int:
        return len(self._free_blocks)

    @property
    def allocated_block_count(self) -> int:
        return self._num_blocks - self.free_block_count

    @property
    def active_sequence_count(self) -> int:
        return len(self._allocations)

    @property
    def sequence_ids(self) -> tuple[str, ...]:
        return tuple(self._allocations)

    @property
    def free_block_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._free_blocks))

    def _normalize_sequence_id(self, sequence_id: str) -> str:
        normalized = str(sequence_id)
        if not normalized:
            raise ValueError("sequence_id must be non-empty")
        return normalized

    def _validate_token_count(self, token_count: int, *, name: str) -> int:
        normalized = int(token_count)
        if normalized < 0:
            raise ValueError(f"{name} must be non-negative")
        if (
            self._max_sequence_tokens is not None
            and normalized > self._max_sequence_tokens
        ):
            raise PagedKvCapacityError(
                f"{name}={normalized} exceeds max_sequence_tokens="
                f"{self._max_sequence_tokens}"
            )
        return normalized

    def _required_blocks(self, token_count: int) -> int:
        return math.ceil(token_count / self._block_size)

    def _require_sequence(self, sequence_id: str) -> _MutableAllocation:
        normalized = self._normalize_sequence_id(sequence_id)
        try:
            return self._allocations[normalized]
        except KeyError as error:
            raise PagedKvStateError(
                f"sequence {normalized!r} has no active block table"
            ) from error

    def _take_blocks(self, count: int) -> list[int]:
        if count < 0:
            raise ValueError("block count must be non-negative")
        if count > self.free_block_count:
            raise PagedKvOutOfMemoryError(
                f"need {count} additional KV blocks but only "
                f"{self.free_block_count} are free"
            )
        return [heapq.heappop(self._free_blocks) for _ in range(count)]

    def _view(self, allocation: _MutableAllocation) -> PagedBlockTable:
        return PagedBlockTable(
            sequence_id=allocation.sequence_id,
            block_ids=tuple(allocation.block_ids),
            token_count=allocation.token_count,
            block_size=self._block_size,
        )

    def allocate(self, sequence_id: str, token_count: int = 0) -> PagedBlockTable:
        """Create a sequence and reserve blocks for its initial token count."""

        normalized = self._normalize_sequence_id(sequence_id)
        if normalized in self._allocations:
            raise PagedKvStateError(f"sequence {normalized!r} is already allocated")
        normalized_tokens = self._validate_token_count(
            token_count,
            name="token_count",
        )
        block_ids = self._take_blocks(self._required_blocks(normalized_tokens))
        allocation = _MutableAllocation(normalized, block_ids, normalized_tokens)
        self._allocations[normalized] = allocation
        return self._view(allocation)

    def append(self, sequence_id: str, token_count: int) -> PagedBlockTable:
        """Grow a sequence by ``token_count`` logical tokens."""

        if int(token_count) < 1:
            raise ValueError("token_count must be positive")
        allocation = self._require_sequence(sequence_id)
        new_token_count = self._validate_token_count(
            allocation.token_count + int(token_count),
            name="new token count",
        )
        old_block_count = len(allocation.block_ids)
        new_block_count = self._required_blocks(new_token_count)
        additional_blocks = self._take_blocks(new_block_count - old_block_count)
        allocation.block_ids.extend(additional_blocks)
        allocation.token_count = new_token_count
        return self._view(allocation)

    def append_many(
        self,
        sequence_ids: Sequence[str],
        token_counts: int | Sequence[int] = 1,
    ) -> tuple[PagedBlockTable, ...]:
        """Atomically reserve growth for several active sequences.

        Static and continuous batches share one free-block pool.  Checking
        every row independently is not enough: two rows can each appear to
        fit while their combined block demand exceeds the remaining pool.
        This method performs the aggregate check before changing any table.
        """

        normalized_ids = tuple(
            self._normalize_sequence_id(value) for value in sequence_ids
        )
        if not normalized_ids:
            raise ValueError("sequence_ids must not be empty")
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("sequence_ids must be unique")
        if isinstance(token_counts, int):
            normalized_counts = (int(token_counts),) * len(normalized_ids)
        else:
            normalized_counts = tuple(int(value) for value in token_counts)
        if len(normalized_counts) != len(normalized_ids):
            raise ValueError("token_counts must contain one value per sequence")
        if any(value < 1 for value in normalized_counts):
            raise ValueError("token_counts must be positive")

        required_blocks = 0
        for sequence_id, token_count in zip(
            normalized_ids,
            normalized_counts,
            strict=True,
        ):
            allocation = self._require_sequence(sequence_id)
            new_token_count = self._validate_token_count(
                allocation.token_count + token_count,
                name="new token count",
            )
            required_blocks += max(
                0,
                self._required_blocks(new_token_count) - len(allocation.block_ids),
            )
        if required_blocks > self.free_block_count:
            raise PagedKvOutOfMemoryError(
                f"need {required_blocks} additional KV blocks but only "
                f"{self.free_block_count} are free"
            )

        return tuple(
            self.append(sequence_id, token_count)
            for sequence_id, token_count in zip(
                normalized_ids,
                normalized_counts,
                strict=True,
            )
        )

    def truncate(self, sequence_id: str, token_count: int) -> PagedBlockTable:
        """Shrink a sequence and return blocks no longer needed to the pool."""

        allocation = self._require_sequence(sequence_id)
        normalized_tokens = self._validate_token_count(
            token_count,
            name="token_count",
        )
        if normalized_tokens > allocation.token_count:
            raise ValueError(
                f"cannot grow sequence {allocation.sequence_id!r} with truncate"
            )
        required_blocks = self._required_blocks(normalized_tokens)
        released = allocation.block_ids[required_blocks:]
        allocation.block_ids = allocation.block_ids[:required_blocks]
        allocation.token_count = normalized_tokens
        for block_id in released:
            heapq.heappush(self._free_blocks, block_id)
        return self._view(allocation)

    def release(self, sequence_id: str) -> PagedBlockTable:
        """Release a sequence and immediately return all of its blocks."""

        normalized = self._normalize_sequence_id(sequence_id)
        try:
            allocation = self._allocations.pop(normalized)
        except KeyError as error:
            raise PagedKvStateError(
                f"sequence {normalized!r} has no active block table"
            ) from error
        table = self._view(allocation)
        for block_id in allocation.block_ids:
            heapq.heappush(self._free_blocks, block_id)
        return table

    def get_block_table(self, sequence_id: str) -> PagedBlockTable:
        return self._view(self._require_sequence(sequence_id))

    def logical_to_physical(
        self,
        sequence_id: str,
        logical_token_index: int,
    ) -> tuple[int, int]:
        return self.get_block_table(sequence_id).physical_location(logical_token_index)

    def can_allocate(self, token_count: int) -> bool:
        normalized_tokens = self._validate_token_count(
            token_count,
            name="token_count",
        )
        return self._required_blocks(normalized_tokens) <= self.free_block_count

    def can_append(self, sequence_id: str, token_count: int) -> bool:
        if int(token_count) < 0:
            raise ValueError("token_count must be non-negative")
        allocation = self._require_sequence(sequence_id)
        new_token_count = self._validate_token_count(
            allocation.token_count + int(token_count),
            name="new token count",
        )
        additional_blocks = self._required_blocks(new_token_count) - len(
            allocation.block_ids
        )
        return additional_blocks <= self.free_block_count

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-safe capacity, utilization, and block-table metrics."""

        tables = [self.get_block_table(sequence_id) for sequence_id in self.sequence_ids]
        used_tokens = sum(table.token_count for table in tables)
        reserved_slots = sum(table.reserved_token_slots for table in tables)
        wasted_slots = sum(table.wasted_token_slots for table in tables)
        capacity_slots = self.capacity_tokens
        return {
            "layout": "paged_fixed_blocks",
            "block_size": self.block_size,
            "total_blocks": self.num_blocks,
            "allocated_blocks": self.allocated_block_count,
            "free_blocks": self.free_block_count,
            "capacity_token_slots": capacity_slots,
            "used_token_slots": used_tokens,
            "reserved_token_slots": reserved_slots,
            "wasted_token_slots": wasted_slots,
            "free_token_slots": self.free_block_count * self.block_size,
            "capacity_utilization": used_tokens / capacity_slots if capacity_slots else 0.0,
            "reserved_slot_utilization": (
                used_tokens / reserved_slots if reserved_slots else 0.0
            ),
            "internal_fragmentation": (
                wasted_slots / reserved_slots if reserved_slots else 0.0
            ),
            "active_sequence_count": self.active_sequence_count,
            "sequences": [table.to_dict() for table in tables],
        }


class PagedKvCache:
    """Fixed physical K/V tensors backed by a :class:`PagedKvAllocator`.

    The physical layout is ``[layers, blocks, kv_heads, block_size, head_dim]``
    for keys and values separately.  ``gather_layer`` reconstructs the dense
    per-sequence layout ``[1, kv_heads, tokens, head_dim]``.  ``gather_batch``
    left-pads those views and concatenates them, matching the current dense
    model interface while retaining non-contiguous physical allocation.
    """

    def __init__(
        self,
        allocator: PagedKvAllocator,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        if int(num_layers) < 1:
            raise ValueError("num_layers must be positive")
        if int(num_kv_heads) < 1:
            raise ValueError("num_kv_heads must be positive")
        if int(head_dim) < 1:
            raise ValueError("head_dim must be positive")
        if not isinstance(dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")

        self.allocator = allocator
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.dtype = dtype
        normalized_device = torch.device(device)
        if (
            normalized_device.type == "cuda"
            and normalized_device.index is None
            and torch.cuda.is_available()
        ):
            normalized_device = torch.device("cuda", torch.cuda.current_device())
        self.device = normalized_device
        physical_shape = (
            self.num_layers,
            allocator.num_blocks,
            self.num_kv_heads,
            allocator.block_size,
            self.head_dim,
        )
        self.key_blocks = torch.empty(physical_shape, dtype=dtype, device=self.device)
        self.value_blocks = torch.empty(physical_shape, dtype=dtype, device=self.device)

    def _validate_layer_index(self, layer_index: int) -> int:
        normalized = int(layer_index)
        if normalized < 0 or normalized >= self.num_layers:
            raise IndexError(
                f"layer_index {normalized} is outside [0, {self.num_layers})"
            )
        return normalized

    def _normalize_kv_tensor(
        self,
        tensor: torch.Tensor,
        *,
        token_count: int,
        name: str,
    ) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise PagedKvShapeError(f"{name} must be a torch.Tensor")
        normalized = tensor
        if normalized.ndim == 4:
            if normalized.shape[0] != 1:
                raise PagedKvShapeError(
                    f"{name} must have batch size 1 when four-dimensional"
                )
            normalized = normalized[0]
        if normalized.ndim != 3:
            raise PagedKvShapeError(
                f"{name} must have shape [heads, tokens, head_dim] or "
                f"[1, heads, tokens, head_dim], got {tuple(tensor.shape)}"
            )
        expected = (self.num_kv_heads, token_count, self.head_dim)
        if tuple(int(value) for value in normalized.shape) != expected:
            raise PagedKvShapeError(
                f"{name} shape {tuple(normalized.shape)} does not match {expected}"
            )
        if normalized.dtype != self.dtype:
            raise PagedKvShapeError(
                f"{name} dtype {normalized.dtype} does not match {self.dtype}"
            )
        if normalized.device != self.device:
            raise PagedKvShapeError(
                f"{name} device {normalized.device} does not match {self.device}"
            )
        return normalized

    def _write_layer(
        self,
        sequence_id: str,
        layer_index: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        start_token: int,
    ) -> None:
        token_count = int(keys.shape[-2])
        consumed = 0
        while consumed < token_count:
            logical_index = start_token + consumed
            block_id, block_offset = self.allocator.logical_to_physical(
                sequence_id,
                logical_index,
            )
            chunk = min(self.allocator.block_size - block_offset, token_count - consumed)
            source_end = consumed + chunk
            self.key_blocks[layer_index, block_id, :, block_offset : block_offset + chunk, :].copy_(
                keys[:, consumed:source_end, :]
            )
            self.value_blocks[layer_index, block_id, :, block_offset : block_offset + chunk, :].copy_(
                values[:, consumed:source_end, :]
            )
            consumed = source_end

    def append(
        self,
        sequence_id: str,
        layer_kv: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> PagedBlockTable:
        """Append one equally-sized K/V segment for every model layer."""

        layers = tuple(layer_kv)
        if len(layers) != self.num_layers:
            raise PagedKvShapeError(
                f"expected {self.num_layers} layer K/V pairs, received {len(layers)}"
            )
        if not layers:
            raise PagedKvShapeError("layer_kv must not be empty")

        token_count: int | None = None
        normalized_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_index, pair in enumerate(layers):
            if len(pair) != 2:
                raise PagedKvShapeError(
                    f"layer {layer_index} must contain exactly key and value tensors"
                )
            keys, values = pair
            if not isinstance(keys, torch.Tensor) or keys.ndim < 2:
                raise PagedKvShapeError(f"layer {layer_index} keys are invalid")
            current_tokens = int(keys.shape[-2])
            if current_tokens < 1:
                raise PagedKvShapeError("an append must contain at least one token")
            if token_count is None:
                token_count = current_tokens
            elif token_count != current_tokens:
                raise PagedKvShapeError("all layers must append the same token count")
            normalized_keys = self._normalize_kv_tensor(
                keys,
                token_count=current_tokens,
                name=f"layer {layer_index} keys",
            )
            normalized_values = self._normalize_kv_tensor(
                values,
                token_count=current_tokens,
                name=f"layer {layer_index} values",
            )
            normalized_layers.append((normalized_keys, normalized_values))

        assert token_count is not None
        current_table = self.allocator.get_block_table(sequence_id)
        start_token = current_table.token_count
        new_table = self.allocator.append(sequence_id, token_count)
        try:
            for layer_index, (keys, values) in enumerate(normalized_layers):
                self._write_layer(
                    sequence_id,
                    layer_index,
                    keys,
                    values,
                    start_token=start_token,
                )
        except Exception:
            # Restore ownership if a device-side copy or validation unexpectedly
            # fails after the allocator reserved new blocks.
            self.allocator.truncate(sequence_id, start_token)
            raise
        return new_table

    def reserve_append(
        self,
        sequence_ids: Sequence[str],
        token_counts: int | Sequence[int] = 1,
    ) -> tuple[PagedBlockTable, ...]:
        """Reserve future token slots without writing layer data.

        The direct attention path reserves the next decode position before the
        model enters its first decoder layer.  Each layer then writes its own
        K/V values into that already-reserved logical position.
        """

        return self.allocator.append_many(sequence_ids, token_counts)

    def write_layer_segment(
        self,
        sequence_id: str,
        layer_index: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        start_token: int,
    ) -> PagedBlockTable:
        """Write one layer's K/V segment into already-reserved page slots.

        Unlike :meth:`append`, this method does not change the allocator's
        logical token count.  The model visits layers one at a time: a decode
        token reserves one logical position once, then writes one K/V pair per
        layer into that position.
        """

        normalized_layer = self._validate_layer_index(layer_index)
        if not isinstance(keys, torch.Tensor) or keys.ndim < 2:
            raise PagedKvShapeError("keys must be a tensor with a token dimension")
        token_count = int(keys.shape[-2])
        if token_count < 1:
            raise PagedKvShapeError("a layer segment must contain at least one token")
        normalized_keys = self._normalize_kv_tensor(
            keys,
            token_count=token_count,
            name=f"layer {normalized_layer} keys",
        )
        normalized_values = self._normalize_kv_tensor(
            values,
            token_count=token_count,
            name=f"layer {normalized_layer} values",
        )
        start = int(start_token)
        if start < 0:
            raise ValueError("start_token must be non-negative")
        table = self.allocator.get_block_table(sequence_id)
        if start + token_count > table.token_count:
            raise PagedKvStateError(
                f"layer segment [{start}, {start + token_count}) exceeds "
                f"reserved sequence length {table.token_count}"
            )
        self._write_layer(
            sequence_id,
            normalized_layer,
            normalized_keys,
            normalized_values,
            start_token=start,
        )
        return table

    def write_dense_batch(
        self,
        sequence_ids: Sequence[str],
        layer_kv: Sequence[tuple[torch.Tensor, torch.Tensor]],
        *,
        start_token: int = 0,
        block_tables: torch.Tensor | None = None,
    ) -> tuple[PagedBlockTable, ...]:
        """Scatter a rectangular dense prefill cache into physical pages.

        ``layer_kv`` contains one ``[batch, kv_heads, tokens, head_dim]``
        pair per model layer. The source is rectangular because it came from
        the trusted dense model path, but the destination remains paged: one
        indexed assignment writes every request/token row directly to its
        physical block. This avoids the request-by-request, layer-by-layer
        Python loop used by the original adapter.

        The method does not change allocator token counts. Callers reserve the
        full request capacity before writing, just as they do for
        :meth:`write_layer_segment`.
        """

        normalized_ids = tuple(str(sequence_id) for sequence_id in sequence_ids)
        if not normalized_ids:
            raise ValueError("sequence_ids must not be empty")
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("sequence_ids must be unique")

        layers = tuple(layer_kv)
        if len(layers) != self.num_layers:
            raise PagedKvShapeError(
                f"expected {self.num_layers} layer K/V pairs, received {len(layers)}"
            )
        if not layers:
            raise PagedKvShapeError("layer_kv must not be empty")

        batch_size = len(normalized_ids)
        token_count: int | None = None
        normalized_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_index, pair in enumerate(layers):
            if len(pair) != 2:
                raise PagedKvShapeError(
                    f"layer {layer_index} must contain exactly key and value tensors"
                )
            keys, values = pair
            if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
                raise PagedKvShapeError(
                    f"layer {layer_index} keys and values must be tensors"
                )
            if keys.ndim != 4 or values.shape != keys.shape:
                raise PagedKvShapeError(
                    f"layer {layer_index} dense K/V must both have shape "
                    "[batch, heads, tokens, head_dim]"
                )
            current_tokens = int(keys.shape[-2])
            expected_shape = (
                batch_size,
                self.num_kv_heads,
                current_tokens,
                self.head_dim,
            )
            if tuple(int(value) for value in keys.shape) != expected_shape:
                raise PagedKvShapeError(
                    f"layer {layer_index} keys shape {tuple(keys.shape)} "
                    f"does not match {expected_shape}"
                )
            if token_count is None:
                token_count = current_tokens
            elif token_count != current_tokens:
                raise PagedKvShapeError(
                    "all layers must contain the same dense token count"
                )
            for name, tensor in (("keys", keys), ("values", values)):
                if tensor.dtype != self.dtype:
                    raise PagedKvShapeError(
                        f"layer {layer_index} {name} dtype {tensor.dtype} "
                        f"does not match {self.dtype}"
                    )
                if tensor.device != self.device:
                    raise PagedKvShapeError(
                        f"layer {layer_index} {name} device {tensor.device} "
                        f"does not match {self.device}"
                    )
            normalized_layers.append((keys, values))

        assert token_count is not None
        start = int(start_token)
        if start < 0:
            raise ValueError("start_token must be non-negative")
        tables = tuple(
            self.allocator.get_block_table(sequence_id)
            for sequence_id in normalized_ids
        )
        if any(start + token_count > table.token_count for table in tables):
            raise PagedKvStateError(
                "dense layer segment exceeds at least one reserved sequence length"
            )

        if block_tables is None:
            physical_tables = self.block_table_tensor(
                normalized_ids,
                device=self.device,
            )
        else:
            if not isinstance(block_tables, torch.Tensor) or block_tables.ndim != 2:
                raise PagedKvShapeError(
                    "block_tables must have shape [batch, logical_blocks]"
                )
            if int(block_tables.shape[0]) != batch_size:
                raise PagedKvShapeError(
                    "block_tables row count must match sequence_ids"
                )
            if block_tables.device != self.device:
                raise PagedKvShapeError(
                    f"block_tables device {block_tables.device} does not match "
                    f"{self.device}"
                )
            physical_tables = block_tables

        required_blocks = max(
            (table.allocated_block_count for table in tables),
            default=0,
        )
        if int(physical_tables.shape[1]) < required_blocks:
            raise PagedKvShapeError(
                "block_tables does not contain every allocated logical block"
            )

        logical_positions = torch.arange(
            start,
            start + token_count,
            dtype=torch.long,
            device=self.device,
        )
        physical_blocks = physical_tables[:, logical_positions // self.allocator.block_size]
        offsets = logical_positions.remainder(self.allocator.block_size)

        # Advanced indexing produces a [batch, tokens, heads, dim] destination;
        # permuting the dense source once gives it the same logical order. The
        # request allocations are disjoint, so no indexed destinations alias.
        for layer_index, (keys, values) in enumerate(normalized_layers):
            self.key_blocks[layer_index][
                physical_blocks,
                :,
                offsets,
                :,
            ] = keys.permute(0, 2, 1, 3)
            self.value_blocks[layer_index][
                physical_blocks,
                :,
                offsets,
                :,
            ] = values.permute(0, 2, 1, 3)
        return tables

    def gather_layer(
        self,
        sequence_id: str,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather one sequence into dense ``[1, heads, tokens, dim]`` tensors."""

        normalized_layer = self._validate_layer_index(layer_index)
        table = self.allocator.get_block_table(sequence_id)
        if table.token_count == 0:
            empty_shape = (1, self.num_kv_heads, 0, self.head_dim)
            return (
                torch.empty(empty_shape, dtype=self.dtype, device=self.device),
                torch.empty(empty_shape, dtype=self.dtype, device=self.device),
            )

        block_indices = torch.tensor(
            table.block_ids,
            dtype=torch.long,
            device=self.device,
        )
        keys = self.key_blocks[normalized_layer].index_select(0, block_indices)
        values = self.value_blocks[normalized_layer].index_select(0, block_indices)

        def flatten_blocks(block_tensor: torch.Tensor) -> torch.Tensor:
            # Physical layout is [blocks, heads, block_size, dim].  Move the
            # token axis next to blocks before flattening them.
            flattened = block_tensor.permute(0, 2, 1, 3).reshape(
                -1,
                self.num_kv_heads,
                self.head_dim,
            )
            return flattened[: table.token_count].permute(1, 0, 2).unsqueeze(0).contiguous()

        return flatten_blocks(keys), flatten_blocks(values)

    def gather_batch(
        self,
        sequence_ids: Sequence[str],
        *,
        left_pad: bool = True,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Gather a batch, optionally left-padding to one dense sequence length."""

        normalized_ids = tuple(str(sequence_id) for sequence_id in sequence_ids)
        if not normalized_ids:
            raise ValueError("sequence_ids must not be empty")
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("sequence_ids must be unique")
        lengths = tuple(
            self.allocator.get_block_table(sequence_id).token_count
            for sequence_id in normalized_ids
        )
        max_length = max(lengths)
        if not left_pad and any(length != max_length for length in lengths):
            raise ValueError("variable-length batches require left_pad=True")

        gathered: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_index in range(self.num_layers):
            key_rows: list[torch.Tensor] = []
            value_rows: list[torch.Tensor] = []
            for sequence_id, length in zip(normalized_ids, lengths, strict=True):
                keys, values = self.gather_layer(sequence_id, layer_index)
                if left_pad and length < max_length:
                    padding_shape = (1, self.num_kv_heads, max_length - length, self.head_dim)
                    key_padding = torch.zeros(
                        padding_shape,
                        dtype=self.dtype,
                        device=self.device,
                    )
                    value_padding = torch.zeros_like(key_padding)
                    keys = torch.cat((key_padding, keys), dim=-2)
                    values = torch.cat((value_padding, values), dim=-2)
                key_rows.append(keys)
                value_rows.append(values)
            gathered.append(
                (
                    torch.cat(key_rows, dim=0),
                    torch.cat(value_rows, dim=0),
                )
            )
        return tuple(gathered)

    def block_table_tensor(
        self,
        sequence_ids: Sequence[str],
        *,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Return padded physical block IDs for a future indirection kernel."""

        normalized_ids = tuple(str(sequence_id) for sequence_id in sequence_ids)
        if not normalized_ids:
            raise ValueError("sequence_ids must not be empty")
        tables = tuple(self.allocator.get_block_table(sequence_id) for sequence_id in normalized_ids)
        width = max((table.allocated_block_count for table in tables), default=0)
        output = torch.full(
            (len(tables), width),
            -1,
            dtype=torch.long,
            device=self.device if device is None else device,
        )
        for row, table in enumerate(tables):
            if table.block_ids:
                output[row, : len(table.block_ids)] = torch.tensor(
                    table.block_ids,
                    dtype=torch.long,
                    device=output.device,
                )
        return output

    def attention_mask(
        self,
        sequence_ids: Sequence[str],
        *,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Build the right-aligned validity mask for a gathered batch."""

        normalized_ids = tuple(str(sequence_id) for sequence_id in sequence_ids)
        if not normalized_ids:
            raise ValueError("sequence_ids must not be empty")
        lengths = tuple(
            self.allocator.get_block_table(sequence_id).token_count
            for sequence_id in normalized_ids
        )
        max_length = max(lengths)
        mask = torch.zeros(
            (len(lengths), max_length),
            dtype=torch.long,
            device=self.device if device is None else device,
        )
        for row, length in enumerate(lengths):
            if length:
                mask[row, max_length - length :] = 1
        return mask

    def as_dynamic_cache(
        self,
        sequence_ids: Sequence[str],
        *,
        model_config: Any,
    ) -> Any:
        """Materialize a dense Transformers cache for the correctness path."""

        from transformers import DynamicCache

        return DynamicCache(
            ddp_cache_data=self.gather_batch(sequence_ids),
            config=model_config,
        )

    def append_from_dynamic_cache(
        self,
        sequence_ids: Sequence[str],
        dense_cache: Any,
        *,
        previous_token_counts: Sequence[int],
        appended_token_counts: int | Sequence[int] = 1,
    ) -> tuple[PagedBlockTable, ...]:
        """Copy right-aligned new tokens from a dense cache into page blocks.

        ``dense_cache`` is expected to use the same left-padded convention as
        :meth:`gather_batch`: each row's valid tokens occupy the suffix of the
        physical sequence dimension.  This adapter lets a model continue to
        run through the trusted Transformers cache while ownership remains in
        the paged store.
        """

        normalized_ids = tuple(str(sequence_id) for sequence_id in sequence_ids)
        if not normalized_ids:
            raise ValueError("sequence_ids must not be empty")
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("sequence_ids must be unique")
        previous = tuple(int(value) for value in previous_token_counts)
        if len(previous) != len(normalized_ids) or any(value < 0 for value in previous):
            raise ValueError(
                "previous_token_counts must contain one non-negative value per sequence"
            )
        if isinstance(appended_token_counts, int):
            appended = (int(appended_token_counts),) * len(normalized_ids)
        else:
            appended = tuple(int(value) for value in appended_token_counts)
        if len(appended) != len(normalized_ids) or any(value < 0 for value in appended):
            raise ValueError(
                "appended_token_counts must contain one non-negative value per sequence"
            )

        layers = getattr(dense_cache, "layers", None)
        if layers is None or len(layers) != self.num_layers:
            raise PagedKvShapeError("dense cache has an incompatible layer count")
        dense_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        physical_length: int | None = None
        batch_size = len(normalized_ids)
        for layer_index, layer in enumerate(layers):
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
                raise PagedKvShapeError(f"dense cache layer {layer_index} is not initialized")
            if keys.shape != values.shape or keys.ndim != 4:
                raise PagedKvShapeError(
                    f"dense cache layer {layer_index} must have matching four-dimensional K/V"
                )
            if int(keys.shape[0]) != batch_size:
                raise PagedKvShapeError("dense cache batch size does not match sequence IDs")
            current_length = int(keys.shape[-2])
            if physical_length is None:
                physical_length = current_length
            elif physical_length != current_length:
                raise PagedKvShapeError("dense cache layers have different sequence lengths")
            dense_layers.append((keys, values))
        assert physical_length is not None

        payloads: list[tuple[tuple[torch.Tensor, torch.Tensor], ...] | None] = []
        additional_blocks_required = 0
        for row, (sequence_id, old_length, append_count) in enumerate(
            zip(normalized_ids, previous, appended, strict=True)
        ):
            current_table = self.allocator.get_block_table(sequence_id)
            if current_table.token_count != old_length:
                raise PagedKvStateError(
                    f"sequence {sequence_id!r} has {current_table.token_count} tokens, "
                    f"expected {old_length}"
                )
            if append_count == 0:
                payloads.append(None)
                continue
            # Validate per-sequence context capacity before the aggregate
            # free-block check below.  The return value is checked in
            # aggregate form because all rows share one pool.
            self.allocator.can_append(sequence_id, append_count)
            target_length = old_length + append_count
            if target_length > physical_length:
                raise PagedKvShapeError(
                    f"dense cache length {physical_length} cannot provide "
                    f"{target_length} right-aligned tokens"
                )
            # The paged store already owns ``old_length`` tokens.  Only copy
            # the newly appended suffix; copying the whole valid suffix would
            # count the old tokens a second time.
            start = physical_length - append_count
            end = physical_length
            payloads.append(
                tuple(
                    (
                        keys[row : row + 1, :, start:end, :],
                        values[row : row + 1, :, start:end, :],
                    )
                    for keys, values in dense_layers
                )
            )
            old_blocks = current_table.allocated_block_count
            new_blocks = math.ceil(target_length / self.allocator.block_size)
            additional_blocks_required += max(0, new_blocks - old_blocks)

        # Check the aggregate page demand before mutating any sequence.  A
        # per-row ``can_append`` check is insufficient when several rows need
        # a new block from the same free pool.
        if additional_blocks_required > self.allocator.free_block_count:
            raise PagedKvOutOfMemoryError(
                f"need {additional_blocks_required} additional KV blocks but only "
                f"{self.allocator.free_block_count} are free"
            )

        result: list[PagedBlockTable] = []
        for sequence_id, payload in zip(normalized_ids, payloads, strict=True):
            if payload is not None:
                self.append(sequence_id, payload)
            result.append(self.allocator.get_block_table(sequence_id))
        return tuple(result)

    def release(self, sequence_id: str) -> PagedBlockTable:
        """Release a request and clear its physical blocks before reuse."""

        table = self.allocator.get_block_table(sequence_id)
        if table.block_ids:
            block_indices = torch.tensor(
                table.block_ids,
                dtype=torch.long,
                device=self.device,
            )
            # The model-facing runner may allocate the physical tensors inside
            # inference mode.  Keep cleanup in that mode as well so PyTorch's
            # inference-tensor mutation guard is respected.
            with torch.inference_mode():
                self.key_blocks.index_fill_(1, block_indices, 0)
                self.value_blocks.index_fill_(1, block_indices, 0)
        return self.allocator.release(sequence_id)
