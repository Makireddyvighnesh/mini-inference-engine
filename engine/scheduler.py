"""Deterministic FIFO static-batch scheduling primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ScheduledRequest:
    request_id: str
    scheduled_arrival_ms: float
    ordinal: int

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty")
        if self.scheduled_arrival_ms < 0:
            raise ValueError("scheduled_arrival_ms must be non-negative")
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative")


class StaticBatchScheduler:
    """Admit arrived requests and dispatch ready work in FIFO batches.

    A dispatched batch runs to completion. Requests arriving during that model
    call remain pending until the next scheduling boundary. That limitation is
    intentional: iteration-level admission belongs to continuous batching.
    """

    def __init__(self, requests: Iterable[ScheduledRequest], *, max_batch_size: int):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        pending = sorted(
            tuple(requests),
            key=lambda item: (item.scheduled_arrival_ms, item.ordinal),
        )
        identifiers = [item.request_id for item in pending]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("scheduled request IDs must be unique")
        self.max_batch_size = max_batch_size
        self._pending = pending
        self._ready: list[ScheduledRequest] = []
        self.maximum_queue_depth = 0

    @property
    def empty(self) -> bool:
        return not self._pending and not self._ready

    @property
    def next_arrival_ms(self) -> float | None:
        return self._pending[0].scheduled_arrival_ms if self._pending else None

    @property
    def ready_count(self) -> int:
        return len(self._ready)

    def admit(self, now_ms: float) -> tuple[ScheduledRequest, ...]:
        admitted: list[ScheduledRequest] = []
        while self._pending and self._pending[0].scheduled_arrival_ms <= now_ms:
            item = self._pending.pop(0)
            self._ready.append(item)
            admitted.append(item)
        self.maximum_queue_depth = max(self.maximum_queue_depth, len(self._ready))
        return tuple(admitted)

    def cancel_ready(self, request_ids: set[str]) -> tuple[ScheduledRequest, ...]:
        cancelled = tuple(item for item in self._ready if item.request_id in request_ids)
        self._ready = [item for item in self._ready if item.request_id not in request_ids]
        return cancelled

    def next_batch(self) -> tuple[ScheduledRequest, ...]:
        batch = tuple(self._ready[: self.max_batch_size])
        del self._ready[: len(batch)]
        return batch

