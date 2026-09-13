"""Deterministic FIFO scheduling primitives for static and continuous batches."""

from __future__ import annotations

from collections.abc import Mapping
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


class ContinuousBatchScheduler:
    """FIFO admission for an iteration-level in-flight batch scheduler.

    The scheduler owns only request availability. Model execution is performed
    by the continuous runner, which rebuilds its active decode batch after each
    iteration. ``max_prefill_tokens`` limits how much new prompt work can be
    admitted in one iteration; the first request is admitted even when it is
    individually larger than that budget so a request cannot starve forever.
    """

    def __init__(
        self,
        requests: Iterable[ScheduledRequest],
        *,
        max_batch_size: int,
        max_prefill_tokens: int,
        max_wait_ms: float = 0.0,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if max_prefill_tokens < 1:
            raise ValueError("max_prefill_tokens must be positive")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be non-negative")
        pending = sorted(
            tuple(requests),
            key=lambda item: (item.scheduled_arrival_ms, item.ordinal),
        )
        identifiers = [item.request_id for item in pending]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("scheduled request IDs must be unique")
        self.max_batch_size = max_batch_size
        self.max_prefill_tokens = max_prefill_tokens
        self.max_wait_ms = float(max_wait_ms)
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
    def oldest_ready_arrival_ms(self) -> float | None:
        return self._ready[0].scheduled_arrival_ms if self._ready else None

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

    def should_wait_for_batch(self, now_ms: float, *, active_count: int) -> bool:
        """Return whether an initial batching window should remain open.

        A live decode batch is never paused just to fill a prefill batch. The
        wait window applies only while no sequence is decoding, which avoids
        adding artificial delay to already-running traffic.
        """

        if active_count > 0 or not self._ready or self.max_wait_ms <= 0:
            return False
        oldest = self.oldest_ready_arrival_ms
        if oldest is None:
            return False
        return now_ms < oldest + self.max_wait_ms

    def next_batch(
        self,
        prompt_tokens_by_id: Mapping[str, int],
        *,
        max_requests: int | None = None,
    ) -> tuple[ScheduledRequest, ...]:
        """Select a FIFO prefill batch under request and token budgets."""

        request_limit = self.max_batch_size if max_requests is None else max_requests
        if request_limit < 1:
            return ()
        selected: list[ScheduledRequest] = []
        total_prompt_tokens = 0
        for item in self._ready:
            if len(selected) >= request_limit:
                break
            try:
                prompt_tokens = int(prompt_tokens_by_id[item.request_id])
            except KeyError as error:
                raise KeyError(
                    f"missing prompt token count for {item.request_id!r}"
                ) from error
            if (
                selected
                and total_prompt_tokens + prompt_tokens > self.max_prefill_tokens
            ):
                break
            selected.append(item)
            total_prompt_tokens += prompt_tokens

        del self._ready[: len(selected)]
        return tuple(selected)
