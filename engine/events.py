"""Requests and what happens to them: the engine's input and event records."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Mapping


EVENT_SCHEMA_VERSION = 1

REQUEST_STATUSES = frozenset({"completed", "failed", "cancelled"})
CANONICAL_EVENTS = frozenset(
    {
        "arrival",
        "admission",
        "execution_start",
        "prefill_start",
        "prefill_end",
        "first_token_ready",
        "token_ready",
        "first_token_sent",
        "token_sent",
        "completion",
        "error",
        "cancelled",
    }
)


def _validate_json_object(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    normalized = dict(value)
    try:
        json.dumps(normalized)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain JSON-compatible values") from error
    return normalized


def _validate_nonempty_string(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class RequestSpec:
    """A deterministic request input independent of any model runtime."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    scheduled_arrival_ms: float = 0.0
    category: str = "default"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_nonempty_string(self.request_id, name="request_id")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        normalized_ids = tuple(int(token_id) for token_id in self.prompt_token_ids)
        if any(token_id < 0 or token_id > 2**32 - 1 for token_id in normalized_ids):
            raise ValueError("prompt_token_ids must contain unsigned 32-bit IDs")
        object.__setattr__(self, "prompt_token_ids", normalized_ids)
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not math.isfinite(float(self.scheduled_arrival_ms)):
            raise ValueError("scheduled_arrival_ms must be finite")
        if self.scheduled_arrival_ms < 0:
            raise ValueError("scheduled_arrival_ms must be non-negative")
        _validate_nonempty_string(self.category, name="category")
        object.__setattr__(
            self,
            "metadata",
            _validate_json_object(self.metadata, name="metadata"),
        )

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def prompt_sha256(self) -> str:
        payload = b"".join(
            struct.pack("<I", token_id) for token_id in self.prompt_token_ids
        )
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prompt_token_ids": list(self.prompt_token_ids),
            "prompt_tokens": self.prompt_tokens,
            "prompt_sha256": self.prompt_sha256,
            "max_new_tokens": self.max_new_tokens,
            "scheduled_arrival_ms": float(self.scheduled_arrival_ms),
            "category": self.category,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RequestSpec":
        return cls(
            request_id=str(payload["request_id"]),
            prompt_token_ids=tuple(int(value) for value in payload["prompt_token_ids"]),
            max_new_tokens=int(payload["max_new_tokens"]),
            scheduled_arrival_ms=float(payload.get("scheduled_arrival_ms", 0.0)),
            category=str(payload.get("category", "default")),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class EventRecord:
    """One request event with a timestamp relative to a benchmark run."""

    request_id: str
    event: str
    timestamp_ns: int
    sequence: int
    token_index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_nonempty_string(self.request_id, name="request_id")
        _validate_nonempty_string(self.event, name="event")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.token_index is not None and self.token_index < 0:
            raise ValueError("token_index must be non-negative")
        object.__setattr__(
            self,
            "metadata",
            _validate_json_object(self.metadata, name="metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVENT_SCHEMA_VERSION,
            "request_id": self.request_id,
            "event": self.event,
            "timestamp_ns": self.timestamp_ns,
            "timestamp_ms": self.timestamp_ns / 1_000_000.0,
            "sequence": self.sequence,
            "token_index": self.token_index,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RequestOutcome:
    """Normalized result returned by a model runner."""

    status: str = "completed"
    generated_token_ids: tuple[int, ...] = ()
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in REQUEST_STATUSES:
            allowed = ", ".join(sorted(REQUEST_STATUSES))
            raise ValueError(f"status must be one of {allowed}")
        normalized_ids = tuple(int(token_id) for token_id in self.generated_token_ids)
        if any(token_id < 0 or token_id > 2**32 - 1 for token_id in normalized_ids):
            raise ValueError("generated_token_ids must contain unsigned 32-bit IDs")
        object.__setattr__(self, "generated_token_ids", normalized_ids)
        if self.status == "failed" and not self.error:
            raise ValueError("failed outcomes must include an error")
        object.__setattr__(
            self,
            "metadata",
            _validate_json_object(self.metadata, name="metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "generated_token_ids": list(self.generated_token_ids),
            "generated_tokens": len(self.generated_token_ids),
            "error": self.error,
            "metadata": dict(self.metadata),
        }


class RequestEventRecorder:
    """Record canonical request events on a run-relative monotonic clock."""

    def __init__(self, request_id: str, *, run_started_ns: int) -> None:
        if not request_id.strip():
            raise ValueError("request_id must not be empty")
        self.request_id = request_id
        self.run_started_ns = run_started_ns
        self._events: list[EventRecord] = []

    @property
    def events(self) -> tuple[EventRecord, ...]:
        return tuple(self._events)

    @property
    def last_timestamp_ns(self) -> int:
        return self._events[-1].timestamp_ns if self._events else 0

    def now_ns(self) -> int:
        return max(0, time.perf_counter_ns() - self.run_started_ns)

    def record(
        self,
        event: str,
        *,
        timestamp_ns: int | None = None,
        token_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EventRecord:
        relative_ns = self.now_ns() if timestamp_ns is None else int(timestamp_ns)
        if relative_ns < 0:
            raise ValueError("event timestamp cannot be before the run start")
        if self._events and relative_ns < self.last_timestamp_ns:
            raise ValueError("request events must be recorded in timestamp order")
        event_record = EventRecord(
            request_id=self.request_id,
            event=event,
            timestamp_ns=relative_ns,
            sequence=len(self._events),
            token_index=token_index,
            metadata=metadata or {},
        )
        self._events.append(event_record)
        return event_record

    def mark_token_ready(
        self,
        token_index: int,
        *,
        token_id: int | None = None,
        timestamp_ns: int | None = None,
    ) -> None:
        metadata = {} if token_id is None else {"token_id": int(token_id)}
        event_timestamp = self.now_ns() if timestamp_ns is None else int(timestamp_ns)
        self.record(
            "token_ready",
            timestamp_ns=event_timestamp,
            token_index=token_index,
            metadata=metadata,
        )
        if token_index == 0:
            self.record(
                "first_token_ready",
                timestamp_ns=event_timestamp,
                token_index=token_index,
                metadata=metadata,
            )

    def mark_token_sent(
        self,
        token_index: int,
        *,
        token_id: int | None = None,
        timestamp_ns: int | None = None,
    ) -> None:
        metadata = {} if token_id is None else {"token_id": int(token_id)}
        event_timestamp = self.now_ns() if timestamp_ns is None else int(timestamp_ns)
        self.record(
            "token_sent",
            timestamp_ns=event_timestamp,
            token_index=token_index,
            metadata=metadata,
        )
        if token_index == 0:
            self.record(
                "first_token_sent",
                timestamp_ns=event_timestamp,
                token_index=token_index,
                metadata=metadata,
            )
