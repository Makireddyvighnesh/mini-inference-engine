"""Request lifecycle state used by the local inference scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RequestState(str, Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODING = "decoding"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"


_ALLOWED_TRANSITIONS = {
    RequestState.WAITING: {RequestState.PREFILL, RequestState.CANCELLED, RequestState.FAILED},
    RequestState.PREFILL: {RequestState.DECODING, RequestState.CANCELLED, RequestState.FAILED},
    RequestState.DECODING: {RequestState.FINISHED, RequestState.CANCELLED, RequestState.FAILED},
    RequestState.FINISHED: set(),
    RequestState.CANCELLED: set(),
    RequestState.FAILED: set(),
}


class RequestStateError(RuntimeError):
    """Raised when a request attempts an invalid lifecycle transition."""


@dataclass
class RequestLifecycle:
    request_id: str
    state: RequestState = RequestState.WAITING
    history: list[str] = field(default_factory=lambda: [RequestState.WAITING.value])
    resources_released: bool = False

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty")

    @property
    def terminal(self) -> bool:
        return self.state in {RequestState.FINISHED, RequestState.CANCELLED, RequestState.FAILED}

    def transition(self, target: RequestState) -> None:
        if target not in _ALLOWED_TRANSITIONS[self.state]:
            raise RequestStateError(
                f"invalid request transition: {self.state.value} -> {target.value}"
            )
        self.state = target
        self.history.append(target.value)

    def release_resources(self) -> None:
        if not self.terminal:
            raise RequestStateError("resources can only be released for terminal requests")
        self.resources_released = True
