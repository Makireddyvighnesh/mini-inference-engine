"""Time-budgeted, traffic-aware chunk sizing for mixed prefill/decode steps.

A forward pass cannot be interrupted, so a request that arrives mid-step
waits for the rest of it.  Chunk size is therefore chosen from a *time* limit,
not a fixed token count:

* busy (rows decoding, other prompts waiting, or a recent arrival): steps are
  capped at ``busy_step_ms`` so a newcomer or a decoding row waits briefly;
* idle (one prompt, nothing else, no recent arrival): steps may grow to
  ``idle_step_ms`` for efficiency, which still bounds the wait of a request
  that arrives just after the step starts.

``StepCostModel`` predicts a step's wall time from its decode rows and prompt
chunks and recalibrates online from measured steps.  Waiting prompts are
served shortest-remaining-first with aging, so a long prompt is not starved.
Tokens per step are also capped by free GPU memory.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


LOWER_RIGHT_MIN_QUERY_ROWS = 129  # mirrors engine.generation.chunked_prefill


def attention_work(start: int, count: int) -> float:
    """Causal attention work for ``count`` new tokens after ``start`` cached.

    A fresh prompt or a chunk of more than 128 tokens computes only its own
    rows (``n*s + n**2/2``); a smaller continuation chunk is padded to the
    whole prompt for exactness and costs ``(s + n)**2 / 2``.
    """

    if start and count < LOWER_RIGHT_MIN_QUERY_ROWS:
        return (start + count) ** 2 / 2.0
    return count * start + count * count / 2.0


def _features(decode_rows: int, segments: Sequence[tuple[int, int]]) -> np.ndarray:
    """[1, prompt tokens, attention work, decode rows] for one step."""

    tokens = sum(count for _, count in segments)
    attention = sum(attention_work(start, count) for start, count in segments)
    return np.array([1.0, float(tokens), attention, float(decode_rows)])


@dataclass
class StepCostModel:
    """Linear step-time model with a prior, refit by ridge regression online.

    Priors come from the L4 prefill-only measurements (2026-10-07): about
    55 ms fixed cost, ~0.17 ms per prompt token, and an attention term that
    reproduces 1-prompt TTFT from 128 to 8192 tokens within ~15%.
    """

    prior: tuple[float, float, float, float] = (55.0, 0.17, 3.4e-5, 0.3)
    prior_weight: float = 4.0
    window: int = 128
    _rows: deque = field(default_factory=lambda: deque(maxlen=128))
    _coef: np.ndarray | None = None

    def __post_init__(self) -> None:
        self._rows = deque(maxlen=self.window)
        self._coef = np.array(self.prior, dtype=float)

    @property
    def coefficients(self) -> tuple[float, ...]:
        return tuple(float(v) for v in self._coef)

    def predict(self, decode_rows: int, segments: Sequence[tuple[int, int]]) -> float:
        return float(_features(decode_rows, segments) @ self._coef)

    def observe(self, decode_rows: int, segments: Sequence[tuple[int, int]], measured_ms: float) -> None:
        self._rows.append((_features(decode_rows, segments), float(measured_ms)))
        if len(self._rows) < 4:
            return
        x = np.stack([row for row, _ in self._rows])
        y = np.array([value for _, value in self._rows])
        prior = np.array(self.prior, dtype=float)
        # Scale columns so the ridge pull toward the prior is comparable per term.
        scale = np.maximum(np.abs(x).max(axis=0), 1e-9)
        xs = x / scale
        penalty = self.prior_weight * np.eye(4)
        coef_scaled = np.linalg.solve(xs.T @ xs + penalty, xs.T @ y + penalty @ (prior * scale))
        self._coef = np.maximum(coef_scaled / scale, 0.0)


@dataclass(frozen=True)
class PromptCandidate:
    key: str
    prompt_tokens: int
    start: int  # tokens of this prompt already prefilled
    waited_ms: float

    @property
    def remaining(self) -> int:
        return self.prompt_tokens - self.start


@dataclass
class AdaptiveChunkPlanner:
    busy_step_ms: float = 150.0
    idle_step_ms: float = 400.0
    min_chunk_tokens: int = 144  # > 128 rows keeps continuation chunks on the fast exact path
    max_step_tokens: int = 16384
    aging_tokens_per_s: float = 2000.0
    recent_arrival_ms: float = 500.0
    granularity: int = 16
    cost: StepCostModel = field(default_factory=StepCostModel)

    def is_busy(self, *, decode_rows: int, waiting_prompts: int, queued: int, ms_since_arrival: float | None) -> bool:
        recent = ms_since_arrival is not None and ms_since_arrival < self.recent_arrival_ms
        return decode_rows > 0 or waiting_prompts > 1 or queued > 0 or recent

    def plan(
        self,
        candidates: Sequence[PromptCandidate],
        *,
        decode_rows: int,
        busy: bool,
        memory_token_cap: int | None = None,
    ) -> tuple[list[tuple[PromptCandidate, int]], dict]:
        """Choose (candidate, tokens) chunks whose predicted step time fits the limit."""

        limit_ms = self.busy_step_ms if busy else self.idle_step_ms
        token_cap = self.max_step_tokens if memory_token_cap is None else min(self.max_step_tokens, memory_token_cap)
        token_cap = max(token_cap - decode_rows, self.min_chunk_tokens)
        order = sorted(candidates, key=lambda c: c.remaining - self.aging_tokens_per_s * c.waited_ms / 1000.0)
        chosen: list[tuple[PromptCandidate, int]] = []
        segments: list[tuple[int, int]] = []
        used = 0
        for candidate in order:
            room = min(candidate.remaining, token_cap - used)
            if room <= 0:
                break
            best = 0
            low, high = 1, room
            while low <= high:  # largest chunk whose predicted step still fits
                mid = (low + high) // 2
                if self.cost.predict(decode_rows, segments + [(candidate.start, mid)]) <= limit_ms:
                    best, low = mid, mid + 1
                else:
                    high = mid - 1
            if best < room:
                best -= best % self.granularity  # page-aligned unless finishing the prompt
            if not chosen and best < min(room, self.min_chunk_tokens):
                best = min(room, self.min_chunk_tokens)  # always make progress
            tail = candidate.remaining - best
            if best and 0 < tail < LOWER_RIGHT_MIN_QUERY_ROWS:
                # Avoid stranding a tiny final chunk on the padded slow path:
                # finish the prompt if it fits, else leave a remainder > 128.
                if candidate.remaining <= room:
                    best = candidate.remaining
                elif best - LOWER_RIGHT_MIN_QUERY_ROWS >= self.min_chunk_tokens:
                    best -= LOWER_RIGHT_MIN_QUERY_ROWS
            if best <= 0:
                break
            chosen.append((candidate, best))
            segments.append((candidate.start, best))
            used += best
        info = {"busy": busy, "limit_ms": limit_ms, "token_cap": token_cap,
                "predicted_ms": self.cost.predict(decode_rows, segments)}
        return chosen, info


__all__ = ["AdaptiveChunkPlanner", "PromptCandidate", "StepCostModel"]
