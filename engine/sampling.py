"""Per-request token selection: greedy, temperature, top-k, top-p, min-p, repetition penalty.

One call handles a whole batch on the GPU; rows may use different settings.
Each request owns a small CPU random stream, seeded per request, and draws one
number per token. A request's random numbers therefore do not depend on which
other requests share its batch.

Filter order: repetition penalty -> temperature -> top-k -> top-p -> min-p.
``temperature == 0`` means greedy (argmax after any repetition penalty). A row
with default settings gets ``logits.argmax``, exactly as the engine without
sampling.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0
    top_k: int = 0  # 0 = off
    top_p: float = 1.0  # 1 = off
    min_p: float = 0.0  # 0 = off
    repetition_penalty: float = 1.0  # 1 = off
    seed: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not 0 <= self.min_p < 1:
            raise ValueError("min_p must be in [0, 1)")
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be finite and positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> SamplingParams:
        return cls(**values)


@dataclass
class RequestSampler:
    """One request's settings, random stream and optional per-token statistics."""

    params: SamplingParams
    record_stats: bool = False
    random: random.Random = field(init=False)
    stats: list[dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.random = random.Random(self.params.seed)


def apply_repetition_penalty(logits: torch.Tensor, token_ids: Sequence[int], penalty: float) -> torch.Tensor:
    """Make already-seen tokens less likely: positive scores divided, negative multiplied."""

    if penalty == 1.0 or not token_ids:
        return logits
    seen = torch.tensor(sorted(set(token_ids)), device=logits.device, dtype=torch.long)
    scores = logits[seen]
    logits = logits.clone()
    logits[seen] = torch.where(scores < 0, scores * penalty, scores / penalty)
    return logits


def select_tokens(
    logits: torch.Tensor,
    samplers: Sequence[RequestSampler | None],
    histories: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Choose one token per row of ``[rows, vocab]`` logits, returned as ``[rows, 1]``.

    A ``None`` sampler means plain greedy. ``histories`` holds each row's prompt
    and generated token IDs, read only for repetition penalties.
    """

    if len(samplers) != logits.shape[0] or len(histories) != logits.shape[0]:
        raise ValueError("one sampler and one history are required per row")
    tokens = logits.argmax(dim=-1, keepdim=True)
    if not any(samplers):
        return tokens
    raw = logits.float()
    adjusted = raw
    for row, sampler in enumerate(samplers):
        if sampler is not None and sampler.params.repetition_penalty != 1.0:
            if adjusted is raw:
                adjusted = raw.clone()
            adjusted[row] = apply_repetition_penalty(raw[row], histories[row], sampler.params.repetition_penalty)
    rows = [row for row, s in enumerate(samplers) if s is not None]
    greedy = [row for row in rows if samplers[row].params.temperature == 0]
    if greedy:
        tokens[greedy] = adjusted[greedy].argmax(dim=-1, keepdim=True).to(tokens.dtype)
    sampled = [row for row in rows if samplers[row].params.temperature > 0]
    kept = {}
    if sampled:
        chosen, kept_counts = _sample_rows(adjusted[sampled], [samplers[row] for row in sampled])
        tokens[sampled] = chosen.to(tokens.dtype)
        kept = dict(zip(sampled, kept_counts.tolist()))
    if any(samplers[row].record_stats for row in rows):
        _record_stats(raw, tokens, samplers, kept)
    return tokens


def _sample_rows(logits: torch.Tensor, samplers: Sequence[RequestSampler]) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter and draw one token per row by inverting the cumulative distribution."""

    device = logits.device
    params = [s.params for s in samplers]
    temperature = torch.tensor([p.temperature for p in params], device=device, dtype=logits.dtype)
    sorted_logits, order = (logits / temperature[:, None]).sort(dim=-1, descending=True, stable=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    vocab = probs.shape[1]
    keep = torch.ones_like(probs, dtype=torch.bool)
    top_k = torch.tensor([p.top_k or vocab for p in params], device=device)
    keep &= torch.arange(vocab, device=device)[None, :] < top_k[:, None]
    top_p = torch.tensor([p.top_p for p in params], device=device, dtype=logits.dtype)
    # Keep the smallest set reaching top_p: a token stays if the mass before it is still short of p.
    keep &= (probs.cumsum(dim=-1) - probs < top_p[:, None]) | (top_p[:, None] >= 1)
    min_p = torch.tensor([p.min_p for p in params], device=device, dtype=logits.dtype)
    keep &= probs >= min_p[:, None] * probs[:, :1]
    keep[:, 0] = True  # the most likely token always survives
    filtered = probs * keep
    cumulative = filtered.cumsum(dim=-1)
    draws = torch.tensor([s.random.random() for s in samplers], device=device, dtype=logits.dtype)
    target = (draws * cumulative[:, -1])[:, None]
    ranks = torch.searchsorted(cumulative, target, right=True)
    # Rounding can push the draw past the last kept token; the cap stays on a kept token.
    ranks = torch.minimum(ranks, keep.sum(dim=-1, keepdim=True) - 1)
    return order.gather(1, ranks), keep.sum(dim=-1)


def _record_stats(
    raw: torch.Tensor,
    tokens: torch.Tensor,
    samplers: Sequence[RequestSampler | None],
    kept: Mapping[int, int],
) -> None:
    """Record the model's own confidence (temperature 1, no filters) and how the choice relates to it."""

    rows = [row for row, s in enumerate(samplers) if s is not None and s.record_stats]
    logp = torch.log_softmax(raw[rows], dim=-1)
    probs = logp.exp()
    entropy = -(probs * logp).sum(dim=-1) / math.log(2)
    top = probs.max(dim=-1)
    chosen = tokens[rows]
    chosen_prob = probs.gather(1, chosen)
    ranks = (probs > chosen_prob).sum(dim=-1)
    for index, row in enumerate(rows):
        samplers[row].stats.append({
            "entropy_bits": float(entropy[index]), "top1_prob": float(top.values[index]),
            "chosen_prob": float(chosen_prob[index, 0]), "chosen_rank": int(ranks[index]),
            "candidates_kept": int(kept.get(row, 1)),
        })
