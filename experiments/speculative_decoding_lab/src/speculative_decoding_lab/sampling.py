"""Reference implementation of exact stochastic speculative verification.

The live benchmark delegates model execution and KV rollback to llama.cpp.
This module isolates the probability correction used by ordinary speculative
sampling so it can be understood and tested without loading either model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SampledVerification:
    committed_tokens: tuple[int, ...]
    accepted_draft_tokens: int
    rejected_at: int | None
    used_bonus_token: bool


def _distribution(values: Sequence[float], label: str) -> list[float]:
    if not values:
        raise ValueError(f"{label} distribution cannot be empty")
    probabilities = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
        raise ValueError(f"{label} probabilities must be finite and non-negative")
    total = math.fsum(probabilities)
    if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError(f"{label} probabilities must sum to 1 (got {total})")
    return [value / total for value in probabilities]


def _validate_uniform(value: float, label: str) -> float:
    uniform = float(value)
    if not math.isfinite(uniform) or not 0.0 <= uniform < 1.0:
        raise ValueError(f"{label} must be in [0, 1)")
    return uniform


def _sample(distribution: Sequence[float], uniform: float) -> int:
    threshold = _validate_uniform(uniform, "sampling uniform")
    cumulative = 0.0
    for token_id, probability in enumerate(distribution):
        cumulative += probability
        if threshold < cumulative:
            return token_id
    # Floating-point summation can leave cumulative a few ULPs below one.
    return len(distribution) - 1


def verify_sampled_drafts(
    draft_tokens: Sequence[int],
    draft_distributions: Sequence[Sequence[float]],
    target_distributions: Sequence[Sequence[float]],
    accept_uniforms: Sequence[float],
    correction_uniforms: Sequence[float],
    bonus_uniform: float,
) -> SampledVerification:
    """Verify draft tokens while preserving the target sampling distribution.

    For a proposal ``x`` sampled from draft distribution ``q``, accept with
    probability ``min(1, p(x) / q(x))``. On rejection, sample from the
    normalized positive residual ``max(0, p - q)``. If all proposals are
    accepted, sample one bonus token from the target distribution after the
    draft block.

    There must be one target distribution and one draft distribution per
    proposed token, plus a final target distribution for the bonus token.
    The probability vectors use token IDs as their indexes.
    """
    draft_count = len(draft_tokens)
    if len(draft_distributions) != draft_count:
        raise ValueError("one draft distribution is required per proposed token")
    if len(target_distributions) != draft_count + 1:
        raise ValueError("target distributions must include one bonus-token distribution")
    if len(accept_uniforms) != draft_count or len(correction_uniforms) != draft_count:
        raise ValueError("one acceptance and correction uniform is required per draft token")

    accepted: list[int] = []
    vocab_size: int | None = None
    for index, token_id in enumerate(draft_tokens):
        draft_probs = _distribution(draft_distributions[index], f"draft[{index}]")
        target_probs = _distribution(target_distributions[index], f"target[{index}]")
        if len(draft_probs) != len(target_probs):
            raise ValueError(f"draft[{index}] and target[{index}] vocab sizes differ")
        if vocab_size is None:
            vocab_size = len(target_probs)
        elif len(target_probs) != vocab_size:
            raise ValueError("all target and draft distributions must use one vocabulary")
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError(f"draft token {token_id!r} must be an integer token ID")
        if not 0 <= token_id < len(draft_probs):
            raise ValueError(f"draft token {token_id} is outside the vocabulary")

        proposal_probability = draft_probs[token_id]
        if proposal_probability <= 0.0:
            raise ValueError("a proposed token must have positive probability under the draft")
        accept_probability = min(
            1.0, target_probs[token_id] / proposal_probability
        )
        if _validate_uniform(accept_uniforms[index], "acceptance uniform") < accept_probability:
            accepted.append(token_id)
            continue

        residual = [
            max(target_probability - draft_probability, 0.0)
            for target_probability, draft_probability in zip(target_probs, draft_probs)
        ]
        residual_mass = math.fsum(residual)
        if residual_mass <= 0.0:
            raise ArithmeticError("rejection occurred but the target-minus-draft residual is empty")
        residual = [probability / residual_mass for probability in residual]
        correction_token = _sample(residual, correction_uniforms[index])
        return SampledVerification(
            committed_tokens=tuple([*accepted, correction_token]),
            accepted_draft_tokens=len(accepted),
            rejected_at=index,
            used_bonus_token=False,
        )

    if vocab_size is None:
        bonus_probs = _distribution(target_distributions[0], "target bonus")
    else:
        bonus_probs = _distribution(target_distributions[-1], "target bonus")
        if len(bonus_probs) != vocab_size:
            raise ValueError("bonus target distribution has a different vocabulary size")
    bonus_token = _sample(bonus_probs, bonus_uniform)
    return SampledVerification(
        committed_tokens=tuple([*accepted, bonus_token]),
        accepted_draft_tokens=len(accepted),
        rejected_at=None,
        used_bonus_token=True,
    )
