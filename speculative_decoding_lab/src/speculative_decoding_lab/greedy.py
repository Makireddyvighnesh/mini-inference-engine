"""Small, model-independent reference for greedy speculative verification.

This makes the accept/reject rule easy to inspect and test. The live benchmark
uses llama.cpp's ``draft-simple`` implementation; this module is not in its
inference path.
"""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class GreedyVerification:
    committed_tokens: tuple[int, ...]
    accepted_draft_tokens: int
    rejected_at: int | None
    used_bonus_token: bool


def verify_greedy_drafts(
    draft_tokens: Sequence[int],
    target_greedy_tokens: Sequence[int],
    bonus_token: int,
) -> GreedyVerification:
    """Accept matching draft tokens; on mismatch commit the target token.

    ``target_greedy_tokens[i]`` is the target model's argmax for the position
    proposed by ``draft_tokens[i]``. Those target positions can be evaluated in
    one causal block forward pass. If all proposals match, ``bonus_token`` is
    the target's next token after the accepted block.

    This is specifically the greedy rule. Exact stochastic speculative
    sampling needs probability-based acceptance and residual sampling instead.
    """
    if len(draft_tokens) != len(target_greedy_tokens):
        raise ValueError("one target verification token is required per draft token")

    accepted: list[int] = []
    for index, (draft_token, target_token) in enumerate(
        zip(draft_tokens, target_greedy_tokens)
    ):
        if draft_token != target_token:
            return GreedyVerification(
                committed_tokens=tuple([*accepted, target_token]),
                accepted_draft_tokens=len(accepted),
                rejected_at=index,
                used_bonus_token=False,
            )
        accepted.append(draft_token)

    return GreedyVerification(
        committed_tokens=tuple([*accepted, bonus_token]),
        accepted_draft_tokens=len(accepted),
        rejected_at=None,
        used_bonus_token=True,
    )
