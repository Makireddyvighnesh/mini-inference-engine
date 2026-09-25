import random

import pytest

from speculative_decoding_lab.sampling import verify_sampled_drafts


def test_rejection_samples_from_positive_target_minus_draft_residual():
    result = verify_sampled_drafts(
        draft_tokens=[0],
        draft_distributions=[[0.8, 0.2]],
        target_distributions=[[0.2, 0.8], [0.2, 0.8]],
        accept_uniforms=[0.5],  # acceptance probability for token 0 is 0.25
        correction_uniforms=[0.5],
        bonus_uniform=0.5,
    )
    assert result.committed_tokens == (1,)
    assert result.accepted_draft_tokens == 0
    assert result.rejected_at == 0
    assert not result.used_bonus_token


def test_accepted_proposal_is_followed_by_a_target_bonus_token():
    result = verify_sampled_drafts(
        draft_tokens=[0],
        draft_distributions=[[0.2, 0.8]],
        target_distributions=[[0.8, 0.2], [0.25, 0.75]],
        accept_uniforms=[0.99],  # p(0) / q(0) > 1, so accept with probability 1
        correction_uniforms=[0.0],
        bonus_uniform=0.3,
    )
    assert result.committed_tokens == (0, 1)
    assert result.accepted_draft_tokens == 1
    assert result.rejected_at is None
    assert result.used_bonus_token


def test_first_emitted_token_matches_target_distribution_in_expectation():
    rng = random.Random(7231)
    draft_probs = [0.8, 0.2]
    target_probs = [0.2, 0.8]
    counts = [0, 0]
    samples = 20_000

    for _ in range(samples):
        proposed = rng.choices([0, 1], weights=draft_probs, k=1)[0]
        result = verify_sampled_drafts(
            draft_tokens=[proposed],
            draft_distributions=[draft_probs],
            target_distributions=[target_probs, target_probs],
            accept_uniforms=[rng.random()],
            correction_uniforms=[rng.random()],
            bonus_uniform=rng.random(),
        )
        counts[result.committed_tokens[0]] += 1

    observed = [count / samples for count in counts]
    assert observed[0] == pytest.approx(target_probs[0], abs=0.015)
    assert observed[1] == pytest.approx(target_probs[1], abs=0.015)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"draft_tokens": [0], "draft_distributions": [], "target_distributions": [[1.0], [1.0]], "accept_uniforms": [0.0], "correction_uniforms": [0.0], "bonus_uniform": 0.0},
        {"draft_tokens": [], "draft_distributions": [], "target_distributions": [], "accept_uniforms": [], "correction_uniforms": [], "bonus_uniform": 0.0},
        {"draft_tokens": [1], "draft_distributions": [[1.0, 0.0]], "target_distributions": [[0.5, 0.5], [0.5, 0.5]], "accept_uniforms": [0.0], "correction_uniforms": [0.0], "bonus_uniform": 0.0},
        {"draft_tokens": [0.9], "draft_distributions": [[1.0]], "target_distributions": [[1.0], [1.0]], "accept_uniforms": [0.0], "correction_uniforms": [0.0], "bonus_uniform": 0.0},
        {"draft_tokens": [True], "draft_distributions": [[0.5, 0.5]], "target_distributions": [[0.5, 0.5], [0.5, 0.5]], "accept_uniforms": [0.0], "correction_uniforms": [0.0], "bonus_uniform": 0.0},
    ],
)
def test_invalid_inputs_are_rejected(kwargs):
    with pytest.raises((ValueError, ArithmeticError)):
        verify_sampled_drafts(**kwargs)
