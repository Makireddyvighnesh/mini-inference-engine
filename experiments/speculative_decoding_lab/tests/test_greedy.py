import pytest

from speculative_decoding_lab.greedy import verify_greedy_drafts


def test_all_drafts_accepted_adds_target_bonus_token():
    result = verify_greedy_drafts([11, 12, 13], [11, 12, 13], bonus_token=14)
    assert result.committed_tokens == (11, 12, 13, 14)
    assert result.accepted_draft_tokens == 3
    assert result.rejected_at is None
    assert result.used_bonus_token


def test_first_mismatch_discards_draft_and_commits_target_token():
    result = verify_greedy_drafts([11, 99, 13], [11, 12, 13], bonus_token=14)
    assert result.committed_tokens == (11, 12)
    assert result.accepted_draft_tokens == 1
    assert result.rejected_at == 1
    assert not result.used_bonus_token


def test_empty_draft_round_emits_bonus_token():
    result = verify_greedy_drafts([], [], bonus_token=7)
    assert result.committed_tokens == (7,)
    assert result.accepted_draft_tokens == 0


def test_requires_one_target_check_per_draft_token():
    with pytest.raises(ValueError):
        verify_greedy_drafts([1, 2], [1], bonus_token=3)
