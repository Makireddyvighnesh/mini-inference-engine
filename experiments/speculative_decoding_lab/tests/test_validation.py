import pytest

from speculative_decoding_lab.validation import _answer_is_correct, _answer_text, _permutation_p_value


def test_answer_text_removes_qwen_thinking_wrapper():
    assert _answer_text("<think>\n\n</think>\n\nOK") == "OK"
    assert _answer_text("OK") == "OK"
    assert _answer_is_correct("<think>\n\n</think>\n\n12", "12")
    assert _answer_is_correct('<think>\n\n</think>\n\n{"ok": true}', {"ok": True})
    assert not _answer_is_correct("not json", {"ok": True})
    assert not _answer_is_correct('{"ok": 1}', {"ok": True})


def test_permutation_test_rejects_completely_disjoint_samples():
    distance, p_value = _permutation_p_value([1] * 32, [2] * 32, permutations=499)
    assert distance == 1.0
    assert p_value < 0.01


def test_permutation_test_accepts_identical_empirical_distributions():
    distance, p_value = _permutation_p_value([1, 2] * 16, [2, 1] * 16, permutations=499)
    assert distance == 0.0
    assert p_value == 1.0


def test_permutation_test_requires_equal_nonempty_samples():
    with pytest.raises(ValueError):
        _permutation_p_value([1], [])
