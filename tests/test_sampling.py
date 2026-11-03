import pytest
import torch

from minillm_l4.engine.sampling import RequestSampler, SamplingParams, apply_repetition_penalty, select_tokens


def sampler(**params):
    return RequestSampler(SamplingParams(**params))


def draw_many(logits, n=400, **params):
    s = sampler(**params)
    return [int(select_tokens(logits[None], [s], [()])) for _ in range(n)]


LOGITS = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0])


def test_no_sampler_is_argmax():
    logits = torch.randn(4, 50)
    assert torch.equal(select_tokens(logits, [None] * 4, [()] * 4), logits.argmax(-1, keepdim=True))


def test_temperature_zero_is_greedy_and_consumes_no_randomness():
    s = sampler(temperature=0.0, seed=5)
    before = s.random.getstate()
    assert int(select_tokens(LOGITS[None], [s], [()])) == 0
    assert s.random.getstate() == before


def test_same_seed_repeats_and_different_seed_differs():
    a = draw_many(LOGITS, temperature=1.0, seed=1)
    assert a == draw_many(LOGITS, temperature=1.0, seed=1)
    assert a != draw_many(LOGITS, temperature=1.0, seed=2)


def test_row_result_does_not_depend_on_batch_neighbours():
    alone = sampler(temperature=1.0, seed=9)
    together = sampler(temperature=1.0, seed=9)
    other = sampler(temperature=0.5, top_p=0.5, seed=3)
    noise = torch.randn(1, 5)
    for _ in range(50):
        a = select_tokens(LOGITS[None], [alone], [()])
        b = select_tokens(torch.cat([noise, LOGITS[None]]), [other, together], [(), ()])
        assert int(a) == int(b[1])


def test_top_k_keeps_only_k_most_likely():
    assert set(draw_many(LOGITS, temperature=5.0, top_k=2)) == {0, 1}


def test_top_p_keeps_the_smallest_set_reaching_p():
    logits = torch.tensor([0.5, 0.3, 0.15, 0.05]).log()
    assert set(draw_many(logits, temperature=1.0, top_p=0.7)) == {0, 1}
    assert set(draw_many(logits, temperature=1.0, top_p=0.01)) == {0}


def test_min_p_drops_tokens_far_below_the_best():
    logits = torch.tensor([0.5, 0.3, 0.15, 0.05]).log()
    assert set(draw_many(logits, temperature=1.0, min_p=0.5)) == {0, 1}


def test_low_temperature_concentrates_and_frequencies_match_probabilities():
    assert set(draw_many(LOGITS, temperature=0.01)) == {0}
    logits = torch.tensor([0.75, 0.25]).log()
    picks = draw_many(logits, n=4000, temperature=1.0, seed=0)
    assert picks.count(0) / 4000 == pytest.approx(0.75, abs=0.03)


def test_repetition_penalty_handles_both_signs():
    out = apply_repetition_penalty(torch.tensor([2.0, -2.0, 1.0]), [0, 1, 0], 2.0)
    assert out.tolist() == [1.0, -4.0, 1.0]


def test_repetition_penalty_changes_greedy_choice():
    s = sampler(temperature=0.0, repetition_penalty=3.0)
    assert int(select_tokens(LOGITS[None], [s], [(0,)])) == 1


def test_mixed_rows_keep_greedy_rows_exact_and_stats_are_recorded():
    logits = torch.randn(3, 100)
    s = RequestSampler(SamplingParams(temperature=1.0, top_k=10, seed=1), record_stats=True)
    tokens = select_tokens(logits, [None, s, None], [(), (), ()])
    assert int(tokens[0]) == int(logits[0].argmax()) and int(tokens[2]) == int(logits[2].argmax())
    stats = s.stats[0]
    assert stats["candidates_kept"] == 10 and stats["chosen_rank"] < 10
    assert 0 < stats["entropy_bits"] <= torch.log2(torch.tensor(100.0)) + 1e-3


@pytest.mark.parametrize("bad", [{"temperature": -1.0}, {"top_p": 0.0}, {"min_p": 1.0}, {"top_k": -1},
                                 {"repetition_penalty": 0.0}])
def test_invalid_settings_raise(bad):
    with pytest.raises(ValueError):
        SamplingParams(**bad)
