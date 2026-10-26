# Copyright (c) 2026 takotime808
"""Bounds, invariants and failure points of acquisition functions and batch selection."""

import numpy as np
import pytest

from multioutreg.active.acquisition import (
    ACQUISITIONS,
    CommitteeSurrogate,
    aggregate,
    expected_improvement,
    max_variance,
    output_scale,
    predict_mean_std,
    probability_of_improvement,
    upper_confidence_bound,
)
from multioutreg.active.batch import default_length_scale, select_batch
from multioutreg.surrogates import LinearRegressionSurrogate


@pytest.fixture
def mean_std():
    rng = np.random.default_rng(0)
    return rng.normal(size=(50, 2)), rng.uniform(0.0, 2.0, size=(50, 2))


# ---------------------------------------------------------------------------
# Acquisition bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("minimize", [True, False])
def test_ei_is_non_negative_and_at_least_plain_improvement(mean_std, minimize):
    mean, std = mean_std
    best = 0.0
    ei = expected_improvement(mean, std, best=best, minimize=minimize)
    mu = mean.sum(axis=1)
    plain = np.maximum(best - mu if minimize else mu - best, 0.0)
    assert np.all(ei >= 0)
    assert np.all(ei >= plain - 1e-12)


def test_ei_grows_with_std_at_fixed_mean():
    mean = np.zeros((4, 1))
    std = np.array([[0.1], [0.5], [1.0], [2.0]])
    ei = expected_improvement(mean, std, best=0.0)
    assert np.all(np.diff(ei) > 0)


def test_ei_decreases_with_larger_xi(mean_std):
    mean, std = mean_std
    lo = expected_improvement(mean, std, best=0.0, xi=0.0)
    hi = expected_improvement(mean, std, best=0.0, xi=1.0)
    assert np.all(hi <= lo + 1e-12)


@pytest.mark.parametrize("minimize", [True, False])
def test_pi_is_a_probability(mean_std, minimize):
    mean, std = mean_std
    pi = probability_of_improvement(mean, std, best=0.3, minimize=minimize)
    assert np.all((pi >= 0) & (pi <= 1))


def test_pi_minimize_and_maximize_are_complementary():
    mean = np.array([[0.2], [-1.0], [3.0]])
    std = np.array([[1.0], [0.5], [2.0]])
    lo = probability_of_improvement(mean, std, best=0.0, minimize=True)
    hi = probability_of_improvement(mean, std, best=0.0, minimize=False)
    np.testing.assert_allclose(lo + hi, 1.0)


def test_pi_zero_std_at_exactly_best_is_zero():
    # Improvement must be strictly positive.
    assert probability_of_improvement(np.zeros((1, 1)), np.zeros((1, 1)), best=0.0)[0] == 0.0


def test_ucb_beta_zero_is_negated_mean():
    mean = np.array([[1.0], [-2.0]])
    np.testing.assert_allclose(upper_confidence_bound(mean, np.ones((2, 1)), beta=0.0), [-1.0, 2.0])


def test_ucb_is_non_decreasing_in_beta(mean_std):
    mean, std = mean_std
    vals = [upper_confidence_bound(mean, std, beta=b) for b in (0.0, 1.0, 4.0, 16.0)]
    for a, b in zip(vals, vals[1:]):
        assert np.all(b >= a - 1e-12)


def test_ucb_negative_beta_raises():
    with pytest.raises(ValueError):
        upper_confidence_bound(np.zeros((2, 1)), np.ones((2, 1)), beta=-1.0)


def test_max_variance_is_non_negative_and_zero_only_for_zero_std():
    std = np.array([[0.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
    mv = max_variance(np.zeros_like(std), std)
    np.testing.assert_allclose(mv, [0.0, 1.0, 4.0])


def test_max_variance_weights_scale_each_output():
    std = np.array([[1.0, 1.0]])
    assert max_variance(None, std, weights=[2.0, 0.0])[0] == pytest.approx(2.0)


def test_weights_wrong_length_raises_for_scalarized_acquisitions():
    mean, std = np.zeros((3, 2)), np.ones((3, 2))
    for fn in (upper_confidence_bound, lambda m, s, weights: expected_improvement(m, s, 0.0, weights=weights)):
        with pytest.raises(ValueError, match="weights"):
            fn(mean, std, weights=[1.0, 1.0, 1.0])


def test_1d_inputs_are_treated_as_single_output():
    ei_1d = expected_improvement(np.array([0.0, 1.0]), np.array([1.0, 1.0]), best=0.5)
    ei_2d = expected_improvement(np.array([[0.0], [1.0]]), np.array([[1.0], [1.0]]), best=0.5)
    np.testing.assert_allclose(ei_1d, ei_2d)


def test_acquisition_registry_maps_to_callables():
    assert set(ACQUISITIONS) == {"max_variance", "ucb", "ei", "pi"}
    assert all(callable(f) for f in ACQUISITIONS.values())


def test_aggregate_max_dominates_mean_and_sum_dominates_max_for_non_negative():
    v = np.abs(np.random.default_rng(1).normal(size=(10, 3)))
    assert np.all(aggregate(v, "max") >= aggregate(v, "mean"))
    assert np.all(aggregate(v, "sum") >= aggregate(v, "max"))


def test_output_scale_is_positive_even_for_single_row():
    np.testing.assert_array_equal(output_scale(np.array([[1.0, 2.0]])), [1.0, 1.0])


# ---------------------------------------------------------------------------
# Uncertainty extraction failure points
# ---------------------------------------------------------------------------


class _ZeroStd:
    def predict(self, X, return_std=False):
        n = len(X)
        return np.zeros((n, 2)), np.column_stack([np.ones(n), np.zeros(n)])


def test_predict_mean_std_names_only_the_dead_output():
    with pytest.raises(ValueError, match=r"\[1\]"):
        predict_mean_std(_ZeroStd(), np.zeros((3, 1)))


def test_predict_mean_std_single_row_is_not_checked():
    # One candidate cannot show "zero everywhere", so it is allowed.
    _, std = predict_mean_std(_ZeroStd(), np.zeros((1, 1)))
    assert std.shape == (1, 2)


def test_predict_mean_std_allow_zero_std_opt_out():
    _, std = predict_mean_std(_ZeroStd(), np.zeros((3, 1)), allow_zero_std=True)
    assert np.all(std[:, 1] == 0)


@pytest.mark.parametrize("n", [0, 1, -3])
def test_committee_rejects_fewer_than_two_members(n):
    with pytest.raises(ValueError, match="at least 2"):
        CommitteeSurrogate(LinearRegressionSurrogate, n_members=n)


def test_committee_is_reproducible_with_fixed_seed():
    rng = np.random.default_rng(2)
    X, Y = rng.uniform(size=(30, 2)), rng.normal(size=(30, 2))
    a = CommitteeSurrogate(LinearRegressionSurrogate, 3, random_state=7).fit(X, Y)
    b = CommitteeSurrogate(LinearRegressionSurrogate, 3, random_state=7).fit(X, Y)
    np.testing.assert_allclose(a.predict(X, return_std=True)[1], b.predict(X, return_std=True)[1])


def test_committee_predict_before_fit_raises():
    with pytest.raises(AttributeError):
        CommitteeSurrogate(LinearRegressionSurrogate).predict(np.zeros((2, 2)))


# ---------------------------------------------------------------------------
# Batch selection
# ---------------------------------------------------------------------------


def test_default_length_scale_bounds():
    assert default_length_scale(1, 3) == 1.0
    assert default_length_scale(0, 3) == 1.0  # clamped to one point
    assert 0 < default_length_scale(1000, 2) < default_length_scale(10, 2) <= 1.0


def test_batch_size_equal_to_pool_returns_every_candidate_once():
    U = np.random.default_rng(3).uniform(size=(6, 2))
    idx = select_batch(U, np.arange(6.0), batch_size=6)
    assert sorted(idx.tolist()) == list(range(6))


def test_batch_size_zero_returns_empty():
    assert select_batch(np.zeros((4, 2)), np.ones(4), batch_size=0).size == 0


def test_batch_indices_are_unique_and_in_range():
    rng = np.random.default_rng(4)
    U = rng.uniform(size=(40, 3))
    idx = select_batch(U, rng.normal(size=40), batch_size=15)
    assert len(set(idx.tolist())) == 15
    assert idx.min() >= 0 and idx.max() < 40


def test_batch_handles_nan_and_negative_scores():
    U = np.random.default_rng(5).uniform(size=(5, 2))
    scores = np.array([np.nan, -5.0, -1.0, np.nan, -3.0])
    idx = select_batch(U, scores, batch_size=3)
    assert idx[0] == 2  # largest finite score
    assert len(set(idx.tolist())) == 3


def test_batch_duplicate_candidates_are_not_both_picked_first():
    U = np.array([[0.5, 0.5], [0.5, 0.5], [0.0, 0.0]])
    idx = select_batch(U, np.array([1.0, 1.0, 0.5]), batch_size=2)
    # The penalty zeroes the duplicate, so the distinct point comes second.
    assert idx[1] == 2


def test_batch_negative_size_raises():
    with pytest.raises(ValueError):
        select_batch(np.zeros((4, 2)), np.ones(4), batch_size=-1)


def test_batch_score_length_mismatch_raises():
    with pytest.raises(ValueError):
        select_batch(np.zeros((4, 2)), np.ones(3), batch_size=2)
