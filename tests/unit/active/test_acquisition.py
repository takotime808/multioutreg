# Copyright (c) 2026 takotime808

import numpy as np
import pytest
from scipy.stats import norm

from multioutreg.active.acquisition import (
    CommitteeSurrogate,
    aggregate,
    expected_improvement,
    max_variance,
    output_scale,
    predict_mean_std,
    probability_of_improvement,
    upper_confidence_bound,
)
from multioutreg.surrogates import KNeighborsSurrogate, LinearRegressionSurrogate


def test_expected_improvement_closed_form():
    mean = np.array([[1.0], [0.0]])
    std = np.array([[1.0], [0.0]])
    ei = expected_improvement(mean, std, best=0.5)
    z = (0.5 - 1.0) / 1.0
    assert ei[0] == pytest.approx(-0.5 * norm.cdf(z) + norm.pdf(z))
    # Zero std: EI is the plain improvement.
    assert ei[1] == pytest.approx(0.5)


def test_expected_improvement_maximize_mirrors_minimize():
    mean = np.array([[0.2], [1.5]])
    std = np.array([[0.3], [0.7]])
    lo = expected_improvement(mean, std, best=1.0, minimize=True)
    hi = expected_improvement(-mean, std, best=-1.0, minimize=False)
    np.testing.assert_allclose(lo, hi)


def test_probability_of_improvement_bounds_and_zero_std():
    mean = np.array([[0.0], [2.0], [0.0], [2.0]])
    std = np.array([[1.0], [1.0], [0.0], [0.0]])
    pi = probability_of_improvement(mean, std, best=1.0)
    assert np.all((pi >= 0) & (pi <= 1))
    assert pi[0] > 0.5 > pi[1]
    np.testing.assert_array_equal(pi[2:], [1.0, 0.0])


def test_ucb_sign_follows_direction():
    mean = np.array([[1.0], [2.0]])
    std = np.zeros((2, 1))
    assert np.argmax(upper_confidence_bound(mean, std, minimize=True)) == 0
    assert np.argmax(upper_confidence_bound(mean, std, minimize=False)) == 1


def test_scalarized_std_combines_outputs_in_quadrature():
    mean = np.zeros((1, 2))
    std = np.array([[3.0, 4.0]])
    assert upper_confidence_bound(mean, std, beta=1.0)[0] == pytest.approx(5.0)
    with pytest.raises(ValueError):
        upper_confidence_bound(mean, std, weights=[1.0])


def test_max_variance_is_invariant_to_output_scale():
    rng = np.random.default_rng(0)
    std = rng.random((20, 2))
    Y = rng.standard_normal((30, 2))
    base = max_variance(None, std, y_scale=output_scale(Y))
    scale = np.array([1.0, 1000.0])
    scaled = max_variance(None, std * scale, y_scale=output_scale(Y * scale))
    np.testing.assert_allclose(base, scaled)


def test_aggregate_modes():
    v = np.array([[1.0, 3.0], [2.0, 2.0]])
    np.testing.assert_allclose(aggregate(v, "sum"), [4, 4])
    np.testing.assert_allclose(aggregate(v, "mean"), [2, 2])
    np.testing.assert_allclose(aggregate(v, "max"), [3, 2])
    np.testing.assert_allclose(aggregate(v, "sum", weights=[1, 0]), [1, 2])
    with pytest.raises(ValueError):
        aggregate(v, "median")


def test_output_scale_handles_constant_outputs():
    Y = np.column_stack([np.ones(5), np.arange(5.0)])
    s = output_scale(Y)
    assert s[0] == 1.0 and s[1] == pytest.approx(np.arange(5.0).std())


def test_predict_mean_std_rejects_zero_uncertainty():
    rng = np.random.default_rng(0)
    X, Y = rng.random((20, 2)), rng.random((20, 2))
    model = LinearRegressionSurrogate().fit(X, Y)
    with pytest.raises(ValueError, match="CommitteeSurrogate"):
        predict_mean_std(model, X)
    mean, std = predict_mean_std(model, X, allow_zero_std=True)
    assert mean.shape == std.shape == (20, 2)


def test_committee_gives_nonzero_std():
    rng = np.random.default_rng(0)
    X = rng.random((40, 2))
    Y = np.column_stack([np.sin(4 * X[:, 0]), X[:, 1] ** 2])
    committee = CommitteeSurrogate(KNeighborsSurrogate, n_members=4).fit(X, Y)
    mean, std = predict_mean_std(committee, rng.random((10, 2)))
    assert mean.shape == std.shape == (10, 2)
    assert np.all(std.max(axis=0) > 0)
    with pytest.raises(ValueError):
        CommitteeSurrogate(KNeighborsSurrogate, n_members=1)
