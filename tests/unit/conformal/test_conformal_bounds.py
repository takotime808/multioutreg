# Copyright (c) 2026 takotime808
"""Bounds, invariants and failure points of the conformal predictors and metrics."""

import warnings

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from multioutreg.conformal import CVPlusConformalPredictor, SplitConformalPredictor
from multioutreg.conformal.base import BaseConformalPredictor
from multioutreg.conformal.metrics import (
    conditional_coverage,
    conformal_coverage,
    conformal_interval_width,
    conformal_summary,
)
from multioutreg.conformal.utils import (
    absolute_residual_score,
    intervals_from_scores,
    normalized_residual_score,
)

quantile = BaseConformalPredictor._conformal_quantile


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(200, 2))
    y = X @ np.array([1.0, -2.0]) + 0.1 * rng.normal(size=200)
    return X, y


# ---------------------------------------------------------------------------
# Nonconformity scores
# ---------------------------------------------------------------------------


def test_absolute_residual_score_is_non_negative_and_symmetric():
    y, yhat = np.array([1.0, -2.0, 3.0]), np.array([0.0, 2.0, 3.0])
    s = absolute_residual_score(y, yhat)
    np.testing.assert_array_equal(s, [1.0, 4.0, 0.0])
    np.testing.assert_array_equal(s, absolute_residual_score(yhat, y))


def test_normalized_score_clips_zero_std_instead_of_dividing_by_zero():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no divide-by-zero RuntimeWarning
        s = normalized_residual_score(np.array([1.0]), np.array([0.0]), np.array([0.0]))
    assert np.isfinite(s).all()
    assert s[0] == pytest.approx(1e12)


def test_normalized_score_negative_std_is_clipped_to_tiny_positive():
    s = normalized_residual_score(np.array([1.0]), np.array([0.0]), np.array([-5.0]))
    assert s[0] > 0


@pytest.mark.parametrize("q", [0.0, 0.5, 3.0])
def test_intervals_from_scores_are_centred_and_ordered(q):
    y_pred = np.array([-1.0, 0.0, 2.0])
    lo, hi = intervals_from_scores(y_pred, q)
    np.testing.assert_allclose((lo + hi) / 2, y_pred)
    assert np.all(lo <= hi)
    np.testing.assert_allclose(hi - lo, 2 * q)


def test_intervals_from_scores_adaptive_width_scales_with_std():
    y_pred, y_std = np.zeros(3), np.array([0.0, 1.0, 2.0])
    lo, hi = intervals_from_scores(y_pred, 1.5, y_std=y_std)
    np.testing.assert_allclose(hi - lo, 2 * 1.5 * y_std)


def test_intervals_from_infinite_quantile_are_unbounded():
    lo, hi = intervals_from_scores(np.zeros(2), np.inf)
    assert np.all(np.isneginf(lo)) and np.all(np.isposinf(hi))


# ---------------------------------------------------------------------------
# Conformal quantile: bounds in alpha
# ---------------------------------------------------------------------------


def test_conformal_quantile_is_monotone_non_increasing_in_alpha():
    scores = np.random.default_rng(1).exponential(size=50)
    alphas = np.linspace(0.02, 0.98, 49)
    qs = [quantile(scores, a) for a in alphas]
    assert all(a >= b for a, b in zip(qs, qs[1:]))


def test_conformal_quantile_lies_within_score_range():
    scores = np.random.default_rng(2).normal(size=40) ** 2
    for alpha in (0.05, 0.25, 0.5, 0.9):
        q = quantile(scores, alpha)
        assert scores.min() <= q <= scores.max()


def test_conformal_quantile_ignores_score_order():
    scores = np.arange(1.0, 30.0)
    shuffled = np.random.default_rng(3).permutation(scores)
    assert quantile(scores, 0.2) == quantile(shuffled, 0.2)


def test_conformal_quantile_single_score_is_infinite_for_any_reasonable_alpha():
    with pytest.warns(UserWarning, match="too few"):
        assert quantile(np.array([1.0]), 0.4) == np.inf


def test_conformal_quantile_empty_scores_is_infinite():
    with pytest.warns(UserWarning, match="too few"):
        assert quantile(np.array([]), 0.1) == np.inf


def test_conformal_quantile_near_one_alpha_gives_smallest_score():
    # k = ceil(11 * 0.001) = 1 -> smallest score.
    assert quantile(np.arange(1.0, 11.0), 0.999) == 1.0


def test_conformal_quantile_alpha_one_raises():
    # Regression: k = 0 used to wrap to the largest score (the widest interval).
    with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\)"):
        quantile(np.arange(1.0, 11.0), 1.0)


@pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5])
def test_predictors_reject_alpha_outside_unit_interval(alpha, linear_data):
    X, y = linear_data
    for cls in (SplitConformalPredictor, CVPlusConformalPredictor):
        cp = cls(LinearRegression(), random_state=0).fit(X, y)
        with pytest.raises(ValueError, match="alpha"):
            cp.predict_interval(X[:2], alpha=alpha)


def test_conformal_quantile_alpha_zero_raises_value_error():
    with pytest.raises(ValueError):
        quantile(np.arange(1.0, 11.0), 0.0)


def test_conformal_quantile_alpha_above_one_raises():
    with pytest.raises(ValueError):
        quantile(np.arange(1.0, 11.0), 1.5)


# ---------------------------------------------------------------------------
# Predictors: interval invariants and failure points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [SplitConformalPredictor, CVPlusConformalPredictor])
def test_intervals_widen_as_alpha_shrinks(cls, linear_data):
    X, y = linear_data
    cp = cls(LinearRegression(), random_state=0).fit(X, y)
    widths = []
    for alpha in (0.5, 0.2, 0.1, 0.05):
        lo, hi = cp.predict_interval(X[:20], alpha=alpha)
        assert np.all(lo <= hi)
        widths.append(np.mean(hi - lo))
    assert all(a <= b + 1e-12 for a, b in zip(widths, widths[1:]))


def test_split_interval_is_symmetric_around_point_prediction(linear_data):
    X, y = linear_data
    cp = SplitConformalPredictor(LinearRegression(), random_state=0).fit(X, y)
    y_pred, lo, hi = cp.predict(X[:30], alpha=0.1)
    assert y_pred.shape == lo.shape == hi.shape == (30,)
    np.testing.assert_allclose((lo + hi) / 2, y_pred)


def test_split_predict_interval_on_empty_input_raises(linear_data):
    X, y = linear_data
    cp = SplitConformalPredictor(LinearRegression(), random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        # sklearn refuses zero-sample input rather than returning empty arrays.
        cp.predict_interval(X[:0], alpha=0.1)


def test_split_too_few_calibration_points_gives_unbounded_interval(linear_data):
    X, y = linear_data
    # 10 points * 0.2 = 2 calibration points: far too few for alpha=0.1.
    cp = SplitConformalPredictor(LinearRegression(), random_state=0).fit(X[:10], y[:10])
    with pytest.warns(UserWarning, match="too few"):
        lo, hi = cp.predict_interval(X[:3], alpha=0.1)
    assert np.all(np.isneginf(lo)) and np.all(np.isposinf(hi))


@pytest.mark.parametrize("size", [0.0, 1.0, 1.5])
def test_split_invalid_calibration_size_raises(size, linear_data):
    X, y = linear_data
    with pytest.raises(ValueError):
        SplitConformalPredictor(LinearRegression(), calibration_size=size).fit(X, y)


def test_split_adaptive_without_return_std_raises(linear_data):
    X, y = linear_data
    with pytest.raises(TypeError):
        SplitConformalPredictor(LinearRegression(), adaptive=True).fit(X, y)


def test_split_mismatched_X_y_lengths_raises(linear_data):
    X, y = linear_data
    with pytest.raises(ValueError):
        SplitConformalPredictor(LinearRegression()).fit(X, y[:-5])


def test_predict_before_fit_raises_attribute_error():
    with pytest.raises(AttributeError, match="not fitted"):
        SplitConformalPredictor(LinearRegression()).predict(np.zeros((2, 2)))


@pytest.mark.parametrize("n_folds", [0, 1])
def test_cv_plus_fewer_than_two_folds_raises(n_folds, linear_data):
    X, y = linear_data
    with pytest.raises(ValueError):
        CVPlusConformalPredictor(LinearRegression(), n_folds=n_folds).fit(X, y)


def test_cv_plus_more_folds_than_samples_raises(linear_data):
    X, y = linear_data
    with pytest.raises(ValueError):
        CVPlusConformalPredictor(LinearRegression(), n_folds=20).fit(X[:5], y[:5])


def test_cv_plus_leave_one_out_assigns_every_point_its_own_fold(linear_data):
    X, y = linear_data
    n = 12
    cp = CVPlusConformalPredictor(LinearRegression(), n_folds=n, random_state=0).fit(X[:n], y[:n])
    assert sorted(cp.fold_ids_) == list(range(n))
    assert np.all(np.isfinite(cp.residuals_))


def test_cv_plus_high_alpha_lower_bound_finite_upper_bound_finite(linear_data):
    X, y = linear_data
    cp = CVPlusConformalPredictor(LinearRegression(), random_state=0).fit(X[:40], y[:40])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lo, hi = cp.predict_interval(X[:5], alpha=0.5)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))


# ---------------------------------------------------------------------------
# Metrics: bounds and degenerate input
# ---------------------------------------------------------------------------


def test_coverage_is_a_fraction_in_unit_interval():
    rng = np.random.default_rng(4)
    y = rng.normal(size=(100, 3))
    lo, hi = y - rng.uniform(-1, 1, size=y.shape), y + rng.uniform(-1, 1, size=y.shape)
    cov = conformal_coverage(y, lo, hi)["coverage"]
    assert ((cov >= 0) & (cov <= 1)).all()


def test_coverage_counts_points_on_the_boundary_as_covered():
    y = np.array([0.0, 1.0])
    df = conformal_coverage(y, y, y)  # zero-width intervals exactly at y
    assert df["coverage"].iloc[0] == 1.0


def test_coverage_with_nan_truth_counts_as_not_covered():
    y = np.array([np.nan, 0.0])
    df = conformal_coverage(y, np.full(2, -1.0), np.full(2, 1.0))
    assert df["coverage"].iloc[0] == 0.5


def test_coverage_with_infinite_bounds_is_full():
    y = np.random.default_rng(5).normal(size=10)
    df = conformal_coverage(y, np.full(10, -np.inf), np.full(10, np.inf))
    assert df["coverage"].iloc[0] == 1.0


def test_inverted_intervals_report_negative_width_and_zero_coverage():
    y = np.zeros(4)
    lo, hi = np.full(4, 1.0), np.full(4, -1.0)
    assert conformal_coverage(y, lo, hi)["coverage"].iloc[0] == 0.0
    assert conformal_interval_width(lo, hi)["mean_width"].iloc[0] == -2.0


def test_output_names_shorter_than_outputs_raises():
    y = np.zeros((5, 2))
    with pytest.raises(IndexError):
        conformal_coverage(y, y, y, output_names=["only one"])


def test_summary_coverage_gap_sign():
    y = np.zeros(10)
    full = conformal_summary(y, y - 1, y + 1, alpha=0.1)
    none = conformal_summary(y, y + 1, y + 2, alpha=0.1)
    assert full["coverage_gap"].iloc[0] == pytest.approx(0.1)
    assert none["coverage_gap"].iloc[0] == pytest.approx(-0.9)


def test_conditional_coverage_constant_target_puts_everything_in_last_bin():
    y = np.ones(5)
    centers, cov = conditional_coverage(y, y - 1, y + 1, n_bins=3)
    assert centers.shape == cov.shape == (3,)
    assert np.isnan(cov[:2]).all() and cov[2] == 1.0


def test_conditional_coverage_values_are_nan_or_fraction():
    rng = np.random.default_rng(6)
    y = rng.normal(size=200)
    _, cov = conditional_coverage(y, y - rng.uniform(0, 1, 200), y + rng.uniform(-1, 1, 200), n_bins=8)
    finite = cov[~np.isnan(cov)]
    assert ((finite >= 0) & (finite <= 1)).all()


def test_conditional_coverage_zero_bins_returns_empty():
    centers, cov = conditional_coverage(np.arange(5.0), np.zeros(5), np.ones(5), n_bins=0)
    assert centers.size == 0 and cov.size == 0


def test_conditional_coverage_empty_input_raises():
    with pytest.raises(ValueError):
        conditional_coverage(np.array([]), np.array([]), np.array([]))
