# Copyright (c) 2026 takotime808
"""Bounds, invariants and failure points of the time-series helpers."""

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

from multioutreg.time_series.chronos_adapter import ForecastResult
from multioutreg.time_series.cv import (
    TimeSeriesSplitWrapper,
    WalkForwardCV,
    walk_forward_splits,
)
from multioutreg.time_series.lag_features import make_lag_features, rolling_window_features
from multioutreg.time_series.metrics import mase, smape, weighted_quantile_loss
from multioutreg.time_series.seasonality import detect_seasonal_period
from multioutreg.time_series.uncertainty import (
    conformal_interval_from_residuals,
    gaussian_quantiles,
    propagate_uncertainty_recursive,
)

# ---------------------------------------------------------------------------
# sMAPE / MASE / WQL
# ---------------------------------------------------------------------------


def test_smape_is_bounded_between_0_and_200():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y, yhat = rng.normal(size=50) * 10, rng.normal(size=50) * 10
        assert 0.0 <= smape(y, yhat) <= 200.0


def test_smape_upper_bound_reached_for_opposite_signs():
    assert smape([1.0, 2.0], [-1.0, -2.0]) == pytest.approx(200.0)


def test_smape_zero_for_perfect_forecast_including_all_zeros():
    assert smape([0.0, 0.0], [0.0, 0.0]) == 0.0
    assert smape([3.0, -4.0], [3.0, -4.0]) == 0.0


def test_smape_is_symmetric_in_its_arguments():
    y, yhat = np.array([1.0, 5.0, 2.0]), np.array([2.0, 3.0, 2.5])
    assert smape(y, yhat) == pytest.approx(smape(yhat, y))


def test_smape_shape_mismatch_raises():
    with pytest.raises(ValueError):
        smape([1.0, 2.0, 3.0], [1.0, 2.0])


def test_mase_is_zero_for_perfect_forecast_and_non_negative():
    train = np.cumsum(np.random.default_rng(1).normal(size=30))
    assert mase([1.0, 2.0], [1.0, 2.0], train) == 0.0
    assert mase([1.0, 2.0], [0.0, 5.0], train) > 0


def test_mase_constant_training_series_does_not_divide_by_zero():
    # Seasonal-naive error is 0; the 1e-12 floor keeps the score finite.
    val = mase([1.0], [2.0], np.ones(10))
    assert np.isfinite(val) and val > 1e9


@pytest.mark.parametrize("m", [0, -1])
def test_mase_rejects_non_positive_seasonality(m):
    with pytest.raises(ValueError, match="seasonality"):
        mase([1.0], [1.0], np.arange(10.0), seasonality=m)


def test_mase_training_series_exactly_one_longer_than_season_is_enough():
    assert np.isfinite(mase([1.0], [1.5], np.arange(5.0), seasonality=4))


def test_wql_is_non_negative_and_zero_iff_all_quantiles_equal_truth():
    y = np.array([1.0, 2.0, 3.0])
    q = (0.1, 0.5, 0.9)
    assert weighted_quantile_loss(y, np.tile(y, (3, 1)), q) == 0.0
    off = weighted_quantile_loss(y, np.tile(y + 1, (3, 1)), q)
    assert off > 0


def test_wql_median_only_equals_normalized_absolute_error():
    # At q=0.5 pinball = |e|/2, so WQL = sum|e| / sum|y|.
    y, yhat = np.array([2.0, 4.0]), np.array([3.0, 1.0])
    expected = np.abs(y - yhat).sum() / np.abs(y).sum()
    assert weighted_quantile_loss(y, yhat[None, :], [0.5]) == pytest.approx(expected)


def test_wql_quantile_count_mismatch_fails():
    with pytest.raises(AssertionError):
        weighted_quantile_loss([1.0, 2.0], [[1.0, 2.0]], [0.1, 0.5])


def test_wql_horizon_mismatch_raises():
    with pytest.raises(ValueError):
        weighted_quantile_loss([1.0, 2.0, 3.0], [[1.0, 2.0]], [0.5])


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------


def test_lag_features_minimum_length_yields_exactly_one_sample():
    X, y = make_lag_features(np.arange(5.0), n_lags=3, horizon=2)
    assert X.shape == (1, 3) and y.shape == (1, 2)
    np.testing.assert_array_equal(X[0], [0, 1, 2])
    np.testing.assert_array_equal(y[0], [3, 4])


def test_lag_features_one_shorter_than_minimum_raises():
    with pytest.raises(ValueError, match="too short"):
        make_lag_features(np.arange(4.0), n_lags=3, horizon=2)


def test_lag_features_error_message_states_true_minimum():
    with pytest.raises(ValueError, match="Need at least 5 observations"):
        make_lag_features(np.arange(4.0), n_lags=3, horizon=2)


def test_lag_features_targets_never_overlap_their_own_inputs():
    series = np.arange(40.0)
    X, y = make_lag_features(series, n_lags=5, horizon=3)
    # Every target index is strictly after every lag index in the same row.
    assert np.all(y.min(axis=1) > X.max(axis=1))


def test_lag_features_time_features_are_bounded_unit_circle():
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    X, _ = make_lag_features(pd.Series(np.arange(60.0), index=idx), 4, include_time_features=True)
    tf = X[:, 4:]
    assert np.all(np.abs(tf) <= 1.0 + 1e-12)
    # sin^2 + cos^2 = 1 for each encoded cycle.
    np.testing.assert_allclose(tf[:, 0] ** 2 + tf[:, 1] ** 2, 1.0)
    np.testing.assert_allclose(tf[:, 2] ** 2 + tf[:, 3] ** 2, 1.0)


def test_lag_features_time_features_ignored_without_datetime_index():
    X, _ = make_lag_features(pd.Series(np.arange(20.0)), 3, include_time_features=True)
    assert X.shape[1] == 3


def test_lag_features_ndarray_never_gets_time_features():
    X, _ = make_lag_features(np.arange(20.0), 3, include_time_features=True)
    assert X.shape[1] == 3


def test_lag_features_zero_lags_raises():
    with pytest.raises(ValueError):
        make_lag_features(np.arange(10.0), n_lags=0)


def test_rolling_features_window_one_is_identity_with_zero_std():
    s = np.array([3.0, -1.0, 4.0])
    out = rolling_window_features(s, windows=(1,))
    np.testing.assert_array_equal(out[:, 0], s)
    np.testing.assert_array_equal(out[:, 1], 0.0)


def test_rolling_features_window_longer_than_series_is_expanding_mean():
    s = np.array([1.0, 2.0, 3.0, 4.0])
    out = rolling_window_features(s, windows=(100,))
    np.testing.assert_allclose(out[:, 0], [1.0, 1.5, 2.0, 2.5])


def test_rolling_features_std_is_non_negative_and_mean_within_range():
    s = np.random.default_rng(2).normal(size=50)
    out = rolling_window_features(s, windows=(3, 7))
    assert np.all(out[:, 1::2] >= 0)
    assert np.all(out[:, 0::2] >= s.min() - 1e-12) and np.all(out[:, 0::2] <= s.max() + 1e-12)


def test_rolling_features_zero_window_raises():
    with pytest.raises(ValueError):
        rolling_window_features(np.arange(5.0), windows=(0,))


# ---------------------------------------------------------------------------
# Uncertainty helpers
# ---------------------------------------------------------------------------


def test_gaussian_quantiles_monotone_in_level():
    q = gaussian_quantiles(np.zeros(4), np.ones(4), quantiles=(0.01, 0.2, 0.5, 0.8, 0.99))
    assert np.all(np.diff(q, axis=0) > 0)


def test_gaussian_quantiles_at_0_and_1_are_infinite():
    q = gaussian_quantiles(np.zeros(2), np.ones(2), quantiles=(0.0, 1.0))
    assert np.all(np.isneginf(q[0])) and np.all(np.isposinf(q[1]))


def test_gaussian_quantiles_outside_unit_interval_are_nan():
    q = gaussian_quantiles(np.zeros(2), np.ones(2), quantiles=(-0.1, 1.1))
    assert np.isnan(q).all()


def test_conformal_interval_from_residuals_monotone_in_alpha():
    resid = np.random.default_rng(3).normal(size=200)
    widths = []
    for alpha in (0.5, 0.2, 0.1, 0.05):
        lo, hi = conformal_interval_from_residuals(np.zeros(1), resid, alpha=alpha)
        widths.append(float(hi[0] - lo[0]))
    assert widths == sorted(widths)


def test_conformal_interval_from_residuals_uses_absolute_value_of_residuals():
    lo1, hi1 = conformal_interval_from_residuals(np.zeros(1), [1.0, 2.0, 3.0] * 10)
    lo2, hi2 = conformal_interval_from_residuals(np.zeros(1), [-1.0, -2.0, -3.0] * 10)
    np.testing.assert_allclose([lo1, hi1], [lo2, hi2])


def test_conformal_interval_from_residuals_empty_calibration_is_unbounded():
    with pytest.warns(UserWarning, match="too few"):
        lo, hi = conformal_interval_from_residuals(np.zeros(3), [])
    assert np.all(np.isneginf(lo)) and np.all(np.isposinf(hi))


def test_conformal_interval_from_residuals_is_kth_order_statistic():
    # n = 19, alpha = 0.1 -> k = ceil(20 * 0.9) = 18 -> 18th smallest |residual|.
    resid = -np.arange(1.0, 20.0)
    lo, hi = conformal_interval_from_residuals(np.zeros(1), resid, alpha=0.1)
    assert hi[0] == 18.0 and lo[0] == -18.0


def test_conformal_interval_from_residuals_rejects_bad_alpha():
    with pytest.raises(ValueError, match="alpha"):
        conformal_interval_from_residuals(np.zeros(1), np.arange(50.0), alpha=0.0)


def test_conformal_interval_from_too_few_residuals_is_unbounded():
    lo, hi = conformal_interval_from_residuals(np.zeros(1), [1.0, 2.0, 3.0], alpha=0.1)
    assert np.isneginf(lo[0]) and np.isposinf(hi[0])


@pytest.mark.parametrize("resid", [[1.0, np.nan, 3.0], [1.0, np.inf], [-np.inf]])
def test_conformal_interval_from_residuals_rejects_non_finite_residuals(resid):
    with pytest.raises(ValueError, match="finite"):
        conformal_interval_from_residuals(np.zeros(1), resid)


@pytest.mark.parametrize("n, alpha", [(4, 0.4), (9, 0.1), (2, 0.1), (19, 0.25)])
def test_conformal_interval_from_residuals_exact_rank_coverage(n, alpha):
    # Hold out each of n + 1 distinct scores in turn: the interval built from
    # the other n must cover the held-out one in at least (1 - alpha) of cases.
    scores = np.arange(n + 1, dtype=float)
    covered = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # k > n gives an unbounded interval
        for i, score in enumerate(scores):
            _, hi = conformal_interval_from_residuals(np.zeros(1), np.delete(scores, i), alpha)
            covered += score <= hi[0]
    assert covered / (n + 1) >= 1 - alpha


def test_propagate_uncertainty_zero_horizon_is_empty():
    assert propagate_uncertainty_recursive(1.0, 0).shape == (0,)


@pytest.mark.parametrize("rho", [0.0, 0.3, -0.3, 0.9, 1.0])
def test_propagate_uncertainty_non_decreasing_for_abs_rho_up_to_one(rho):
    s = propagate_uncertainty_recursive(2.0, 10, correlation=rho)
    assert s[0] == pytest.approx(2.0)
    assert np.all(np.diff(s) >= -1e-12)


@pytest.mark.parametrize("rho", [0.5, 0.9])
def test_propagate_uncertainty_bounded_by_stationary_limit(rho):
    s = propagate_uncertainty_recursive(1.0, 200, correlation=rho)
    limit = 1.0 / np.sqrt(1 - rho ** 2)
    assert np.all(s <= limit + 1e-12)
    assert s[-1] == pytest.approx(limit, rel=1e-6)


def test_propagate_uncertainty_sign_of_rho_does_not_matter():
    np.testing.assert_allclose(
        propagate_uncertainty_recursive(1.0, 6, 0.4),
        propagate_uncertainty_recursive(1.0, 6, -0.4),
    )


def test_propagate_uncertainty_uses_only_first_element_of_array_input():
    s = propagate_uncertainty_recursive(np.array([3.0, 100.0]), 2)
    assert s[0] == pytest.approx(3.0)


def test_propagate_uncertainty_rho_minus_one_is_random_walk_limit():
    s = propagate_uncertainty_recursive(1.0, 4, correlation=-1.0)
    np.testing.assert_allclose(s, np.sqrt([1, 2, 3, 4]))


# ---------------------------------------------------------------------------
# Walk-forward CV
# ---------------------------------------------------------------------------


def test_walk_forward_exact_fit_gives_one_fold():
    folds = list(walk_forward_splits(n=10, min_train=7, horizon=3))
    assert len(folds) == 1
    np.testing.assert_array_equal(folds[0][1], [7, 8, 9])


def test_walk_forward_too_short_series_gives_no_folds():
    assert list(walk_forward_splits(n=9, min_train=7, horizon=3)) == []


def test_walk_forward_test_windows_never_exceed_series_and_follow_train():
    for tr, te in walk_forward_splits(n=50, min_train=10, horizon=4, step=3, max_train=12):
        assert te.max() < 50
        assert tr.max() + 1 == te.min()
        assert len(tr) <= 12
        assert len(te) == 4


def test_walk_forward_max_train_larger_than_series_is_expanding():
    a = list(walk_forward_splits(30, 10, 2, step=5, max_train=1000))
    b = list(walk_forward_splits(30, 10, 2, step=5))
    for (ta, ea), (tb, eb) in zip(a, b):
        np.testing.assert_array_equal(ta, tb)
        np.testing.assert_array_equal(ea, eb)


@pytest.mark.parametrize("step", [0, -1])
def test_walk_forward_non_positive_step_raises(step):
    with pytest.raises(ValueError):
        # islice bounds the run: without validation this would never end.
        list(itertools.islice(walk_forward_splits(20, 5, 1, step=step), 1000))


def test_split_wrapper_get_n_splits_matches_split_count():
    w = TimeSeriesSplitWrapper(n_splits=5, min_train=30, horizon=1)
    X = np.zeros((100, 1))
    assert len(list(w.split(X))) == w.get_n_splits(X)


def test_split_wrapper_series_shorter_than_min_train_gives_no_folds():
    w = TimeSeriesSplitWrapper(n_splits=3, min_train=30, horizon=1)
    assert list(w.split(np.zeros((20, 1)))) == []


class _MeanForecaster:
    """Predicts the training mean for every quantile."""

    def fit(self, train):
        self.mu_ = float(np.mean(train))

    def predict(self, horizon, quantiles):
        q = np.full((len(quantiles), horizon), self.mu_)
        return ForecastResult(quantiles=q[None], q_levels=tuple(quantiles))


def test_walk_forward_cv_without_median_uses_mean_of_quantiles():
    cv = WalkForwardCV(min_train=10, horizon=2, step=5)
    res = cv.evaluate(np.arange(30.0), _MeanForecaster(), quantiles=(0.2, 0.8))
    assert res and all(np.allclose(r.y_pred, np.mean(np.arange(r.train_size))) for r in res)
    assert all(r.wql is not None and r.wql >= 0 for r in res)


def test_walk_forward_cv_summary_of_no_folds_is_nan():
    cv = WalkForwardCV()
    with pytest.warns(RuntimeWarning):
        out = cv.summary([])
    assert out["n_folds"] == 0
    assert np.isnan(out["mean_smape"]) and out["mean_wql"] is None


# ---------------------------------------------------------------------------
# Seasonality
# ---------------------------------------------------------------------------


def test_seasonal_period_all_nan_returns_none():
    assert detect_seasonal_period([np.nan] * 50) is None


def test_seasonal_period_ignores_non_finite_values():
    x = np.sin(np.arange(120) * 2 * np.pi / 6)
    x_with_gaps = x.copy()
    x_with_gaps[[5, 50]] = np.nan
    assert detect_seasonal_period(x) == 6
    assert detect_seasonal_period(np.append(x, [np.inf, -np.inf])) == 6


def test_seasonal_period_max_lag_below_three_returns_none():
    x = np.sin(np.arange(60) * 2 * np.pi / 6)
    assert detect_seasonal_period(x, max_lag=2) is None


def test_seasonal_period_never_exceeds_max_lag():
    x = np.sin(np.arange(200) * 2 * np.pi / 24)
    p = detect_seasonal_period(x, max_lag=10)
    assert p is None or p < 10


def test_seasonal_period_threshold_above_one_finds_nothing():
    x = np.sin(np.arange(120) * 2 * np.pi / 6)
    assert detect_seasonal_period(x, threshold=1.01) is None
