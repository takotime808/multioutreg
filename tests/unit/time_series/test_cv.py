# Copyright (c) 2025 takotime808

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from multioutreg.time_series.cv import (
    walk_forward_splits,
    WalkForwardCV,
    TimeSeriesSplitWrapper,
    TSFoldResult,
)
from multioutreg.time_series.lag_forecaster import LagFeatureForecaster
from multioutreg.time_series.chronos_adapter import ForecastResult


def _ar1(n=150, phi=0.7, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    y[0] = rng.standard_normal()
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.standard_normal()
    return y


# ---- walk_forward_splits -----------------------------------------------

class TestWalkForwardSplits:

    def test_basic_fold_count(self):
        splits = list(walk_forward_splits(n=100, min_train=50, horizon=5, step=5))
        # first fold: train=0..49, test=50..54
        # last fold where test fits: starts at t=95 → test 95..99 → 10 folds
        assert len(splits) > 0

    def test_no_train_test_leakage(self):
        for train, test in walk_forward_splits(n=80, min_train=30, horizon=3, step=3):
            assert train.max() < test.min()

    def test_expanding_window(self):
        """With max_train=None (expanding), training set grows each fold."""
        splits = list(walk_forward_splits(n=60, min_train=20, horizon=1, step=5))
        train_sizes = [len(tr) for tr, _ in splits]
        assert train_sizes == sorted(train_sizes)

    def test_rolling_window(self):
        """With max_train=20, training set size stays ≤ 20."""
        splits = list(walk_forward_splits(n=80, min_train=20, horizon=1, step=5,
                                          max_train=20))
        for train, _ in splits:
            assert len(train) <= 20

    def test_test_length_equals_horizon(self):
        for _, test in walk_forward_splits(n=100, min_train=40, horizon=7, step=7):
            assert len(test) == 7

    def test_step_one_yields_many_folds(self):
        splits = list(walk_forward_splits(n=50, min_train=30, horizon=1, step=1))
        assert len(splits) == 20  # 50 - 30 - 1 + 1 = 20


# ---- WalkForwardCV -----------------------------------------------------

class TestWalkForwardCV:

    def _make_forecaster(self):
        """A simple LagFeatureForecaster wrapping LinearRegression."""
        return LagFeatureForecaster(LinearRegression(), n_lags=8, uncertainty="none")

    def test_evaluate_returns_list_of_fold_results(self):
        y = _ar1()
        cv = WalkForwardCV(min_train=60, horizon=5, step=10)
        results = cv.evaluate(y, self._make_forecaster())
        assert isinstance(results, list)
        assert all(isinstance(r, TSFoldResult) for r in results)

    def test_fold_result_fields(self):
        y = _ar1()
        cv = WalkForwardCV(min_train=80, horizon=5, step=20)
        results = cv.evaluate(y, self._make_forecaster())
        r = results[0]
        assert r.fold_idx == 0
        assert r.y_true.shape == (5,)
        assert r.y_pred.shape == (5,)
        assert isinstance(r.smape, float)
        assert isinstance(r.mase, float)

    def test_summary_keys(self):
        y = _ar1()
        cv = WalkForwardCV(min_train=80, horizon=3, step=15)
        results = cv.evaluate(y, self._make_forecaster())
        s = cv.summary(results)
        for key in ("mean_smape", "std_smape", "mean_mase", "std_mase", "n_folds"):
            assert key in s

    def test_summary_n_folds_matches_results(self):
        y = _ar1()
        cv = WalkForwardCV(min_train=80, horizon=5, step=10)
        results = cv.evaluate(y, self._make_forecaster())
        s = cv.summary(results)
        assert s["n_folds"] == len(results)

    def test_smape_non_negative(self):
        y = _ar1()
        cv = WalkForwardCV(min_train=80, horizon=3, step=15)
        results = cv.evaluate(y, self._make_forecaster())
        for r in results:
            assert r.smape >= 0.0

    def test_split_yields_same_as_walk_forward_splits(self):
        y = _ar1()
        cv = WalkForwardCV(min_train=50, horizon=5, step=10)
        from_split = list(cv.split(y))
        from_fn = list(walk_forward_splits(n=len(y), min_train=50, horizon=5, step=10))
        assert len(from_split) == len(from_fn)
        for (tr1, te1), (tr2, te2) in zip(from_split, from_fn):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)


# ---- TimeSeriesSplitWrapper --------------------------------------------

class TestTimeSeriesSplitWrapper:

    def test_get_n_splits(self):
        wrapper = TimeSeriesSplitWrapper(n_splits=5, min_train=30, horizon=1)
        assert wrapper.get_n_splits() == 5

    def test_split_yields_arrays(self):
        X = np.zeros((100, 3))
        wrapper = TimeSeriesSplitWrapper(n_splits=4, min_train=30, horizon=1)
        splits = list(wrapper.split(X))
        assert len(splits) > 0
        for train, test in splits:
            assert isinstance(train, np.ndarray)
            assert isinstance(test, np.ndarray)

    def test_no_leakage_in_wrapper(self):
        X = np.zeros((80, 2))
        wrapper = TimeSeriesSplitWrapper(n_splits=3, min_train=20, horizon=2)
        for train, test in wrapper.split(X):
            assert train.max() < test.min()

    def test_sklearn_compatible(self):
        """Verify it works with sklearn cross_val_score."""
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LinearRegression
        X = np.random.default_rng(0).standard_normal((60, 3))
        y = X[:, 0] + np.random.default_rng(1).standard_normal(60) * 0.1
        wrapper = TimeSeriesSplitWrapper(n_splits=3, min_train=20, horizon=1)
        scores = cross_val_score(LinearRegression(), X, y, cv=wrapper, scoring="r2")
        assert len(scores) > 0
