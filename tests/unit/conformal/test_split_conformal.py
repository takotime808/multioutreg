# Copyright (c) 2026 takotime808

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from multioutreg.conformal.split_conformal import SplitConformalPredictor


@pytest.fixture
def single_output_data():
    X, y = make_regression(n_samples=300, n_features=5, noise=10, random_state=42)
    return X[:200], y[:200], X[200:], y[200:]


@pytest.fixture
def multi_output_data():
    X, y = make_regression(
        n_samples=300, n_features=5, n_targets=3, noise=10, random_state=42
    )
    return X[:200], y[:200], X[200:], y[200:]


class TestSplitConformalSingleOutput:
    def test_fit_and_predict_interval(self, single_output_data):
        X_train, y_train, X_test, y_test = single_output_data
        cp = SplitConformalPredictor(LinearRegression(), random_state=0)
        cp.fit(X_train, y_train)
        y_lower, y_upper = cp.predict_interval(X_test, alpha=0.1)
        assert y_lower.shape == (100,)
        assert y_upper.shape == (100,)
        assert np.all(y_lower <= y_upper)

    def test_coverage_guarantee(self, single_output_data):
        X_train, y_train, X_test, y_test = single_output_data
        alpha = 0.1
        cp = SplitConformalPredictor(
            RandomForestRegressor(n_estimators=50, random_state=0),
            random_state=0,
        )
        cp.fit(X_train, y_train)
        y_lower, y_upper = cp.predict_interval(X_test, alpha=alpha)
        coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))
        # Allow some statistical slack
        assert coverage >= (1 - alpha) - 0.1

    def test_predict_without_alpha(self, single_output_data):
        X_train, y_train, X_test, _ = single_output_data
        cp = SplitConformalPredictor(LinearRegression(), random_state=0)
        cp.fit(X_train, y_train)
        y_pred = cp.predict(X_test)
        assert y_pred.shape == (100,)

    def test_predict_with_alpha(self, single_output_data):
        X_train, y_train, X_test, _ = single_output_data
        cp = SplitConformalPredictor(LinearRegression(), random_state=0)
        cp.fit(X_train, y_train)
        result = cp.predict(X_test, alpha=0.1)
        assert len(result) == 3
        y_pred, y_lower, y_upper = result
        assert y_pred.shape == (100,)
        assert y_lower.shape == (100,)

    def test_not_fitted_raises(self, single_output_data):
        X_train, y_train, X_test, _ = single_output_data
        cp = SplitConformalPredictor(LinearRegression())
        with pytest.raises(AttributeError):
            cp.predict_interval(X_test)

    def test_various_estimators(self, single_output_data):
        X_train, y_train, X_test, _ = single_output_data
        estimators = [
            LinearRegression(),
            RandomForestRegressor(n_estimators=10, random_state=0),
            SVR(),
        ]
        for est in estimators:
            cp = SplitConformalPredictor(est, random_state=0)
            cp.fit(X_train, y_train)
            y_lower, y_upper = cp.predict_interval(X_test, alpha=0.1)
            assert np.all(y_lower <= y_upper)


class TestSplitConformalMultiOutput:
    def test_fit_and_predict_interval(self, multi_output_data):
        X_train, y_train, X_test, y_test = multi_output_data
        cp = SplitConformalPredictor(
            RandomForestRegressor(n_estimators=50, random_state=0),
            random_state=0,
        )
        cp.fit(X_train, y_train)
        y_lower, y_upper = cp.predict_interval(X_test, alpha=0.1)
        assert y_lower.shape == (100, 3)
        assert y_upper.shape == (100, 3)
        assert np.all(y_lower <= y_upper)

    def test_coverage_per_output(self, multi_output_data):
        X_train, y_train, X_test, y_test = multi_output_data
        alpha = 0.1
        cp = SplitConformalPredictor(
            RandomForestRegressor(n_estimators=50, random_state=0),
            random_state=0,
        )
        cp.fit(X_train, y_train)
        y_lower, y_upper = cp.predict_interval(X_test, alpha=alpha)
        for j in range(3):
            coverage = np.mean(
                (y_test[:, j] >= y_lower[:, j]) & (y_test[:, j] <= y_upper[:, j])
            )
            assert coverage >= (1 - alpha) - 0.15


class TestSplitConformalAdaptive:
    def test_adaptive_with_gp(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 3)
        y = X[:, 0] ** 2 + 0.5 * rng.randn(200)
        X_test = rng.randn(50, 3)
        y_test = X_test[:, 0] ** 2 + 0.5 * rng.randn(50)

        cp = SplitConformalPredictor(
            GaussianProcessRegressor(random_state=0),
            adaptive=True,
            random_state=0,
        )
        cp.fit(X, y)
        y_lower, y_upper = cp.predict_interval(X_test, alpha=0.1)
        widths = y_upper - y_lower
        # Adaptive intervals should have varying widths
        assert widths.std() > 0
        assert np.all(y_lower <= y_upper)

    def test_adaptive_coverage(self):
        rng = np.random.RandomState(42)
        X = rng.randn(300, 3)
        y = X[:, 0] * 2 + rng.randn(300)

        cp = SplitConformalPredictor(
            GaussianProcessRegressor(random_state=0),
            adaptive=True,
            random_state=0,
        )
        cp.fit(X[:200], y[:200])
        y_lower, y_upper = cp.predict_interval(X[200:], alpha=0.1)
        coverage = np.mean((y[200:] >= y_lower) & (y[200:] <= y_upper))
        assert coverage >= 0.8


class TestConformalQuantile:
    def test_finite_sample_correction(self):
        scores = np.arange(1, 11, dtype=float)  # [1, 2, ..., 10]
        # For alpha=0.1, n=10: q_level = ceil(11*0.9)/10 = ceil(9.9)/10 = 10/10 = 1.0
        q = SplitConformalPredictor._conformal_quantile(scores, 0.1)
        assert q == 10.0

    def test_quantile_capped_at_1(self):
        scores = np.array([1.0, 2.0, 3.0])
        # For alpha=0.01, n=3: q_level = ceil(4*0.99)/3 = ceil(3.96)/3 = 4/3 > 1 -> capped to 1.0
        q = SplitConformalPredictor._conformal_quantile(scores, 0.01)
        assert q == 3.0
