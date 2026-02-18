# Copyright (c) 2025 takotime808

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from multioutreg.conformal.cv_plus import CVPlusConformalPredictor


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


class TestCVPlusSingleOutput:
    def test_fit_and_predict_interval(self, single_output_data):
        X_train, y_train, X_test, y_test = single_output_data
        cp = CVPlusConformalPredictor(LinearRegression(), n_folds=5, random_state=0)
        cp.fit(X_train, y_train)
        y_lower, y_upper = cp.predict_interval(X_test, alpha=0.1)
        assert y_lower.shape == (100,)
        assert y_upper.shape == (100,)
        assert np.all(y_lower <= y_upper)

    def test_coverage_guarantee(self, single_output_data):
        X_train, y_train, X_test, y_test = single_output_data
        alpha = 0.1
        cp = CVPlusConformalPredictor(
            RandomForestRegressor(n_estimators=50, random_state=0),
            n_folds=5,
            random_state=0,
        )
        cp.fit(X_train, y_train)
        y_lower, y_upper = cp.predict_interval(X_test, alpha=alpha)
        coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))
        assert coverage >= (1 - alpha) - 0.1

    def test_predict_without_alpha(self, single_output_data):
        X_train, y_train, X_test, _ = single_output_data
        cp = CVPlusConformalPredictor(LinearRegression(), random_state=0)
        cp.fit(X_train, y_train)
        y_pred = cp.predict(X_test)
        assert y_pred.shape == (100,)

    def test_predict_with_alpha(self, single_output_data):
        X_train, y_train, X_test, _ = single_output_data
        cp = CVPlusConformalPredictor(LinearRegression(), random_state=0)
        cp.fit(X_train, y_train)
        result = cp.predict(X_test, alpha=0.1)
        assert len(result) == 3
        y_pred, y_lower, y_upper = result
        assert y_pred.shape == (100,)
        assert y_lower.shape == (100,)

    def test_not_fitted_raises(self, single_output_data):
        X_train, y_train, X_test, _ = single_output_data
        cp = CVPlusConformalPredictor(LinearRegression())
        with pytest.raises(AttributeError):
            cp.predict_interval(X_test)

    def test_all_residuals_computed(self, single_output_data):
        X_train, y_train, _, _ = single_output_data
        cp = CVPlusConformalPredictor(LinearRegression(), n_folds=5, random_state=0)
        cp.fit(X_train, y_train)
        # Every training point should have a residual (no NaNs)
        assert not np.any(np.isnan(cp.residuals_))
        assert cp.residuals_.shape[0] == X_train.shape[0]

    def test_fold_models_stored(self, single_output_data):
        X_train, y_train, _, _ = single_output_data
        cp = CVPlusConformalPredictor(LinearRegression(), n_folds=5, random_state=0)
        cp.fit(X_train, y_train)
        assert len(cp.fold_models_) == 5


class TestCVPlusMultiOutput:
    def test_fit_and_predict_interval(self, multi_output_data):
        X_train, y_train, X_test, y_test = multi_output_data
        cp = CVPlusConformalPredictor(
            RandomForestRegressor(n_estimators=50, random_state=0),
            n_folds=5,
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
        cp = CVPlusConformalPredictor(
            RandomForestRegressor(n_estimators=50, random_state=0),
            n_folds=5,
            random_state=0,
        )
        cp.fit(X_train, y_train)
        y_lower, y_upper = cp.predict_interval(X_test, alpha=alpha)
        for j in range(3):
            coverage = np.mean(
                (y_test[:, j] >= y_lower[:, j]) & (y_test[:, j] <= y_upper[:, j])
            )
            assert coverage >= (1 - alpha) - 0.15
