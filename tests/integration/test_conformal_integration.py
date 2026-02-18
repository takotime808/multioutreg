# Copyright (c) 2026 takotime808

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from multioutreg.conformal import SplitConformalPredictor, CVPlusConformalPredictor
from multioutreg.conformal.metrics import conformal_summary
from multioutreg.surrogates import RandomForestSurrogate


class TestEndToEndSplitConformal:
    def test_full_pipeline(self):
        X, y = make_regression(
            n_samples=500, n_features=5, n_targets=2, noise=10, random_state=42
        )
        X_train, X_test = X[:400], X[400:]
        y_train, y_test = y[:400], y[400:]

        cp = SplitConformalPredictor(
            RandomForestRegressor(n_estimators=50, random_state=0),
            calibration_size=0.25,
            random_state=0,
        )
        cp.fit(X_train, y_train)

        y_pred, y_lower, y_upper = cp.predict(X_test, alpha=0.1)
        assert y_pred.shape == (100, 2)
        assert y_lower.shape == (100, 2)

        summary = conformal_summary(y_test, y_lower, y_upper, alpha=0.1)
        assert len(summary) == 2
        for _, row in summary.iterrows():
            assert row["coverage"] >= 0.75  # generous slack for test stability


class TestEndToEndCVPlus:
    def test_full_pipeline(self):
        X, y = make_regression(
            n_samples=500, n_features=5, n_targets=2, noise=10, random_state=42
        )
        X_train, X_test = X[:400], X[400:]
        y_train, y_test = y[:400], y[400:]

        cp = CVPlusConformalPredictor(
            RandomForestRegressor(n_estimators=50, random_state=0),
            n_folds=5,
            random_state=0,
        )
        cp.fit(X_train, y_train)

        y_pred, y_lower, y_upper = cp.predict(X_test, alpha=0.1)
        assert y_pred.shape == (100, 2)

        summary = conformal_summary(y_test, y_lower, y_upper, alpha=0.1)
        assert len(summary) == 2
        for _, row in summary.iterrows():
            assert row["coverage"] >= 0.75


class TestBaseSurrogateConformalIntegration:
    def test_wrap_conformal_split(self):
        X, y = make_regression(
            n_samples=300, n_features=5, n_targets=2, noise=10, random_state=42
        )
        X_train, X_cal, X_test = X[:150], X[150:250], X[250:]
        y_train, y_cal, y_test = y[:150], y[150:250], y[250:]

        surrogate = RandomForestSurrogate()
        surrogate.fit(X_train, y_train)
        surrogate.wrap_conformal(X_cal, y_cal)

        y_lower, y_upper = surrogate.conformal_predict(X_test, alpha=0.1)
        assert y_lower.shape[0] == 50
        assert np.all(y_lower <= y_upper)

    def test_conformal_predict_without_wrapping_raises(self):
        surrogate = RandomForestSurrogate()
        X = np.random.randn(10, 3)
        y = np.random.randn(10, 2)
        surrogate.fit(X, y)
        with pytest.raises(AttributeError):
            surrogate.conformal_predict(X)


class TestAutoDetectConformalIntegration:
    def test_calibrate_conformal_split(self):
        X, y = make_regression(
            n_samples=500, n_features=5, n_targets=2, noise=10, random_state=42
        )
        X_train, X_cal, X_test = X[:300], X[300:400], X[400:]
        y_train, y_cal, y_test = y[:300], y[300:400], y[400:]

        from multioutreg.model_selection import AutoDetectMultiOutputRegressor

        model = AutoDetectMultiOutputRegressor(
            estimators=[
                LinearRegression(),
                RandomForestRegressor(n_estimators=50, random_state=0),
            ],
            param_spaces=[{}, {"n_estimators": [50]}],
            cv=2,
        )
        model.fit(X_train, y_train)
        model.calibrate_conformal(X_cal, y_cal)

        y_lower, y_upper = model.predict_interval(X_test, alpha=0.1)
        assert y_lower.shape == (100, 2)
        assert y_upper.shape == (100, 2)
        assert np.all(y_lower <= y_upper)

        # Check coverage
        for j in range(2):
            coverage = np.mean(
                (y_test[:, j] >= y_lower[:, j]) & (y_test[:, j] <= y_upper[:, j])
            )
            assert coverage >= 0.7

    def test_calibrate_conformal_cv_plus(self):
        X, y = make_regression(
            n_samples=400, n_features=5, n_targets=2, noise=10, random_state=42
        )
        X_train, X_cal, X_test = X[:200], X[200:300], X[300:]
        y_train, y_cal, y_test = y[:200], y[200:300], y[300:]

        from multioutreg.model_selection import AutoDetectMultiOutputRegressor

        model = AutoDetectMultiOutputRegressor(
            estimators=[RandomForestRegressor(n_estimators=50, random_state=0)],
            param_spaces=[{"n_estimators": [50]}],
            cv=2,
        )
        model.fit(X_train, y_train)
        model.calibrate_conformal(X_cal, y_cal)

        y_lower, y_upper = model.predict_interval(X_test, alpha=0.1)
        assert y_lower.shape == (100, 2)
        assert np.all(y_lower <= y_upper)

    def test_predict_interval_without_calibration_raises(self):
        X, y = make_regression(n_samples=100, n_features=5, n_targets=2, random_state=42)

        from multioutreg.model_selection import AutoDetectMultiOutputRegressor

        model = AutoDetectMultiOutputRegressor(
            estimators=[LinearRegression()],
            param_spaces=[{}],
            cv=2,
        )
        model.fit(X, y)
        with pytest.raises(AttributeError):
            model.predict_interval(X)
