# Copyright (c) 2026 takotime808
"""AutoSurrogateForecaster with the AutoDetect grid search stubbed out."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

import multioutreg.model_selection as model_selection
from multioutreg.surrogates import LinearRegressionSurrogate
from multioutreg.time_series.lag_forecaster import LagFeatureForecaster
from multioutreg.time_series.surrogate_forecaster import AutoSurrogateForecaster


class _FakeDetector:
    """Stands in for AutoDetectMultiOutputRegressor; records what it was fit on."""

    last = None

    def __init__(self, best, name=None):
        self.best_estimator_ = best
        if name is not None:
            self.best_model_name_ = name

    @classmethod
    def factory(cls, best, name=None):
        def with_vendored_surrogates(cv):
            det = cls(best, name)
            det.cv = cv
            cls.last = det
            return det

        return type("Stub", (), {"with_vendored_surrogates": staticmethod(with_vendored_surrogates)})

    def fit(self, X, y):
        self.X_shape, self.y_shape = X.shape, np.shape(y)
        return self


@pytest.fixture
def series():
    t = np.arange(80.0)
    return np.sin(2 * np.pi * t / 8) + 0.01 * t


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        AutoSurrogateForecaster().predict()


def test_fit_uses_detector_choice_and_reports_name(monkeypatch, series):
    stub = _FakeDetector.factory(LinearRegression(), name="linear")
    monkeypatch.setattr(model_selection, "AutoDetectMultiOutputRegressor", stub)
    f = AutoSurrogateForecaster(n_lags=6, horizon=1, cv=2, uncertainty="none").fit(series)
    assert f.best_model_name == "linear"
    assert _FakeDetector.last.cv == 2
    # Single-step targets are squeezed to 1-D for the detector.
    assert _FakeDetector.last.X_shape == (80 - 6, 6)
    assert _FakeDetector.last.y_shape == (80 - 6,)


def test_best_model_name_falls_back_to_class_name(monkeypatch, series):
    stub = _FakeDetector.factory(LinearRegression())
    monkeypatch.setattr(model_selection, "AutoDetectMultiOutputRegressor", stub)
    f = AutoSurrogateForecaster(n_lags=4, uncertainty="none").fit(series)
    assert f.best_model_name == "LinearRegression"


def test_multi_step_horizon_keeps_2d_targets(monkeypatch, series):
    stub = _FakeDetector.factory(LinearRegression())
    monkeypatch.setattr(model_selection, "AutoDetectMultiOutputRegressor", stub)
    AutoSurrogateForecaster(n_lags=5, horizon=3, uncertainty="none").fit(series)
    assert _FakeDetector.last.y_shape == (80 - 5 - 3 + 1, 3)


@pytest.mark.parametrize("uncertainty", ["conformal", "none"])
def test_forecast_shape_and_ordered_quantiles(monkeypatch, series, uncertainty):
    stub = _FakeDetector.factory(LinearRegression())
    monkeypatch.setattr(model_selection, "AutoDetectMultiOutputRegressor", stub)
    f = AutoSurrogateForecaster(n_lags=8, uncertainty=uncertainty).fit(series)
    res = f.predict(horizon=4, quantiles=(0.1, 0.5, 0.9))
    q = res.quantiles[0]
    assert q.shape == (3, 4)
    assert np.all(np.isfinite(q))
    assert np.all(q[0] <= q[1] + 1e-12) and np.all(q[1] <= q[2] + 1e-12)


def test_series_too_short_for_lags_raises(monkeypatch):
    stub = _FakeDetector.factory(LinearRegression())
    monkeypatch.setattr(model_selection, "AutoDetectMultiOutputRegressor", stub)
    with pytest.raises(ValueError, match="too short"):
        AutoSurrogateForecaster(n_lags=12).fit(np.arange(10.0))


def test_lag_forecaster_accepts_base_surrogate_for_single_step(series):
    f = LagFeatureForecaster(LinearRegressionSurrogate(), n_lags=6, horizon=1, uncertainty="none")
    f.fit(series)
    assert f.predict(horizon=3).quantiles.shape[-1] == 3
