# Copyright (c) 2025 takotime808

import sys
import types
import unittest.mock as mock

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers to build a mock prophet module
# ---------------------------------------------------------------------------

def _make_mock_prophet_module():
    """Return a minimal mock of the prophet package."""
    prophet_mod = types.ModuleType("prophet")

    class FakeProphet:
        def __init__(self, **kwargs):
            self._fitted = False

        def add_seasonality(self, **kwargs):
            pass

        def fit(self, df):
            self._fitted = True
            return self

        def make_future_dataframe(self, periods, freq, include_history):
            return pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=periods, freq="D")})

        def predict(self, df):
            n = len(df)
            return pd.DataFrame({
                "yhat": np.ones(n) * 5.0,
                "yhat_lower": np.ones(n) * 3.0,
                "yhat_upper": np.ones(n) * 7.0,
            })

    prophet_mod.Prophet = FakeProphet
    return prophet_mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProphetForecasterImportGuard:

    def test_raises_if_prophet_not_available(self):
        """ProphetForecaster.__init__ should raise ImportError when prophet is absent."""
        import multioutreg.time_series.prophet_adapter as pa_mod
        original = pa_mod._PROPHET_AVAILABLE
        original_prophet = pa_mod._Prophet
        try:
            pa_mod._PROPHET_AVAILABLE = False
            pa_mod._Prophet = None
            with pytest.raises(ImportError, match="prophet is required"):
                pa_mod.ProphetForecaster()
        finally:
            pa_mod._PROPHET_AVAILABLE = original
            pa_mod._Prophet = original_prophet


class TestProphetForecasterWithMock:

    @pytest.fixture(autouse=True)
    def patch_prophet(self):
        """Inject a fake prophet module so no real installation is needed."""
        mock_mod = _make_mock_prophet_module()
        import multioutreg.time_series.prophet_adapter as pa_mod
        self._pa_mod = pa_mod
        orig_avail = pa_mod._PROPHET_AVAILABLE
        orig_prophet = pa_mod._Prophet
        pa_mod._PROPHET_AVAILABLE = True
        pa_mod._Prophet = mock_mod.Prophet
        yield
        pa_mod._PROPHET_AVAILABLE = orig_avail
        pa_mod._Prophet = orig_prophet

    def _make_series(self, n=120):
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        vals = np.sin(np.linspace(0, 4 * np.pi, n))
        return pd.Series(vals, index=dates)

    def test_fit_returns_self(self):
        from multioutreg.time_series.prophet_adapter import ProphetForecaster
        f = ProphetForecaster()
        y = self._make_series()
        result = f.fit(y)
        assert result is f

    def test_predict_returns_forecast_result(self):
        from multioutreg.time_series.prophet_adapter import ProphetForecaster
        from multioutreg.time_series.chronos_adapter import ForecastResult
        f = ProphetForecaster()
        f.fit(self._make_series())
        res = f.predict(prediction_length=12)
        assert isinstance(res, ForecastResult)

    def test_predict_shape(self):
        from multioutreg.time_series.prophet_adapter import ProphetForecaster
        f = ProphetForecaster()
        f.fit(self._make_series())
        res = f.predict(prediction_length=10, quantiles=(0.1, 0.5, 0.9))
        assert res.quantiles.shape == (1, 3, 10)

    def test_q_levels_stored(self):
        from multioutreg.time_series.prophet_adapter import ProphetForecaster
        f = ProphetForecaster()
        f.fit(self._make_series())
        res = f.predict(prediction_length=5, quantiles=(0.1, 0.5, 0.9))
        assert list(res.q_levels) == [0.1, 0.5, 0.9]

    def test_quantile_ordering(self):
        """lower ≤ median ≤ upper at every step."""
        from multioutreg.time_series.prophet_adapter import ProphetForecaster
        f = ProphetForecaster()
        f.fit(self._make_series())
        res = f.predict(prediction_length=8, quantiles=(0.1, 0.5, 0.9))
        q = res.quantiles[0]  # (3, 8)
        assert np.all(q[0] <= q[1] + 1e-9), "lower > median at some step"
        assert np.all(q[1] <= q[2] + 1e-9), "median > upper at some step"

    def test_fit_with_numpy_array(self):
        """Passing a plain np.ndarray should not raise."""
        from multioutreg.time_series.prophet_adapter import ProphetForecaster
        f = ProphetForecaster()
        y = np.linspace(0.0, 1.0, 60)
        f.fit(y, freq="D")
        res = f.predict(prediction_length=5)
        assert res.quantiles.shape == (1, 3, 5)

    def test_predict_before_fit_raises(self):
        from multioutreg.time_series.prophet_adapter import ProphetForecaster
        f = ProphetForecaster()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            f.predict(prediction_length=5)

    def test_custom_quantiles(self):
        from multioutreg.time_series.prophet_adapter import ProphetForecaster
        f = ProphetForecaster()
        f.fit(self._make_series())
        res = f.predict(prediction_length=6, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95))
        assert res.quantiles.shape == (1, 5, 6)
        assert list(res.q_levels) == [0.05, 0.25, 0.5, 0.75, 0.95]
