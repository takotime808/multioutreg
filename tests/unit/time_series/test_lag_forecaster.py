# Copyright (c) 2025 takotime808

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from multioutreg.time_series.lag_forecaster import LagFeatureForecaster
from multioutreg.time_series.chronos_adapter import ForecastResult


def _ar1(n=100, phi=0.7, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    y[0] = rng.standard_normal()
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.standard_normal()
    return y


class TestLagFeatureForecasterUncertaintyNone:

    def test_fit_predict_returns_forecast_result(self):
        y = _ar1()
        lff = LagFeatureForecaster(LinearRegression(), n_lags=10, horizon=1, uncertainty="none")
        lff.fit(y)
        res = lff.predict(horizon=5)
        assert isinstance(res, ForecastResult)

    def test_output_shape(self):
        """Shape should be [1, n_quantiles, horizon]."""
        y = _ar1()
        lff = LagFeatureForecaster(LinearRegression(), n_lags=8, uncertainty="none")
        lff.fit(y)
        res = lff.predict(horizon=6, quantiles=(0.1, 0.5, 0.9))
        assert res.quantiles.shape == (1, 3, 6)

    def test_median_equals_lower_equals_upper_when_no_uncertainty(self):
        """With uncertainty='none' all quantiles should equal the point prediction."""
        y = _ar1()
        lff = LagFeatureForecaster(LinearRegression(), n_lags=8, uncertainty="none")
        lff.fit(y)
        res = lff.predict(horizon=4, quantiles=(0.1, 0.5, 0.9))
        # All rows of quantiles[0] should be equal
        np.testing.assert_allclose(res.quantiles[0, 0], res.quantiles[0, 1])
        np.testing.assert_allclose(res.quantiles[0, 1], res.quantiles[0, 2])

    def test_predict_before_fit_raises(self):
        lff = LagFeatureForecaster(LinearRegression(), n_lags=5, uncertainty="none")
        with pytest.raises(RuntimeError, match="fit"):
            lff.predict(horizon=3)

    def test_invalid_uncertainty_raises(self):
        with pytest.raises(ValueError, match="uncertainty"):
            LagFeatureForecaster(LinearRegression(), uncertainty="invalid")


class TestLagFeatureForecasterReturnStd:

    def test_return_std_produces_spreading_intervals(self):
        """With return_std, lower <= median <= upper for all steps."""
        y = _ar1(n=120, phi=0.8)
        lff = LagFeatureForecaster(
            RandomForestRegressor(n_estimators=20, random_state=0),
            n_lags=10,
            uncertainty="return_std",
        )
        lff.fit(y)
        res = lff.predict(horizon=5, quantiles=(0.1, 0.5, 0.9))
        lower = res.quantiles[0, 0]
        median = res.quantiles[0, 1]
        upper = res.quantiles[0, 2]
        assert np.all(lower <= median + 1e-9)
        assert np.all(median <= upper + 1e-9)

    def test_q_levels_correct(self):
        y = _ar1()
        lff = LagFeatureForecaster(Ridge(), n_lags=8, uncertainty="none")
        lff.fit(y)
        res = lff.predict(horizon=3, quantiles=(0.25, 0.75))
        assert list(res.q_levels) == [0.25, 0.75]


class TestLagFeatureForecasterWithRolling:

    def test_rolling_windows_does_not_crash(self):
        y = _ar1(n=150)
        lff = LagFeatureForecaster(
            LinearRegression(), n_lags=12, uncertainty="none",
            include_rolling_windows=True, rolling_windows=(4, 8),
        )
        lff.fit(y)
        res = lff.predict(horizon=5)
        assert res.quantiles.shape[2] == 5


class TestLagFeatureForecasterPdSeries:

    def test_accepts_pd_series(self):
        y = pd.Series(_ar1(n=100))
        lff = LagFeatureForecaster(LinearRegression(), n_lags=8, uncertainty="none")
        lff.fit(y)
        # Default quantiles=(0.1, 0.5, 0.9) → 3 quantile levels
        res = lff.predict(horizon=4)
        assert res.quantiles.shape == (1, 3, 4)
