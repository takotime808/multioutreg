"""Simple time-series forecasting utilities.

The implementation here converts a univariate time series into a supervised
learning problem using lagged features and trains a multi-output regressor to
predict several future steps at once. This design is inspired by the
"Programmable forecasting in the age of large language models" paper
(arXiv:2403.07815) which advocates modular forecasting components.
"""

from __future__ import annotations

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import RegressorMixin


class TimeSeriesForecaster:
    """Forecast a univariate series using lagged features.

    Parameters
    ----------
    base_estimator : RegressorMixin
        Any regressor following the scikit-learn API.
    lags : int
        Number of past observations to use as features.
    horizon : int
        Number of future steps to predict.
    """

    def __init__(self, base_estimator: RegressorMixin, lags: int, horizon: int):
        self.lags = lags
        self.horizon = horizon
        self.model = MultiOutputRegressor(base_estimator)

    def _create_dataset(self, series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create lagged features and targets from ``series``."""
        series = np.asarray(series, dtype=float)
        n_samples = len(series) - self.lags - self.horizon + 1
        if n_samples <= 0:
            raise ValueError("Series is too short for the given lags and horizon.")
        X = np.zeros((n_samples, self.lags))
        Y = np.zeros((n_samples, self.horizon))
        for i in range(n_samples):
            X[i] = series[i : i + self.lags]
            Y[i] = series[i + self.lags : i + self.lags + self.horizon]
        return X, Y

    def fit(self, series: np.ndarray) -> "TimeSeriesForecaster":
        """Fit the underlying model to ``series``."""
        X, Y = self._create_dataset(series)
        self.model.fit(X, Y)
        return self

    def predict(self, series: np.ndarray) -> np.ndarray:
        """Predict future values for ``series``.

        The model is applied to the most recent ``lags`` observations and
        returns ``horizon`` future predictions.
        """
        series = np.asarray(series, dtype=float)
        if len(series) < self.lags:
            raise ValueError("Series must contain at least `lags` values.")
        last_obs = series[-self.lags :].reshape(1, -1)
        return self.model.predict(last_obs).ravel()
