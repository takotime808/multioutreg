# Copyright (c) 2025 takotime808

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from multioutreg.time_series.chronos_adapter import ForecastResult
from multioutreg.time_series.lag_features import make_lag_features
from multioutreg.time_series.lag_forecaster import LagFeatureForecaster


class AutoSurrogateForecaster:
    """Auto-select the best surrogate for time series via lag features.

    Uses ``AutoDetectMultiOutputRegressor.with_vendored_surrogates()`` to run
    a grid search on the lag feature matrix and automatically select the best
    surrogate, then wraps it in a ``LagFeatureForecaster`` for forecasting.

    Parameters
    ----------
    n_lags : int, default 12
        Number of lagged features per sample.
    horizon : int, default 1
        Forecast horizon.
    cv : int, default 3
        Cross-validation folds for AutoDetect grid search.
    uncertainty : str, default "conformal"
        Uncertainty strategy passed to ``LagFeatureForecaster``:
        ``"conformal"``, ``"return_std"``, or ``"none"``.
    cal_frac : float, default 0.2
        Fraction of data held out for conformal calibration.

    Examples
    --------
    >>> forecaster = AutoSurrogateForecaster(n_lags=12, horizon=5)
    >>> forecaster.fit(y_train)
    >>> result = forecaster.predict(horizon=5)
    """

    def __init__(
        self,
        n_lags: int = 12,
        horizon: int = 1,
        cv: int = 3,
        uncertainty: str = "conformal",
        cal_frac: float = 0.2,
    ):
        self.n_lags = n_lags
        self.horizon = horizon
        self.cv = cv
        self.uncertainty = uncertainty
        self.cal_frac = cal_frac

        self._forecaster: Optional[LagFeatureForecaster] = None
        self._best_model_name: Optional[str] = None

    def fit(self, series: np.ndarray | pd.Series) -> "AutoSurrogateForecaster":
        """Build lag features, run AutoDetect, wrap best surrogate, calibrate.

        Parameters
        ----------
        series : 1D time series array or pd.Series

        Returns
        -------
        self
        """
        from multioutreg.model_selection import AutoDetectMultiOutputRegressor

        arr = np.asarray(series, dtype=float) if not isinstance(series, pd.Series) else series.to_numpy(dtype=float)

        X, y = make_lag_features(arr, n_lags=self.n_lags, horizon=self.horizon)
        y_1d = y.squeeze(axis=1) if self.horizon == 1 else y

        # Run AutoDetect to find the best surrogate
        detector = AutoDetectMultiOutputRegressor.with_vendored_surrogates(cv=self.cv)
        detector.fit(X, y_1d)

        best_surrogate = detector.best_estimator_
        self._best_model_name = getattr(detector, "best_model_name_", type(best_surrogate).__name__)

        self._forecaster = LagFeatureForecaster(
            surrogate=best_surrogate,
            n_lags=self.n_lags,
            horizon=self.horizon,
            uncertainty=self.uncertainty,
        )
        self._forecaster.fit(arr, cal_frac=self.cal_frac)
        return self

    def predict(
        self,
        horizon: Optional[int] = None,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        alpha: float = 0.1,
    ) -> ForecastResult:
        """Recursive multi-step forecast with uncertainty intervals.

        Parameters
        ----------
        horizon : int | None
            Steps to forecast.  Defaults to ``self.horizon``.
        quantiles : sequence of float
        alpha : float
            Coverage level for conformal prediction (1 - alpha).

        Returns
        -------
        ForecastResult
        """
        if self._forecaster is None:
            raise RuntimeError("Call fit() before predict().")
        return self._forecaster.predict(horizon=horizon, quantiles=quantiles, alpha=alpha)

    @property
    def best_model_name(self) -> Optional[str]:
        """Name of the surrogate selected by AutoDetect."""
        return self._best_model_name
