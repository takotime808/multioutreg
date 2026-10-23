# Copyright (c) 2025 takotime808

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from multioutreg.time_series.lag_features import make_lag_features, rolling_window_features
from multioutreg.time_series.chronos_adapter import ForecastResult

# z-score at q=0.9 for Gaussian: scipy.stats.norm.ppf(0.9) ≈ 1.2816
# Used to convert a conformal half-width into a "sigma" for the Gaussian quantile formula.
_Z_90 = 1.2815515655446004


class LagFeatureForecaster:
    """Wrap any surrogate or sklearn estimator to forecast a time series via lag features.

    Converts a 1D series to a tabular (X_lag, y) sliding-window matrix, fits the
    surrogate, then iterates multi-step predictions by feeding each step's prediction
    back as the next row's lag context.

    Uncertainty is propagated using one of three strategies:

    - ``"conformal"``  : calls ``wrap_conformal()`` / ``conformal_predict()`` if the
                         surrogate exposes a ``ConformalMixin`` API; falls back to
                         ``"return_std"`` otherwise.
    - ``"return_std"`` : calls ``surrogate.predict(X, return_std=True)`` to get std.
    - ``"none"``       : point prediction only; lower/upper equal the point prediction.

    Parameters
    ----------
    surrogate : Any
        Fitted or unfitted sklearn-compatible estimator or BaseSurrogate.
    n_lags : int, default 12
        Number of historical steps used as features.
    horizon : int, default 1
        Steps to forecast ahead in a single call to ``predict()``.
    uncertainty : str, default "return_std"
        Uncertainty strategy: ``"conformal"``, ``"return_std"``, or ``"none"``.
    include_time_features : bool, default False
        Append sin/cos-encoded calendar features (requires DatetimeIndex).
    include_rolling_windows : bool, default False
        Append rolling mean/std columns to the lag matrix.
    rolling_windows : sequence of int, default (4, 8, 24)
        Window sizes used when include_rolling_windows=True.
    """

    def __init__(
        self,
        surrogate: Any,
        n_lags: int = 12,
        horizon: int = 1,
        uncertainty: str = "return_std",
        include_time_features: bool = False,
        include_rolling_windows: bool = False,
        rolling_windows: Sequence[int] = (4, 8, 24),
    ):
        if uncertainty not in ("conformal", "return_std", "none"):
            raise ValueError(
                f"uncertainty must be 'conformal', 'return_std', or 'none'; got {uncertainty!r}"
            )
        self.surrogate = surrogate
        self.n_lags = n_lags
        self.horizon = horizon
        self.uncertainty = uncertainty
        self.include_time_features = include_time_features
        self.include_rolling_windows = include_rolling_windows
        self.rolling_windows = tuple(rolling_windows)

        self._series: np.ndarray | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        series: np.ndarray | pd.Series,
        cal_frac: float = 0.2,
    ) -> "LagFeatureForecaster":
        """Fit the surrogate on lag features derived from ``series``.

        Parameters
        ----------
        series : 1D time series array or pd.Series
        cal_frac : float, default 0.2
            Fraction of the lag-feature matrix held out for conformal calibration.
            Only used when ``uncertainty="conformal"``.

        Returns
        -------
        self
        """
        arr = _to_array(series)
        self._series = arr

        X, y = make_lag_features(arr, n_lags=self.n_lags, horizon=self.horizon,
                                 include_time_features=self.include_time_features)
        X = self._maybe_add_rolling(arr, X)

        # single-step: y is (n,1) → squeeze to (n,) for sklearn compatibility
        y_fit = y.squeeze(axis=1) if self.horizon == 1 else y

        if self.uncertainty == "conformal":
            n_cal = max(1, int(len(X) * cal_frac))
            X_train, X_cal = X[:-n_cal], X[-n_cal:]
            y_train, y_cal = y_fit[:-n_cal], y_fit[-n_cal:]
            if hasattr(self.surrogate, "wrap_conformal"):
                self.surrogate.fit(X_train, y_train)
                self.surrogate.wrap_conformal(X_cal, y_cal)
            else:
                # Residual-based split-conformal fallback for plain sklearn estimators
                self.surrogate.fit(X_train, y_train)
                preds_cal = self.surrogate.predict(X_cal)
                abs_res = np.abs(y_cal.ravel() - preds_cal.ravel())
                n_res = len(abs_res)
                level = min(1.0, np.ceil((n_res + 1) * 0.9) / n_res)
                self._conformal_q_ = float(np.quantile(abs_res, level))
        else:
            self.surrogate.fit(X, y_fit)

        self._fitted = True
        return self

    def predict(
        self,
        horizon: Optional[int] = None,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        alpha: float = 0.1,
    ) -> ForecastResult:
        """Forecast ``horizon`` steps ahead.

        Parameters
        ----------
        horizon : int | None
            Number of steps to forecast.  Defaults to ``self.horizon``.
        quantiles : sequence of float
            Desired quantile levels in the output.
        alpha : float
            Coverage level for conformal intervals (1 - alpha coverage).

        Returns
        -------
        ForecastResult
            Shape: ``[1, n_quantiles, horizon]`` (single series).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        h = horizon if horizon is not None else self.horizon
        context = self._series[-self.n_lags:]
        means, stds = self._recursive_forecast(context, h)

        q_levels = tuple(float(q) for q in quantiles)
        q_arr = _build_quantile_array(means, stds, q_levels, self.surrogate, alpha,
                                      uncertainty=self.uncertainty)
        return ForecastResult(
            quantiles=q_arr[np.newaxis, :, :],  # [1, Q, H]
            q_levels=q_levels,
            ids=("y",),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recursive_forecast(
        self,
        context: np.ndarray,
        horizon: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Iterative multi-step: re-feed predicted point as the next lag.

        When ``include_rolling_windows=True`` the rolling features are recomputed
        from the growing context buffer at every step so that inference uses the
        same feature space as training.

        Returns
        -------
        means : np.ndarray, shape (horizon,)
        stds  : np.ndarray, shape (horizon,)  — zeros when uncertainty="none"
        """
        # Keep the full tail of the series so rolling stats have history
        ctx = self._series.tolist()
        means = np.zeros(horizon)
        stds = np.zeros(horizon)

        for step in range(horizon):
            lags = np.array(ctx[-self.n_lags:], dtype=float).reshape(1, -1)

            if self.include_rolling_windows:
                ctx_arr = np.array(ctx, dtype=float)
                roll = rolling_window_features(ctx_arr, windows=self.rolling_windows)
                roll_row = roll[-1:, :]  # last row = current timestep's rolling stats
                x_row = np.hstack([lags, roll_row])
            else:
                x_row = lags

            if self.uncertainty == "conformal" and hasattr(self.surrogate, "conformal_predict"):
                y_lower, y_upper = self.surrogate.conformal_predict(x_row, alpha=0.1)
                mu = (y_lower.ravel()[0] + y_upper.ravel()[0]) / 2.0
                sigma = (y_upper.ravel()[0] - y_lower.ravel()[0]) / (2 * _Z_90)
            elif self.uncertainty == "conformal" and hasattr(self, "_conformal_q_"):
                # Residual-based conformal fallback: treat conformal_q as the ±sigma
                # at q=0.1/q=0.9 so Gaussian formula gives the correct interval width.
                mu = float(np.asarray(self.surrogate.predict(x_row)).ravel()[0])
                sigma = self._conformal_q_ / _Z_90
            elif self.uncertainty == "return_std" and _supports_return_std(self.surrogate):
                mu_arr, std_arr = self.surrogate.predict(x_row, return_std=True)
                mu = float(np.asarray(mu_arr).ravel()[0])
                sigma = float(np.asarray(std_arr).ravel()[0])
            else:
                mu = float(np.asarray(self.surrogate.predict(x_row)).ravel()[0])
                sigma = 0.0

            means[step] = mu
            stds[step] = sigma
            ctx.append(mu)

        return means, stds

    def _maybe_add_rolling(self, arr: np.ndarray, X: np.ndarray) -> np.ndarray:
        if not self.include_rolling_windows:
            return X
        roll = rolling_window_features(arr, windows=self.rolling_windows)
        n_samples = X.shape[0]
        roll_trimmed = roll[self.n_lags: self.n_lags + n_samples]
        return np.hstack([X, roll_trimmed])


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _to_array(series: np.ndarray | pd.Series) -> np.ndarray:
    if isinstance(series, pd.Series):
        return series.to_numpy(dtype=float)
    return np.asarray(series, dtype=float)


def _supports_return_std(surrogate: Any) -> bool:
    """Heuristic check: does the surrogate support predict(return_std=True)?"""
    import inspect
    try:
        sig = inspect.signature(surrogate.predict)
        return "return_std" in sig.parameters
    except (ValueError, TypeError):
        return False


def _build_quantile_array(
    means: np.ndarray,
    stds: np.ndarray,
    q_levels: tuple,
    surrogate: Any,
    alpha: float,
    uncertainty: str,
) -> np.ndarray:
    """Build (Q, H) quantile array from means and stds using Gaussian approximation."""
    from scipy.special import erfinv

    q_arr = np.zeros((len(q_levels), len(means)))
    for i, q in enumerate(q_levels):
        if uncertainty == "none" or np.all(stds == 0):
            q_arr[i] = means
        else:
            # Gaussian quantile: mu + std * sqrt(2) * erfinv(2q - 1)
            z = np.sqrt(2.0) * erfinv(2.0 * float(q) - 1.0)
            q_arr[i] = means + stds * z
    return q_arr
