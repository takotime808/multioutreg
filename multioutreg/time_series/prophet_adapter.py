# Copyright (c) 2025 takotime808

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from multioutreg.time_series.chronos_adapter import ForecastResult

try:
    from prophet import Prophet as _Prophet
    _PROPHET_AVAILABLE = True
except ImportError:
    _Prophet = None  # type: ignore[assignment]
    _PROPHET_AVAILABLE = False


def _require_prophet() -> None:
    if not _PROPHET_AVAILABLE:
        raise ImportError(
            "prophet is required for ProphetForecaster. "
            "Install it with:  pip install prophet"
        )


class ProphetForecaster:
    """Thin sklearn-style wrapper around Meta Prophet.

    Prophet handles trend and seasonality automatically via its additive
    decomposition model.  Quantile uncertainty is obtained from Prophet's
    built-in prediction interval sampling.

    Requires::

        pip install prophet

    Parameters
    ----------
    seasonality_mode : str, default "additive"
        ``"additive"`` or ``"multiplicative"``.
    interval_width : float, default 0.8
        Width of the inner prediction interval that Prophet uses internally
        (``yhat_lower`` / ``yhat_upper``).  Wider = larger fan.
    mcmc_samples : int, default 0
        If 0, uses MAP estimation (fast). Set > 0 for full Bayesian
        posterior (slow but better-calibrated intervals).
    additional_seasonalities : list[dict] | None
        Extra seasonality components passed to ``Prophet.add_seasonality()``.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> dates = pd.date_range("2022-01-01", periods=200, freq="D")
    >>> y = pd.Series(np.sin(np.linspace(0, 4*np.pi, 200)), index=dates)
    >>> f = ProphetForecaster().fit(y)
    >>> result = f.predict(prediction_length=30)
    >>> result.quantiles.shape  # (1, 3, 30)
    """

    def __init__(
        self,
        seasonality_mode: str = "additive",
        interval_width: float = 0.8,
        mcmc_samples: int = 0,
        additional_seasonalities: Optional[list] = None,
    ):
        _require_prophet()
        self.seasonality_mode = seasonality_mode
        self.interval_width = interval_width
        self.mcmc_samples = mcmc_samples
        self.additional_seasonalities = additional_seasonalities or []

        self._model: Optional[_Prophet] = None  # type: ignore[type-arg]
        self._last_date: Optional[pd.Timestamp] = None
        self._freq: Optional[str] = None

    def fit(
        self,
        y: pd.Series | np.ndarray,
        datetime_index: Optional[pd.DatetimeIndex] = None,
        freq: str = "D",
    ) -> "ProphetForecaster":
        """Fit Prophet model.

        Parameters
        ----------
        y : 1D array or pd.Series
            If pd.Series with DatetimeIndex, the index is used automatically.
            If np.ndarray, ``datetime_index`` must be provided or a daily
            range starting from today is synthesised.
        datetime_index : pd.DatetimeIndex | None
            Explicit datetime index (used when ``y`` is a plain array).
        freq : str, default "D"
            Frequency string used when synthesising a date range.

        Returns
        -------
        self
        """
        _require_prophet()

        if isinstance(y, pd.Series) and isinstance(y.index, pd.DatetimeIndex):
            idx = y.index
            vals = y.to_numpy(dtype=float)
            self._freq = pd.infer_freq(idx) or freq
        elif datetime_index is not None:
            idx = datetime_index
            vals = np.asarray(y, dtype=float)
            self._freq = pd.infer_freq(idx) or freq
        else:
            n = len(y)
            vals = np.asarray(y, dtype=float)
            idx = pd.date_range("2000-01-01", periods=n, freq=freq)
            self._freq = freq

        df_prophet = pd.DataFrame({"ds": idx, "y": vals})
        self._last_date = df_prophet["ds"].iloc[-1]

        self._model = _Prophet(
            seasonality_mode=self.seasonality_mode,
            interval_width=self.interval_width,
            mcmc_samples=self.mcmc_samples,
        )
        for s in self.additional_seasonalities:
            self._model.add_seasonality(**s)

        import logging
        # Suppress Stan / Prophet verbose output
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        self._model.fit(df_prophet)
        return self

    def predict(
        self,
        prediction_length: int,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    ) -> ForecastResult:
        """Forecast ``prediction_length`` steps ahead.

        Prophet's ``yhat_lower`` / ``yhat_upper`` are mapped to the outermost
        requested quantiles; ``yhat`` becomes the median.

        Parameters
        ----------
        prediction_length : int
        quantiles : sequence of float

        Returns
        -------
        ForecastResult, shape ``[1, len(quantiles), prediction_length]``
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        _require_prophet()

        future = self._model.make_future_dataframe(
            periods=prediction_length, freq=self._freq or "D", include_history=False
        )
        forecast = self._model.predict(future)

        yhat = forecast["yhat"].to_numpy()[:prediction_length]
        lower = forecast["yhat_lower"].to_numpy()[:prediction_length]
        upper = forecast["yhat_upper"].to_numpy()[:prediction_length]

        q_levels = tuple(float(q) for q in quantiles)
        q_arr = _map_quantiles(yhat, lower, upper, q_levels)

        return ForecastResult(
            quantiles=q_arr[np.newaxis, :, :],  # [1, Q, H]
            q_levels=q_levels,
            ids=("y",),
        )


def _map_quantiles(
    median: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    q_levels: tuple,
) -> np.ndarray:
    """Map (lower, median, upper) to arbitrary quantile levels by linear interpolation.

    Treats lower=q=0.1, median=q=0.5, upper=q=0.9 as anchor points and
    linearly interpolates for other levels.
    """
    anchors = np.array([0.1, 0.5, 0.9])
    anchor_vals = np.stack([lower, median, upper], axis=0)  # (3, H)

    q_arr = np.zeros((len(q_levels), len(median)))
    for i, q in enumerate(q_levels):
        q_arr[i] = np.array([
            np.interp(q, anchors, anchor_vals[:, h])
            for h in range(len(median))
        ])
    return q_arr
