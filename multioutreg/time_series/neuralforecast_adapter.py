# Copyright (c) 2025 takotime808

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from multioutreg.time_series.chronos_adapter import ForecastResult

try:
    from neuralforecast import NeuralForecast as _NeuralForecast
    from neuralforecast.models import NBEATS as _NBEATS, NHITS as _NHITS
    _NEURALFORECAST_AVAILABLE = True
except ImportError:
    _NeuralForecast = None  # type: ignore[assignment]
    _NBEATS = None          # type: ignore[assignment]
    _NHITS = None           # type: ignore[assignment]
    _NEURALFORECAST_AVAILABLE = False


def _require_neuralforecast() -> None:
    if not _NEURALFORECAST_AVAILABLE:
        raise ImportError(
            "neuralforecast is required for NeuralForecaster. "
            "Install it with:  pip install neuralforecast"
        )


class NeuralForecaster:
    """Thin sklearn-style wrapper around Nixtla NeuralForecast (N-BEATS / N-HiTS).

    Supports N-BEATS and N-HiTS from the ``neuralforecast`` library.  Both
    models use split-conformal prediction intervals (via NeuralForecast's
    ``val_size`` mechanism) when enough data is available; otherwise the
    lower and upper bounds mirror the point forecast (no uncertainty fan).

    Requires::

        pip install neuralforecast

    Parameters
    ----------
    model_type : str, default "nbeats"
        ``"nbeats"`` (N-BEATS) or ``"nhits"`` (N-HiTS).
    input_size : int, default 24
        Look-back context window fed to the model.
    max_steps : int, default 500
        Maximum training iterations.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> dates = pd.date_range("2022-01-01", periods=200, freq="D")
    >>> y = pd.Series(np.sin(np.linspace(0, 4*np.pi, 200)), index=dates)
    >>> f = NeuralForecaster("nbeats", input_size=24, max_steps=50).fit(y)
    >>> result = f.predict(prediction_length=12)
    >>> result.quantiles.shape  # (1, 3, 12)
    """

    def __init__(
        self,
        model_type: str = "nbeats",
        input_size: int = 24,
        max_steps: int = 500,
    ):
        _require_neuralforecast()
        if model_type not in ("nbeats", "nhits"):
            raise ValueError(
                f"model_type must be 'nbeats' or 'nhits', got {model_type!r}"
            )
        self.model_type = model_type
        self.input_size = input_size
        self.max_steps = max_steps

        self._series: Optional[np.ndarray] = None
        self._freq: str = "D"
        self._last_date: Optional[pd.Timestamp] = None

    def fit(
        self,
        y: "pd.Series | np.ndarray",
        freq: str = "D",
    ) -> "NeuralForecaster":
        """Store the training series; actual model training is deferred to predict().

        NeuralForecast requires the forecast horizon (``h``) at model
        construction time, so training is deferred until predict() is called
        with the desired ``prediction_length``.

        Parameters
        ----------
        y : 1D array or pd.Series
            If pd.Series with DatetimeIndex, the index is used automatically.
        freq : str, default "D"
            Frequency string used when synthesising a date range.

        Returns
        -------
        self
        """
        _require_neuralforecast()
        if isinstance(y, pd.Series) and isinstance(y.index, pd.DatetimeIndex):
            self._last_date = y.index[-1]
            self._freq = pd.infer_freq(y.index) or freq
            self._series = y.to_numpy(dtype=float)
        else:
            self._series = np.asarray(y, dtype=float)
            self._freq = freq
            self._last_date = None
        return self

    def predict(
        self,
        prediction_length: int,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    ) -> ForecastResult:
        """Train and forecast ``prediction_length`` steps ahead.

        If enough data is available (``len(y) - input_size > 2 *
        prediction_length``), split-conformal prediction intervals are
        computed via NeuralForecast's ``val_size`` mechanism.  The resulting
        ``lo-80`` / ``hi-80`` columns are mapped to the requested quantiles
        using linear interpolation (same strategy as ProphetForecaster).

        Parameters
        ----------
        prediction_length : int
        quantiles : sequence of float

        Returns
        -------
        ForecastResult, shape ``[1, len(quantiles), prediction_length]``
        """
        if self._series is None:
            raise RuntimeError("Call fit() before predict().")
        _require_neuralforecast()

        q_levels = tuple(float(q) for q in quantiles)
        n = len(self._series)

        # Build long-format DataFrame required by NeuralForecast
        if self._last_date is not None:
            dates = pd.date_range(end=self._last_date, periods=n, freq=self._freq)
        else:
            dates = pd.date_range("2000-01-01", periods=n, freq=self._freq)

        df_nf = pd.DataFrame({
            "unique_id": "y",
            "ds": dates,
            "y": self._series,
        })

        # Conformal intervals need val_size >= prediction_length and enough train data
        val_size = max(prediction_length, min(prediction_length * 2, n // 5))
        use_conformal = (
            val_size >= prediction_length
            and (n - val_size) >= self.input_size + 1
        )
        if not use_conformal:
            val_size = 0

        model_cls = _NBEATS if self.model_type == "nbeats" else _NHITS

        model = model_cls(
            h=prediction_length,
            input_size=self.input_size,
            max_steps=self.max_steps,
        )
        # NeuralForecast names forecast columns after the model class name
        model_name = model.__class__.__name__

        import logging
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
        logging.getLogger("lightning").setLevel(logging.WARNING)

        nf = _NeuralForecast(models=[model], freq=self._freq)

        if use_conformal:
            nf.fit(df_nf, val_size=val_size)
            forecast_df = nf.predict(level=[80])
            yhat = forecast_df[model_name].to_numpy()[:prediction_length]
            lower = forecast_df[f"{model_name}-lo-80"].to_numpy()[:prediction_length]
            upper = forecast_df[f"{model_name}-hi-80"].to_numpy()[:prediction_length]
        else:
            nf.fit(df_nf)
            forecast_df = nf.predict()
            yhat = forecast_df[model_name].to_numpy()[:prediction_length]
            # No uncertainty available — replicate point forecast for all quantiles
            lower = yhat.copy()
            upper = yhat.copy()

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
