# Copyright (c) 2025 takotime808

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Sequence, Tuple


def make_lag_features(
    series: np.ndarray | pd.Series,
    n_lags: int,
    horizon: int = 1,
    include_time_features: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a 1D series into tabular (X_lag, y) for supervised learning.

    Each row of X_lag is [y[t-n_lags], ..., y[t-1]].
    y[i] contains the targets: y[t], ..., y[t+horizon-1].

    Parameters
    ----------
    series : 1D array-like of floats
    n_lags : int
        Number of lagged features.
    horizon : int, default 1
        Prediction horizon.  horizon=1 => single-step; horizon>1 => multi-step.
    include_time_features : bool, default False
        If True and series is a pd.Series with DatetimeIndex, appends
        sin/cos-encoded day-of-week and month as additional columns.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_lags [+ n_time_feats])
    y : np.ndarray, shape (n_samples, horizon)
    """
    if isinstance(series, pd.Series):
        index = series.index if include_time_features else None
        arr = series.to_numpy(dtype=float)
    else:
        arr = np.asarray(series, dtype=float)
        index = None

    n = len(arr)
    n_samples = n - n_lags - horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"Series length {n} is too short for n_lags={n_lags} and horizon={horizon}. "
            f"Need at least {n_lags + horizon + 1} observations."
        )

    X_lag = np.lib.stride_tricks.sliding_window_view(arr, n_lags)[:n_samples]
    y = np.lib.stride_tricks.sliding_window_view(arr[n_lags:], horizon)[:n_samples]

    if include_time_features and index is not None and hasattr(index, "dayofweek"):
        # sin/cos encoding for day-of-week (period=7) and month (period=12)
        t_idx = index[n_lags: n_lags + n_samples]
        dow = t_idx.dayofweek.to_numpy(dtype=float)
        mon = (t_idx.month.to_numpy(dtype=float) - 1)
        time_feats = np.column_stack([
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * mon / 12),
            np.cos(2 * np.pi * mon / 12),
        ])
        X_lag = np.hstack([X_lag, time_feats])

    return X_lag, y


def rolling_window_features(
    series: np.ndarray | pd.Series,
    windows: Sequence[int] = (4, 8, 24),
) -> np.ndarray:
    """Compute rolling mean and std columns for a series.

    For each window w, produces two columns: rolling mean and rolling std
    (with min_periods=1 so no leading NaNs).

    Parameters
    ----------
    series : 1D array-like
    windows : sequence of window sizes

    Returns
    -------
    np.ndarray, shape (len(series), 2 * len(windows))
        Columns: [mean_w0, std_w0, mean_w1, std_w1, ...]
    """
    s = pd.Series(np.asarray(series, dtype=float))
    cols = []
    for w in windows:
        cols.append(s.rolling(w, min_periods=1).mean().to_numpy())
        cols.append(s.rolling(w, min_periods=1).std(ddof=0).fillna(0).to_numpy())
    return np.column_stack(cols)
