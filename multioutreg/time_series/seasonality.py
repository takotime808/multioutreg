# Copyright (c) 2026 takotime808

"""Seasonal-period detection from the autocorrelation function."""

from __future__ import annotations

import numpy as np


def detect_seasonal_period(
    series,
    max_lag: int | None = None,
    threshold: float = 0.3,
) -> int | None:
    """Estimate the dominant seasonal period of a (stationary) series.

    A lag ``L >= 2`` is a candidate when the ACF has a local peak there that
    is above both ``threshold`` and the 95% white-noise bound ``1.96/sqrt(n)``,
    and the ACF dips by at least half that bound somewhere between lag 1 and
    ``L``. The candidate with the largest ACF wins.

    Lag 1 is never a season. The dip rule rejects the slow, monotone ACF decay
    of a persistent series (AR with phi near 1, a trend), where every early
    lag is above ``threshold`` but none is a peak. Pass a differenced series
    when the input is not stationary.

    Parameters
    ----------
    series : array-like
        Time series values; non-finite values are dropped.
    max_lag : int | None
        Largest lag to scan. Default ``min(n // 2, 100)``.
    threshold : float
        Minimum ACF value for a peak.

    Returns
    -------
    int | None
        The period, or ``None`` when there is no clear seasonal peak.
    """
    from statsmodels.tsa.stattools import acf

    x = np.asarray(series, dtype=float).ravel()
    x = x[np.isfinite(x)]
    n = len(x)
    if max_lag is None:
        max_lag = min(n // 2, 100)
    max_lag = min(max_lag, n - 1)
    # A peak at lag L needs lag L + 1 to compare against.
    if max_lag < 3 or np.ptp(x) == 0:
        return None

    r = acf(x, nlags=max_lag, fft=True)
    if not np.all(np.isfinite(r)):
        return None

    bound = max(threshold, 1.96 / np.sqrt(n))
    peaks = [
        lag for lag in range(2, max_lag)
        if r[lag] > bound
        and r[lag] >= r[lag - 1] and r[lag] >= r[lag + 1]
        and r[lag] - r[1:lag].min() >= bound / 2
    ]
    if not peaks:
        return None
    # Multiples of the period peak too, about as high (the ACF estimator
    # shrinks towards 0 with the lag, but noise can still put 2s above s).
    best_r = max(r[lag] for lag in peaks)
    return min(lag for lag in peaks if r[lag] >= 0.8 * best_r)
