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

    A lag ``L >= 2`` is a candidate when the ACF peaks there -- it is the
    largest value within ``max(1, L // 10)`` lags either side -- above both
    ``threshold`` and the 95% white-noise bound ``1.96/sqrt(n)``, and the ACF
    dips by at least half that bound somewhere between lag 1 and ``L``. The
    candidate with the largest ACF wins.

    Lag 1 is never a season. The dip rule rejects the slow, monotone ACF decay
    of a persistent series (AR with phi near 1, a trend), where every early
    lag is above ``threshold`` but none is a peak. The +-10% window matters
    for long periods (365 for daily data): the ACF of a cycle stays within
    ``cos(36deg) ~ 0.8`` of its peak over that window, so noise wiggles on the
    broad hump would otherwise pass as peaks and the earliest one would win
    (332 instead of 365 on Melbourne temperatures). Below lag 20 the window
    is +-1, an ordinary local maximum. Pass a differenced series when the
    input is not stationary.

    Parameters
    ----------
    series : array-like
        Time series values; non-finite values are dropped.
    max_lag : int | None
        Largest lag to scan. Default ``n // 2``: a period needs two full
        cycles to show in the ACF.
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
        max_lag = n // 2
    max_lag = min(max_lag, n - 1)
    # A peak at lag L needs lag L + 1 to compare against.
    if max_lag < 3 or np.ptp(x) == 0:
        return None

    # Scan past max_lag so the window of the last candidates is complete.
    r = acf(x, nlags=min(max_lag + max_lag // 10, n - 1), fft=True)
    if not np.all(np.isfinite(r)):
        return None

    bound = max(threshold, 1.96 / np.sqrt(n))
    dip = np.minimum.accumulate(r[1:])   # dip[k] = min(r[1 : k + 2])
    peaks = []
    for lag in range(2, max_lag):
        if r[lag] <= bound or r[lag] - dip[lag - 2] < bound / 2:
            continue
        w = max(1, lag // 10)
        if r[lag] >= r[max(1, lag - w): lag + w + 1].max():
            peaks.append(lag)
    if not peaks:
        return None
    # Multiples of the period peak too, about as high (the ACF estimator
    # shrinks towards 0 with the lag, but noise can still put 2s above s).
    best_r = max(r[lag] for lag in peaks)
    return min(lag for lag in peaks if r[lag] >= 0.8 * best_r)
