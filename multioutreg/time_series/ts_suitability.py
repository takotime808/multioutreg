# Copyright (c) 2026 takotime808

"""Statistical tests to determine whether uploaded data is suitable for time series modeling."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.signal import detrend

from multioutreg.time_series.seasonality import detect_seasonal_period


def check_ts_suitability(
    df: pd.DataFrame,
    target_col: str,
    datetime_col: str | None = None,
    min_length: int = 30,
) -> dict:
    """Run statistical tests to determine if data is suitable for time series modeling.

    Runs four tests:
    - ADF (Augmented Dickey-Fuller): null = unit root. KPSS: null = level
      stationary. The series counts as stationary only when ADF rejects and
      KPSS does not; otherwise it is differenced once for the next two tests.
    - Ljung-Box: checks for significant autocorrelation (serial dependence)
      in the stationary series. On levels, any trend or random walk would pass,
      though a naive forecast is already optimal for a random walk.
    - ACF seasonality scan: estimates a dominant seasonal period if present
      (see :func:`multioutreg.time_series.seasonality.detect_seasonal_period`),
      on the linearly detrended levels and then the differences when the
      series is not stationary.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    target_col : str
        Column to analyse.
    datetime_col : str | None
        If provided, data is sorted by this column before testing.
    min_length : int
        Minimum number of non-null observations required to run tests (default 30).

    Returns
    -------
    dict with keys:
        n_obs, adf_statistic, adf_pvalue, kpss_statistic, kpss_pvalue,
        is_stationary, tested_on ("levels" or "first differences"),
        ljungbox_pvalue, has_autocorrelation, seasonal_period,
        min_length_ok, suitable, recommendation.
        On error: suitable=False, error=<message>.
    """
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except ImportError as exc:
        return {"suitable": False, "error": f"statsmodels not available: {exc}"}

    try:
        if target_col not in df.columns:
            return {"suitable": False, "error": f"Column '{target_col}' not found in dataframe."}

        if datetime_col and datetime_col in df.columns:
            df = df.sort_values(datetime_col)

        series = df[target_col].dropna()
        n_obs = int(len(series))
        min_length_ok = n_obs >= min_length

        if not min_length_ok:
            return {
                "n_obs": n_obs,
                "min_length_ok": False,
                "suitable": False,
                "recommendation": (
                    f"Insufficient data (n={n_obs} < {min_length}). "
                    f"At least {min_length} observations are required for time series modeling."
                ),
            }

        # ADF (null: unit root) and KPSS (null: stationary) together
        adf_result = adfuller(series)
        adf_stat = float(adf_result[0])
        adf_pvalue = float(adf_result[1])
        with warnings.catch_warnings():
            # KPSS p-values are interpolated from a table clipped to [0.01, 0.1].
            warnings.simplefilter("ignore")
            kpss_stat, kpss_pvalue, *_ = kpss(series, regression="c", nlags="auto")
        kpss_stat, kpss_pvalue = float(kpss_stat), float(kpss_pvalue)
        is_stationary = adf_pvalue <= 0.05 and kpss_pvalue > 0.05

        # Serial dependence and seasonality are judged on a stationary series
        tested = series if is_stationary else series.diff().dropna()
        tested_on = "levels" if is_stationary else "first differences"

        # Ljung-Box test (lag=10, fewer for short series)
        lb_lags = max(1, min(10, len(tested) // 5))
        lb_result = acorr_ljungbox(tested, lags=[lb_lags], return_df=True)
        lb_pvalue = float(lb_result["lb_pvalue"].iloc[0])
        has_autocorrelation = bool(np.isfinite(lb_pvalue) and lb_pvalue < 0.05)

        # Differencing damps a slow season below the noise, so a non-stationary
        # series is first scanned with only its linear trend removed.
        if is_stationary:
            seasonal_period = detect_seasonal_period(series)
        else:
            seasonal_period = (
                detect_seasonal_period(detrend(series.to_numpy(dtype=float)))
                or detect_seasonal_period(tested)
            )

        suitable = min_length_ok and has_autocorrelation

        # Build human-readable recommendation
        if suitable:
            period_str = f" Detected seasonal period: {seasonal_period}." if seasonal_period else ""
            stat_str = "stationary" if is_stationary else "non-stationary (differencing will be applied)"
            recommendation = (
                f"Data shows significant autocorrelation in its {tested_on} "
                f"(Ljung-Box p={lb_pvalue:.4f}) "
                f"with {n_obs} observations. Series is {stat_str}.{period_str} "
                f"Suitable for ARIMA / SARIMA / LSTM modeling."
            )
        else:
            recommendation = (
                f"Data does not show significant autocorrelation in its {tested_on} "
                f"(Ljung-Box p={lb_pvalue:.4f}, n={n_obs}). "
                f"Time series models may not outperform a simple baseline — "
                f"surrogate models are likely a better fit."
            )

        return {
            "n_obs": n_obs,
            "adf_statistic": round(adf_stat, 6),
            "adf_pvalue": round(adf_pvalue, 6),
            "kpss_statistic": round(kpss_stat, 6),
            "kpss_pvalue": round(kpss_pvalue, 6),
            "is_stationary": is_stationary,
            "tested_on": tested_on,
            "ljungbox_pvalue": round(lb_pvalue, 6),
            "has_autocorrelation": has_autocorrelation,
            "seasonal_period": seasonal_period,
            "min_length_ok": min_length_ok,
            "suitable": suitable,
            "recommendation": recommendation,
        }

    except Exception as exc:
        return {"suitable": False, "error": str(exc)}
