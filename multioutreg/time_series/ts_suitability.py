# Copyright (c) 2025 takotime808

"""Statistical tests to determine whether uploaded data is suitable for time series modeling."""

from __future__ import annotations

import pandas as pd


def check_ts_suitability(
    df: pd.DataFrame,
    target_col: str,
    datetime_col: str | None = None,
    min_length: int = 30,
) -> dict:
    """Run statistical tests to determine if data is suitable for time series modeling.

    Runs three tests:
    - ADF (Augmented Dickey-Fuller): checks for stationarity.
    - Ljung-Box: checks for significant autocorrelation (serial dependence).
    - ACF seasonality scan: estimates a dominant seasonal period if present.

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
        n_obs, adf_statistic, adf_pvalue, is_stationary,
        ljungbox_pvalue, has_autocorrelation, seasonal_period,
        min_length_ok, suitable, recommendation.
        On error: suitable=False, error=<message>.
    """
    try:
        from statsmodels.tsa.stattools import adfuller, acf
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

        # ADF test
        adf_result = adfuller(series)
        adf_stat = float(adf_result[0])
        adf_pvalue = float(adf_result[1])
        is_stationary = adf_pvalue <= 0.05

        # Ljung-Box test (lag=10)
        lb_result = acorr_ljungbox(series, lags=[10], return_df=True)
        lb_pvalue = float(lb_result["lb_pvalue"].iloc[0])
        has_autocorrelation = lb_pvalue < 0.05

        # Seasonality: scan ACF for first lag > 0.3 (lag 1 excluded as trivial)
        max_lag = min(n_obs // 2, 100)
        seasonal_period: int | None = None
        if max_lag >= 2:
            acf_vals = acf(series, nlags=max_lag, fft=True)
            for lag in range(2, max_lag + 1):
                if abs(acf_vals[lag]) > 0.3:
                    seasonal_period = lag
                    break

        suitable = min_length_ok and has_autocorrelation

        # Build human-readable recommendation
        if suitable:
            period_str = f" Detected seasonal period: {seasonal_period}." if seasonal_period else ""
            stat_str = "stationary" if is_stationary else "non-stationary (differencing will be applied)"
            recommendation = (
                f"Data shows significant autocorrelation (Ljung-Box p={lb_pvalue:.4f}) "
                f"with {n_obs} observations. Series is {stat_str}.{period_str} "
                f"Suitable for ARIMA / SARIMA / LSTM modeling."
            )
        else:
            recommendation = (
                f"Data does not show significant autocorrelation "
                f"(Ljung-Box p={lb_pvalue:.4f}, n={n_obs}). "
                f"Time series models may not outperform a simple baseline — "
                f"surrogate models are likely a better fit."
            )

        return {
            "n_obs": n_obs,
            "adf_statistic": round(adf_stat, 6),
            "adf_pvalue": round(adf_pvalue, 6),
            "is_stationary": is_stationary,
            "ljungbox_pvalue": round(lb_pvalue, 6),
            "has_autocorrelation": has_autocorrelation,
            "seasonal_period": seasonal_period,
            "min_length_ok": min_length_ok,
            "suitable": suitable,
            "recommendation": recommendation,
        }

    except Exception as exc:
        return {"suitable": False, "error": str(exc)}
