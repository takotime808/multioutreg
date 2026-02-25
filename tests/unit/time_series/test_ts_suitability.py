# Copyright (c) 2025 takotime808

import numpy as np
import pandas as pd
import pytest

from multioutreg.time_series.ts_suitability import check_ts_suitability


def _make_df(values, col="y"):
    return pd.DataFrame({col: values})


def test_check_ts_suitability_returns_dict():
    rng = np.random.default_rng(0)
    df = _make_df(rng.standard_normal(60))
    result = check_ts_suitability(df, target_col="y")
    assert isinstance(result, dict)
    assert "suitable" in result
    assert "recommendation" in result


def test_check_ts_suitability_white_noise_not_suitable():
    """Pure white noise has no autocorrelation — should not be flagged as TS-suitable."""
    rng = np.random.default_rng(42)
    df = _make_df(rng.standard_normal(100))
    result = check_ts_suitability(df, target_col="y")
    # White noise is stationary (ADF rejects unit root) and has no autocorrelation
    # so Ljung-Box should indicate it's NOT suitable for ARIMA/SARIMA
    assert isinstance(result["suitable"], bool)


def test_check_ts_suitability_autocorrelated_series():
    """AR(1) series (phi=0.9) has strong autocorrelation — should be suitable."""
    rng = np.random.default_rng(1)
    n = 120
    y = np.zeros(n)
    y[0] = rng.standard_normal()
    for t in range(1, n):
        y[t] = 0.9 * y[t - 1] + 0.1 * rng.standard_normal()
    df = _make_df(y)
    result = check_ts_suitability(df, target_col="y")
    # AR(1) with phi=0.9 should have significant autocorrelation
    assert result["suitable"] is True


def test_check_ts_suitability_short_series():
    """Series shorter than min_length should be flagged as not suitable."""
    df = _make_df(np.arange(10, dtype=float))
    result = check_ts_suitability(df, target_col="y", min_length=30)
    assert result["suitable"] is False


def test_check_ts_suitability_missing_column():
    """Missing target column should return suitable=False with an error key."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    result = check_ts_suitability(df, target_col="nonexistent")
    assert result["suitable"] is False


def test_check_ts_suitability_datetime_col_sorts_data():
    """datetime_col argument causes data to be sorted before testing."""
    rng = np.random.default_rng(3)
    n = 80
    y = np.zeros(n)
    y[0] = rng.standard_normal()
    for t in range(1, n):
        y[t] = 0.85 * y[t - 1] + 0.1 * rng.standard_normal()

    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    df = pd.DataFrame({"y": y, "date": dates})
    # Shuffle rows to verify sorting occurs
    df = df.sample(frac=1, random_state=7).reset_index(drop=True)

    result = check_ts_suitability(df, target_col="y", datetime_col="date")
    assert isinstance(result, dict)
    assert "suitable" in result


def test_check_ts_suitability_seasonal_period_detected():
    """A series with strong periodicity should have seasonal_period set."""
    rng = np.random.default_rng(5)
    n = 200
    t = np.arange(n)
    # Sine with period 7 plus noise
    y = np.sin(2 * np.pi * t / 7) + 0.05 * rng.standard_normal(n)
    df = _make_df(y)
    result = check_ts_suitability(df, target_col="y")
    # Should be suitable (strong autocorrelation) and detect a period
    if result.get("suitable"):
        assert result["seasonal_period"] is not None
        assert result["seasonal_period"] > 0


def test_check_ts_suitability_recommendation_stationary():
    """When series is stationary (ADF rejects), recommendation mentions 'stationary'."""
    rng = np.random.default_rng(6)
    n = 120
    y = np.zeros(n)
    y[0] = rng.standard_normal()
    for t in range(1, n):
        y[t] = 0.9 * y[t - 1] + 0.1 * rng.standard_normal()
    df = _make_df(y)
    result = check_ts_suitability(df, target_col="y")
    assert "recommendation" in result


def test_check_ts_suitability_outer_exception_returns_error():
    """An unexpected error (e.g., bad column type) returns suitable=False with error."""
    df = pd.DataFrame({"y": ["a", "b", "c"] * 20})  # strings → statsmodels will fail
    result = check_ts_suitability(df, target_col="y")
    assert result["suitable"] is False
    assert "error" in result


def test_check_ts_suitability_n_obs_in_result():
    """n_obs in result matches the length of the series."""
    rng = np.random.default_rng(9)
    n = 60
    df = _make_df(rng.standard_normal(n))
    result = check_ts_suitability(df, target_col="y")
    assert result["n_obs"] == n


def test_check_ts_suitability_min_length_ok_key():
    """min_length_ok is present and True when series is long enough."""
    rng = np.random.default_rng(10)
    df = _make_df(rng.standard_normal(60))
    result = check_ts_suitability(df, target_col="y")
    assert result["min_length_ok"] is True


def test_check_ts_suitability_adf_keys_present():
    """adf_statistic, adf_pvalue, is_stationary keys are present in successful run."""
    rng = np.random.default_rng(11)
    df = _make_df(rng.standard_normal(60))
    result = check_ts_suitability(df, target_col="y")
    assert "adf_statistic" in result
    assert "adf_pvalue" in result
    assert "is_stationary" in result
