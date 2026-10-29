# Copyright (c) 2026 takotime808

"""Audit items 21-22: seasonal period detection and time-series suitability."""

import numpy as np
import pandas as pd
import pytest

from multioutreg.time_series.seasonality import detect_seasonal_period
from multioutreg.time_series.ts_suitability import check_ts_suitability
from multioutreg.time_series.ts_dynamic_fit.algs.sarima import SARIMA


def _ar1(phi, n, seed):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    e = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def _seasonal(period, n, seed, noise=0.5, trend=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return trend * t + np.sin(2 * np.pi * t / period) + noise * rng.standard_normal(n)


# --- detect_seasonal_period -------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_no_period_for_white_noise_random_walk_and_trend(seed):
    rng = np.random.default_rng(seed)
    n = 200
    assert detect_seasonal_period(rng.standard_normal(n)) is None
    assert detect_seasonal_period(np.cumsum(rng.standard_normal(n))) is None
    assert detect_seasonal_period(0.5 * np.arange(n) + rng.standard_normal(n)) is None


def test_persistent_ar1_is_not_seasonal():
    # The old rule returned the first lag with ACF > 0.3: s = 1 (SARIMA) or 2 (suitability).
    hits = [detect_seasonal_period(_ar1(0.9, 200, s)) for s in range(20)]
    assert 1 not in hits and 2 not in hits
    assert sum(h is None for h in hits) >= 18


@pytest.mark.parametrize("period", [4, 7, 12])
def test_finds_true_period_not_a_multiple(period):
    found = [detect_seasonal_period(_seasonal(period, 240, s)) for s in range(10)]
    assert sum(f == period for f in found) >= 9


def test_constant_and_short_series():
    assert detect_seasonal_period(np.ones(50)) is None
    assert detect_seasonal_period([1.0, 2.0, 3.0]) is None


@pytest.mark.parametrize("period, n", [(52, 260), (132, 700), (365, 1100)])
def test_finds_long_periods(period, n):
    # The scan used to stop at lag 100, so 132 (sunspots, monthly) and 365
    # (yearly, daily) were never found; with the cap lifted, the earliest noise
    # wiggle on the broad ACF hump won instead (~10% short: 330 for 365).
    found = [detect_seasonal_period(_seasonal(period, n, s)) for s in range(10)]
    assert all(f is not None and abs(f - period) <= 0.05 * period for f in found), found


# --- SARIMA.detect_seasonal_period (item 22) --------------------------------

def _sarima(values):
    return SARIMA(pd.DataFrame({"y": values}), feature_column="y")


def test_sarima_never_returns_lag_one():
    x = _ar1(0.8, 200, 0)
    model = _sarima(x)
    assert model.detect_seasonal_period(x) == 0


def test_sarima_detects_weekly_season():
    x = _seasonal(7, 210, 0)
    assert _sarima(x).detect_seasonal_period(x) == 7


# --- check_ts_suitability (item 21) -----------------------------------------

def _check(values):
    return check_ts_suitability(pd.DataFrame({"y": values}), "y")


def test_random_walk_tested_on_differences_and_not_suitable():
    rng = np.random.default_rng(0)
    res = _check(np.cumsum(rng.standard_normal(300)))
    assert "error" not in res
    assert res["is_stationary"] is False
    assert res["tested_on"] == "first differences"
    # Levels would pass Ljung-Box trivially; the increments are white noise.
    assert res["has_autocorrelation"] is False
    assert res["suitable"] is False


def test_stationary_ar1_is_suitable():
    res = _check(_ar1(0.6, 300, 1))
    assert res["is_stationary"] is True
    assert res["tested_on"] == "levels"
    assert res["suitable"] is True
    assert res["seasonal_period"] is None
    assert {"kpss_statistic", "kpss_pvalue", "adf_pvalue"} <= res.keys()


def test_trend_is_flagged_non_stationary_by_kpss():
    rng = np.random.default_rng(2)
    res = _check(0.05 * np.arange(300) + rng.standard_normal(300))
    assert res["kpss_pvalue"] <= 0.05
    assert res["is_stationary"] is False
    assert res["seasonal_period"] is None


def test_seasonal_with_trend_reports_period():
    res = _check(_seasonal(12, 240, 3, trend=0.3))
    assert res["seasonal_period"] == 12
    assert res["suitable"] is True


def test_short_series_unchanged():
    res = _check(np.arange(10.0))
    assert res["min_length_ok"] is False and res["suitable"] is False
