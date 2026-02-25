# Copyright (c) 2025 takotime808

import numpy as np
import pandas as pd
import pytest

from multioutreg.time_series.ts_dynamic_fit import SARIMA


def _seasonal_ar_series(n=120, period=4, phi=0.6, seed=42):
    """Seasonal AR(1) — stationary and has a clear ACF peak at `period`.

    The peak at lag `period` exceeds the 0.3 threshold in detect_seasonal_period,
    so SARIMA.run() receives a valid non-zero s_value from the grid search.
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for t in range(period, n):
        y[t] = phi * y[t - period] + 0.4 * rng.standard_normal()
    return pd.DataFrame({"value": y})


def test_sarima_run_returns_metadata_keys():
    """SARIMA.run() returns a dict with expected performance keys."""
    df = _seasonal_ar_series()
    sarima = SARIMA(df, feature_column="value", verbose=False)
    metadata, data, model = sarima.run()

    assert "performance" in metadata
    assert metadata["performance"]["Model"] == "SARIMA"
    assert "RMSE" in metadata["performance"]
    assert "MAE" in metadata["performance"]
    assert "MAPE" in metadata["performance"]
    assert "AIC" in metadata["performance"]
    assert "BIC" in metadata["performance"]


def test_sarima_run_returns_correct_types():
    """Return types: dict, pd.Series, statsmodels results object."""
    df = _seasonal_ar_series()
    sarima = SARIMA(df, feature_column="value", verbose=False)
    metadata, data, model = sarima.run()

    assert isinstance(metadata, dict)
    assert isinstance(data, pd.Series)
    assert hasattr(model, "predict")


def test_sarima_diff_count_is_non_negative():
    """diff_count in metadata is a non-negative integer."""
    df = _seasonal_ar_series()
    sarima = SARIMA(df, feature_column="value", verbose=False)
    metadata, _, _ = sarima.run()
    assert isinstance(metadata["metadata"]["diff_count"], int)
    assert metadata["metadata"]["diff_count"] >= 0


def test_sarima_no_file_side_effects(tmp_path, monkeypatch):
    """SARIMA.run() must not create any files in the working directory."""
    monkeypatch.chdir(tmp_path)
    df = _seasonal_ar_series()
    sarima = SARIMA(df, feature_column="value", verbose=False)
    sarima.run()

    created = list(tmp_path.iterdir())
    assert created == [], f"Unexpected files created: {created}"


def test_sarima_detect_seasonal_period_no_seasonality():
    """High threshold on white noise returns 0 (no strong peak)."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal(100)
    df = pd.DataFrame({"value": data})
    sarima = SARIMA(df, feature_column="value")
    # Threshold=1.0 ensures no lag can exceed ACF=1.0 (except lag=0)
    s = sarima.detect_seasonal_period(data, threshold=1.0)
    assert s == 0


def test_sarima_detect_seasonal_period_finds_peak():
    """Sine wave with period 12 should yield a seasonal period > 0."""
    t = np.arange(240)
    data = np.sin(2 * np.pi * t / 12) + 0.05 * np.random.default_rng(1).standard_normal(240)
    df = pd.DataFrame({"value": data})
    sarima = SARIMA(df, feature_column="value")
    s = sarima.detect_seasonal_period(data, threshold=0.3)
    assert s > 0


def test_sarima_detect_seasonal_period_max_lag():
    """max_lag parameter is respected — no lag beyond max_lag is returned."""
    t = np.arange(200)
    data = np.sin(2 * np.pi * t / 20)
    df = pd.DataFrame({"value": data})
    sarima = SARIMA(df, feature_column="value")
    s = sarima.detect_seasonal_period(data, max_lag=5, threshold=0.1)
    # Period is 20 but max_lag=5, so either returns within [0, 5] or 0
    assert s <= 5


def test_sarima_make_stationary_stationary_data():
    """ADF on a stationary series sets self.stationary=True."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal(100)
    df = pd.DataFrame({"value": data})
    sarima = SARIMA(df, feature_column="value")
    sarima.make_stationary(sarima.data)
    assert sarima.stationary is True


def test_sarima_k_order_data_keys_present():
    """The 'K-order data' section of metadata has diff_0_data."""
    df = _seasonal_ar_series()
    sarima = SARIMA(df, feature_column="value", verbose=False)
    metadata, _, _ = sarima.run()
    assert "K-order data" in metadata
    assert "diff_0_data" in metadata["K-order data"]
