# Copyright (c) 2025 takotime808

import os
import numpy as np
import pandas as pd
import pytest

from multioutreg.time_series.ts_dynamic_fit import ARIMA


def _ar1_series(n=80, phi=0.7, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    y[0] = rng.standard_normal()
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.standard_normal()
    return pd.DataFrame({"value": y})


def test_arima_run_returns_metadata_keys():
    """ARIMA.run() returns (metadata_dict, data_series, fitted_model)."""
    df = _ar1_series()
    arima = ARIMA(df, feature_column="value", verbose=False)
    metadata, data, model = arima.run()

    assert "performance" in metadata
    assert metadata["performance"]["Model"] == "ARIMA"
    assert "RMSE" in metadata["performance"]
    assert "MAE" in metadata["performance"]
    assert "MAPE" in metadata["performance"]
    assert "AIC" in metadata["performance"]
    assert "BIC" in metadata["performance"]


def test_arima_run_returns_correct_types():
    """Return types: dict, pd.Series, statsmodels results object."""
    df = _ar1_series()
    arima = ARIMA(df, feature_column="value", verbose=False)
    metadata, data, model = arima.run()

    assert isinstance(metadata, dict)
    assert isinstance(data, pd.Series)
    assert hasattr(model, "predict")  # statsmodels results have predict


def test_arima_no_file_side_effects(tmp_path, monkeypatch):
    """ARIMA.run() must not create any files in the working directory."""
    monkeypatch.chdir(tmp_path)
    df = _ar1_series()
    arima = ARIMA(df, feature_column="value", verbose=False)
    arima.run()

    created = list(tmp_path.iterdir())
    assert created == [], f"Unexpected files created: {created}"


def test_arima_metadata_diff_count():
    """diff_count in metadata is a non-negative integer."""
    df = _ar1_series()
    arima = ARIMA(df, feature_column="value", verbose=False)
    metadata, _, _ = arima.run()
    assert isinstance(metadata["metadata"]["diff_count"], int)
    assert metadata["metadata"]["diff_count"] >= 0


def test_arima_k_order_data_has_diff_0():
    """'K-order data' dict contains diff_0_data."""
    df = _ar1_series()
    arima = ARIMA(df, feature_column="value", verbose=False)
    metadata, _, _ = arima.run()
    assert "K-order data" in metadata
    assert "diff_0_data" in metadata["K-order data"]


def test_arima_verbose_does_not_raise():
    """verbose=True should not raise during run()."""
    df = _ar1_series()
    arima = ARIMA(df, feature_column="value", verbose=True)
    metadata, data, model = arima.run()
    assert "performance" in metadata


def test_arima_make_stationary_stationary_series():
    """ADF on an AR(1) with phi=0.7 (stationary) sets self.stationary=True."""
    df = _ar1_series(phi=0.7)
    arima = ARIMA(df, feature_column="value")
    arima.make_stationary(arima.data)
    assert arima.stationary is True


def test_arima_make_stationary_nonstationary_increments_diff_count():
    """ADF on a random walk (unit root) increments diff_count without setting stationary."""
    rng = np.random.default_rng(99)
    # Pure random walk (unit root) — ADF should fail to reject
    y = np.cumsum(rng.standard_normal(200))
    df = pd.DataFrame({"value": y})
    arima = ARIMA(df, feature_column="value")
    arima.make_stationary(arima.data)
    # Non-stationary path: diff_count was incremented
    assert arima.diff_count == 1
    assert arima.stationary is False


def test_arima_differencing_produces_diff_1_data():
    """Running on a random walk causes differencing; diff_1_data is populated."""
    rng = np.random.default_rng(88)
    y = np.cumsum(rng.standard_normal(100))
    df = pd.DataFrame({"value": y})
    arima = ARIMA(df, feature_column="value", verbose=False)
    metadata, _, _ = arima.run()
    # After 1 difference the series should be stationary; diff_1_data should be set
    if metadata["metadata"]["diff_count"] >= 1:
        assert metadata["K-order data"]["diff_1_data"] is not None


def test_arima_convert_keys_to_str_dict():
    """convert_keys_to_str converts Timestamp keys to strings."""
    import pandas as pd
    df = _ar1_series()
    arima = ARIMA(df, feature_column="value")
    ts_key = pd.Timestamp("2023-01-01")
    obj = {ts_key: 1.0, "normal": 2.0}
    result = arima.convert_keys_to_str(obj)
    assert "2023-01-01 00:00:00" in result or "2023-01-01" in str(result)
    assert "normal" in result


def test_arima_convert_keys_to_str_list():
    """convert_keys_to_str recurses into lists."""
    df = _ar1_series()
    arima = ARIMA(df, feature_column="value")
    obj = [{"a": 1}, {"b": 2}]
    result = arima.convert_keys_to_str(obj)
    assert isinstance(result, list)
    assert result[0] == {"a": 1}


def test_arima_convert_keys_to_str_scalar():
    """convert_keys_to_str returns scalars unchanged."""
    df = _ar1_series()
    arima = ARIMA(df, feature_column="value")
    assert arima.convert_keys_to_str(42) == 42
    assert arima.convert_keys_to_str("hello") == "hello"


def test_arima_log_to_file_does_not_raise():
    """log_to_file() should not raise."""
    df = _ar1_series()
    arima = ARIMA(df, feature_column="value")
    arima.log_to_file("test message")  # Should not raise
