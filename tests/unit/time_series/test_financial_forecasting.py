# Copyright (c) 2025 takotime808

import pytest
import numpy as np
import pandas as pd

from multioutreg.time_series.financial import (
    load_financial_csv,
    forecast_with_chronos,
)

try:  # determine if chronos is available and model can be loaded
    from multioutreg.time_series.chronos_adapter import ChronosForecaster
    try:
        ChronosForecaster("amazon/chronos-bolt-tiny")
    except Exception:
        _CHRONOS = False
    else:
        _CHRONOS = True
except Exception:
    _CHRONOS = False


def test_load_financial_csv(tmp_path):
    data = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=5, freq="D"),
        "Close": [100, 101, 102, 103, 104],
    })
    csv = tmp_path / "prices.csv"
    data.to_csv(csv, index=False)
    s = load_financial_csv(csv)
    assert list(s.index) == list(data["Date"])
    assert np.allclose(s.to_numpy(), data["Close"].to_numpy())


@pytest.mark.skipif(not _CHRONOS, reason="chronos-forecasting not installed")
def test_forecast_with_chronos(tmp_path):
    data = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=40, freq="D"),
        "Close": np.linspace(100, 120, 40),
    })
    csv = tmp_path / "prices.csv"
    data.to_csv(csv, index=False)
    series = load_financial_csv(csv)
    res = forecast_with_chronos(series, horizon=5, model_name="amazon/chronos-bolt-tiny")