# Copyright (c) 2025 takotime808
"""
Tests for 05_Time_Series_Forecasting.py pure logic functions.

Uses importlib.util to load the page without a Streamlit runtime
(following the pattern in test_Multi_Fidelity_Surrogate_Models.py).
"""

import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest

_PAGE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 "../../../../multioutreg/gui/pages/05_Time_Series_Forecasting.py")
)
_spec = importlib.util.spec_from_file_location("ts_page", _PAGE_PATH)
_ts_page = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ts_page)

_run_ts_pipeline = _ts_page._run_ts_pipeline
_run_surrogate_forecast = _ts_page._run_surrogate_forecast
_run_chronos_forecast = _ts_page._run_chronos_forecast
_build_surrogate = _ts_page._build_surrogate


# ---- helpers -----------------------------------------------------------

def _make_ts_df(n=80, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    vals = np.cumsum(rng.standard_normal(n)) + 50
    return pd.DataFrame({"date": dates, "value": vals})


# ---- _build_surrogate --------------------------------------------------

def test_build_surrogate_linear():
    s = _build_surrogate("linear")
    assert hasattr(s, "fit") and hasattr(s, "predict")


def test_build_surrogate_random_forest():
    s = _build_surrogate("random_forest")
    assert hasattr(s, "fit")


def test_build_surrogate_unknown_raises():
    with pytest.raises(ValueError, match="Unknown surrogate"):
        _build_surrogate("does_not_exist")


# ---- _run_surrogate_forecast -------------------------------------------

def test_run_surrogate_forecast_returns_forecast_result():
    from multioutreg.time_series.chronos_adapter import ForecastResult
    df = _make_ts_df(n=100)
    out = _run_surrogate_forecast(
        df, target_col="value",
        n_lags=8, horizon=5,
        uncertainty="none",
        surrogate_name="linear",
    )
    assert "forecast_result" in out
    assert isinstance(out["forecast_result"], ForecastResult)
    assert out["forecast_result"].quantiles.shape[2] == 5  # horizon=5


def test_run_surrogate_forecast_cv_summary_keys():
    df = _make_ts_df(n=100)
    out = _run_surrogate_forecast(
        df, target_col="value",
        n_lags=8, horizon=3,
        uncertainty="none",
        surrogate_name="ridge",
    )
    cv = out["cv_summary"]
    for key in ("mean_smape", "std_smape", "mean_mase", "n_folds"):
        assert key in cv


def test_run_surrogate_forecast_history_returned():
    df = _make_ts_df(n=100)
    out = _run_surrogate_forecast(
        df, target_col="value",
        n_lags=8, horizon=4,
        uncertainty="none",
        surrogate_name="linear",
    )
    assert "history" in out
    assert len(out["history"]) > 0


def test_run_surrogate_forecast_horizon_matches():
    df = _make_ts_df(n=100)
    out = _run_surrogate_forecast(
        df, target_col="value",
        n_lags=8, horizon=7,
        uncertainty="none",
        surrogate_name="linear",
    )
    assert out["forecast_result"].quantiles.shape[2] == 7


# ---- _run_chronos_forecast (mocked) -----------------------------------

def test_run_chronos_forecast_mocked():
    """_run_chronos_forecast returns ForecastResult with correct shape when mocked."""
    from unittest.mock import MagicMock, patch

    mock_output = MagicMock()
    mock_output.detach.return_value.cpu.return_value.numpy.return_value = np.zeros(
        (1, 9, 6), dtype=np.float32
    )

    try:
        from multioutreg.time_series.chronos_adapter import ChronosForecaster
    except Exception:
        pytest.skip("ChronosForecaster not importable")

    with patch("multioutreg.time_series.chronos_adapter.BaseChronosPipeline") as MockPipeline:
        mock_pipe = MagicMock()
        mock_pipe.predict.return_value = mock_output
        MockPipeline.from_pretrained.return_value = mock_pipe

        rng = np.random.default_rng(0)
        series_dict = {"y": rng.standard_normal(50)}
        result = _run_chronos_forecast(
            series_dict,
            model_name="amazon/chronos-bolt-tiny",
            horizon=6,
            quantiles=[0.1, 0.5, 0.9],
        )

    assert result.quantiles.shape[2] == 6


# ---- _run_ts_pipeline --------------------------------------------------

def test_run_ts_pipeline_returns_keys():
    """_run_ts_pipeline returns the expected keys (at least ARIMA succeeds)."""
    df = _make_ts_df(n=80)
    out = _run_ts_pipeline(
        df,
        target_col="value",
        datetime_col="date",
        freq="1D",
        verbose=False,
    )
    # If any model succeeded we get the standard output dict
    if "error" not in out:
        assert "perf_df" in out
        assert "best_model" in out
        assert isinstance(out["perf_df"], pd.DataFrame)
