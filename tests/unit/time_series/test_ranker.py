# Copyright (c) 2025 takotime808

import pandas as pd
import pytest

from multioutreg.time_series.ts_dynamic_fit import Ranker


def _make_models_df(records):
    """Build the DataFrame format expected by Ranker."""
    return pd.DataFrame({"performance": records})


def test_ranker_returns_model_name():
    df = _make_models_df([
        {"Model": "ARIMA", "RMSE": 1.0, "MAE": 0.8, "MAPE": 0.05},
        {"Model": "SARIMA", "RMSE": 2.0, "MAE": 1.5, "MAPE": 0.10},
    ])
    ranker = Ranker(df)
    best = ranker.get_best()
    assert best == "ARIMA"


def test_ranker_picks_lowest_error():
    df = _make_models_df([
        {"Model": "M1", "RMSE": 5.0, "MAE": 4.0, "MAPE": 0.4},
        {"Model": "M2", "RMSE": 1.0, "MAE": 1.0, "MAPE": 0.1},
        {"Model": "M3", "RMSE": 3.0, "MAE": 2.5, "MAPE": 0.2},
    ])
    ranker = Ranker(df)
    assert ranker.get_best() == "M2"


def test_ranker_three_models_correct_order():
    df = _make_models_df([
        {"Model": "A", "RMSE": 10.0, "MAE": 8.0, "MAPE": 0.8},
        {"Model": "B", "RMSE": 2.0,  "MAE": 1.5, "MAPE": 0.15},
        {"Model": "C", "RMSE": 0.5,  "MAE": 0.4, "MAPE": 0.04},
    ])
    ranker = Ranker(df)
    assert ranker.get_best() == "C"


def test_ranker_verbose_does_not_raise():
    df = _make_models_df([
        {"Model": "ARIMA", "RMSE": 1.0, "MAE": 0.8, "MAPE": 0.05},
        {"Model": "SARIMA", "RMSE": 2.0, "MAE": 1.5, "MAPE": 0.10},
    ])
    ranker = Ranker(df, verbose=True)
    result = ranker.get_best()
    assert result == "ARIMA"


def test_ranker_tie_returns_a_model():
    """When two models are equal on all metrics, get_best() still returns a valid name."""
    df = _make_models_df([
        {"Model": "A", "RMSE": 1.0, "MAE": 1.0, "MAPE": 0.1},
        {"Model": "B", "RMSE": 1.0, "MAE": 1.0, "MAPE": 0.1},
    ])
    ranker = Ranker(df)
    best = ranker.get_best()
    assert best in ("A", "B")


def test_ranker_metrics_attribute():
    df = _make_models_df([
        {"Model": "X", "RMSE": 1.0, "MAE": 1.0, "MAPE": 0.1},
    ])
    ranker = Ranker(df)
    assert "RMSE" in ranker.metrics
    assert "MAE" in ranker.metrics
    assert "MAPE" in ranker.metrics


def test_ranker_single_model():
    """Single model should always be returned as best."""
    df = _make_models_df([
        {"Model": "OnlyOne", "RMSE": 99.0, "MAE": 99.0, "MAPE": 0.99},
    ])
    ranker = Ranker(df)
    assert ranker.get_best() == "OnlyOne"
