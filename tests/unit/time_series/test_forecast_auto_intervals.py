# Copyright (c) 2025 takotime808

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from multioutreg.time_series.ts_dynamic_fit import ForecastAutoIntervals


# ---------------------------------------------------------------------------
# Branch 1: statsmodels-style model (has get_forecast)
# ---------------------------------------------------------------------------

def test_forecast_statsmodels_style_returns_dataframe():
    """model.get_forecast() path produces a DataFrame with mean/lower/upper."""
    mock_model = MagicMock()
    mock_pred = MagicMock()
    mock_pred.predicted_mean = pd.Series([1.0, 2.0, 3.0])
    mock_pred.conf_int.return_value = pd.DataFrame({
        "lower y": [0.5, 1.5, 2.5],
        "upper y": [1.5, 2.5, 3.5],
    })
    mock_model.get_forecast.return_value = mock_pred

    fi = ForecastAutoIntervals(mock_model)
    result = fi.forecast(steps=3)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["mean", "lower", "upper"]
    assert len(result) == 3


def test_forecast_statsmodels_passes_steps():
    """get_forecast is called with the requested number of steps."""
    mock_model = MagicMock()
    mock_pred = MagicMock()
    mock_pred.predicted_mean = pd.Series([1.0, 2.0])
    mock_pred.conf_int.return_value = pd.DataFrame({
        "lower y": [0.5, 1.5],
        "upper y": [1.5, 2.5],
    })
    mock_model.get_forecast.return_value = mock_pred

    fi = ForecastAutoIntervals(mock_model)
    fi.forecast(steps=2)
    mock_model.get_forecast.assert_called_once_with(steps=2)


def test_forecast_statsmodels_passes_alpha():
    """conf_int is called with the requested alpha."""
    mock_model = MagicMock()
    mock_pred = MagicMock()
    mock_pred.predicted_mean = pd.Series([1.0])
    mock_pred.conf_int.return_value = pd.DataFrame({
        "lower y": [0.5],
        "upper y": [1.5],
    })
    mock_model.get_forecast.return_value = mock_pred

    fi = ForecastAutoIntervals(mock_model)
    fi.forecast(steps=1, alpha=0.10)
    mock_pred.conf_int.assert_called_once_with(alpha=0.10)


# ---------------------------------------------------------------------------
# Branch 2: Quantile regressor (has predict + set_params + quantile)
# ---------------------------------------------------------------------------

def test_forecast_quantile_regressor_returns_dataframe():
    """Quantile regressor branch returns mean/lower/upper columns."""
    # Use spec to block 'get_forecast' so the quantile branch is reached
    mock_model = MagicMock(spec=["predict", "set_params", "quantile"])
    mock_model.quantile = 0.5
    mock_model.predict.return_value = np.array([1.0, 2.0, 3.0])

    fi = ForecastAutoIntervals(mock_model)
    X = np.ones((3, 2))
    result = fi.forecast(steps=3, X=X)

    assert isinstance(result, pd.DataFrame)
    assert "mean" in result.columns
    assert "lower" in result.columns
    assert "upper" in result.columns


def test_forecast_quantile_regressor_restores_original_quantile():
    """After forecasting, the original quantile is restored via set_params."""
    original_q = 0.5
    mock_model = MagicMock(spec=["predict", "set_params", "quantile"])
    mock_model.quantile = original_q
    mock_model.predict.return_value = np.array([1.0, 2.0])

    fi = ForecastAutoIntervals(mock_model)
    fi.forecast(X=np.ones((2, 1)))

    # Last set_params call should restore original_q
    last_call = mock_model.set_params.call_args
    assert last_call[1].get("quantile") == original_q or last_call[0] == ({"quantile": original_q},)


# ---------------------------------------------------------------------------
# Branch 3: Generic bootstrap (has predict, y_train and X provided)
# ---------------------------------------------------------------------------

def test_forecast_generic_bootstrap_returns_dataframe():
    """Bootstrap fallback returns a DataFrame with mean/lower/upper.

    The implementation slices X[:len(y_train)] to compute residuals, so X must
    have at least len(y_train) rows.
    """
    n = 20
    rng = np.random.default_rng(0)
    y_train = rng.standard_normal(n)
    # X covers the full series so X[:n] is valid
    X = rng.standard_normal((n, 2))

    mock_model = MagicMock(spec=["predict"])
    mock_model.predict.side_effect = lambda arr: np.ones(len(arr))

    fi = ForecastAutoIntervals(mock_model)
    result = fi.forecast(X=X, y_train=y_train)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"mean", "lower", "upper"}
    assert len(result) == n


def test_forecast_generic_bootstrap_lower_le_upper():
    """Bootstrap lower bound <= upper bound for every step."""
    n = 25
    rng = np.random.default_rng(1)
    y_train = rng.standard_normal(n)
    X = rng.standard_normal((n, 3))

    mock_model = MagicMock(spec=["predict"])
    mock_model.predict.side_effect = lambda arr: rng.standard_normal(len(arr))

    fi = ForecastAutoIntervals(mock_model)
    result = fi.forecast(X=X, y_train=y_train)

    assert (result["lower"].values <= result["upper"].values).all()


# ---------------------------------------------------------------------------
# Branch 4: Unsupported model → NotImplementedError
# ---------------------------------------------------------------------------

def test_forecast_unsupported_raises_not_implemented():
    """No X/y_train and no get_forecast/quantile → NotImplementedError."""
    mock_model = MagicMock(spec=["predict"])
    fi = ForecastAutoIntervals(mock_model)
    with pytest.raises(NotImplementedError):
        fi.forecast(steps=3)


# ---------------------------------------------------------------------------
# Model attribute stored correctly
# ---------------------------------------------------------------------------

def test_model_attribute_stored():
    dummy = object()
    fi = ForecastAutoIntervals(dummy)
    assert fi.model is dummy
