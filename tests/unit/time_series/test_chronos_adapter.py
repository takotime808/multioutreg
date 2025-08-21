# Copyright (c) 2025 takotime808

import numpy as np
import pytest

from multioutreg.time_series.metrics import (
    smape,
    mase,
    weighted_quantile_loss,
)

try:
    from multioutreg.time_series.chronos_adapter import ChronosForecaster
    _CHRONOS = True
except Exception:
    _CHRONOS = False


def test_metrics_basic():
    y = np.array([1,2,3,4,5], dtype=float)
    yhat = np.array([1,2,2,5,5], dtype=float)
    assert 0 <= smape(y, yhat) <= 200
    assert mase(y, yhat, seasonality=1) >= 0.0
    qs = np.vstack([yhat-0.2, yhat, yhat+0.2])
    w = weighted_quantile_loss(y, qs, [0.1, 0.5, 0.9])
    assert w >= 0.0

@pytest.mark.skipif(not _CHRONOS, reason="chronos-forecasting not installed")
def test_chronos_shapes():
    # short synthetic series
    y = np.sin(np.linspace(0, 4*np.pi, 48)).astype(np.float32)
    f = ChronosForecaster("amazon/chronos-bolt-tiny").fit(y)
    res = f.predict(prediction_length=8, quantiles=(0.1, 0.5, 0.9))
    assert res.quantiles.shape == (1, 3, 8)
    assert list(res.q_levels) == [0.1, 0.5, 0.9]
