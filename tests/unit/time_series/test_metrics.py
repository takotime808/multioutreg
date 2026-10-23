# Copyright (c) 2025 takotime808

import numpy as np
import pytest

from multioutreg.time_series.metrics import mase, weighted_quantile_loss


def test_mase_scales_by_training_naive_error():
    y_train = np.array([0.0, 1.0, 3.0, 6.0])  # naive abs diffs 1, 2, 3 -> MAE 2
    y_true = np.array([10.0, 12.0])
    y_pred = np.array([11.0, 9.0])  # abs errors 1, 3 -> MAE 2
    assert mase(y_true, y_pred, y_train) == pytest.approx(1.0)


def test_mase_one_step_window_is_finite():
    y_train = np.arange(10, dtype=float)
    value = mase(np.array([10.0]), np.array([10.5]), y_train)
    assert np.isfinite(value)
    assert value == pytest.approx(0.5)


def test_mase_seasonal_naive():
    y_train = np.array([1.0, 5.0, 2.0, 7.0, 4.0, 9.0])  # m=2 diffs: 1, 2, 2, 2 -> 1.75
    value = mase(np.array([3.0]), np.array([4.75]), y_train, seasonality=2)
    assert value == pytest.approx(1.0)


def test_mase_is_scale_invariant():
    rng = np.random.default_rng(0)
    y_train = np.cumsum(rng.normal(size=50))
    y_true = y_train[-1] + np.cumsum(rng.normal(size=5))
    y_pred = np.full(5, y_train[-1])
    base = mase(y_true, y_pred, y_train)
    assert mase(1000 * y_true, 1000 * y_pred, 1000 * y_train) == pytest.approx(base)


def test_mase_rejects_short_training_series():
    with pytest.raises(ValueError):
        mase(np.array([1.0]), np.array([1.0]), np.array([1.0]), seasonality=1)
    with pytest.raises(ValueError):
        mase(np.array([1.0]), np.array([1.0]), np.arange(5.0), seasonality=0)


def test_wql_matches_chronos_definition():
    y = np.array([2.0, 4.0])
    q_levels = [0.1, 0.9]
    qf = np.array([[1.0, 5.0], [3.0, 3.0]])
    # q=0.1: pinball(2,1)=0.1, pinball(4,5)=0.9 -> 2*1.0/6
    # q=0.9: pinball(2,3)=0.1, pinball(4,3)=0.9 -> 2*1.0/6
    expected = 2.0 * 1.0 / 6.0
    assert weighted_quantile_loss(y, qf, q_levels) == pytest.approx(expected)


def test_wql_is_scale_invariant_and_zero_for_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0])
    qf = np.vstack([y - 0.5, y, y + 0.5])
    q_levels = [0.1, 0.5, 0.9]
    base = weighted_quantile_loss(y, qf, q_levels)
    assert weighted_quantile_loss(10 * y, 10 * qf, q_levels) == pytest.approx(base)
    assert weighted_quantile_loss(y, np.vstack([y, y, y]), q_levels) == pytest.approx(0.0)
