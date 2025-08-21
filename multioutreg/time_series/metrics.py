# Copyright (c) 2025 takotime808

from __future__ import annotations

import numpy as np

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps))

def mase(y_true: np.ndarray, y_pred: np.ndarray, seasonality: int = 1) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.mean(np.abs(y_true[seasonality:] - y_true[:-seasonality]))
    return float(np.mean(np.abs(y_true - y_pred)) / (denom + 1e-12))

def weighted_quantile_loss(y_true: np.ndarray, q_forecasts: np.ndarray, q_levels) -> float:
    """
    WQL as used in the Chronos evals (lower is better).
    y_true: [H], q_forecasts: [Q, H], q_levels: [Q]
    """
    y_true = np.asarray(y_true).ravel()
    q_forecasts = np.asarray(q_forecasts)
    q = np.asarray(q_levels)
    assert q_forecasts.shape[0] == q.shape[0]
    diff = y_true[None, :] - q_forecasts
    wql = np.mean(np.mean(np.maximum(q[:, None] * diff, (q[:, None] - 1) * diff), axis=1))
    return float(wql)
