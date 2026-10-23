# Copyright (c) 2025 takotime808

from __future__ import annotations

import numpy as np

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps))

def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
) -> float:
    """
    Mean absolute scaled error (Hyndman & Koehler, 2006).

    The forecast MAE on ``y_true`` is divided by the in-sample MAE of the
    seasonal-naive forecast on ``y_train`` (``y_train[t] - y_train[t - m]``).
    Scaling by the training series keeps the denominator defined for a
    1-step evaluation window and makes the score comparable across windows.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_train = np.asarray(y_train, dtype=float).ravel()
    m = int(seasonality)
    if m < 1:
        raise ValueError(f"seasonality must be >= 1, got {seasonality}")
    if y_train.size <= m:
        raise ValueError(
            f"y_train needs more than seasonality={m} points to compute the "
            f"seasonal-naive scale, got {y_train.size}"
        )
    denom = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return float(np.mean(np.abs(y_true - y_pred)) / (denom + 1e-12))

def weighted_quantile_loss(y_true: np.ndarray, q_forecasts: np.ndarray, q_levels) -> float:
    """
    WQL as defined in the Chronos paper (Ansari et al., 2024; lower is better).

    For each quantile level q, ``WQL_q = 2 * sum_t pinball_q(y_t, yhat_t) / sum_t |y_t|``;
    the result is the mean over the Q levels.
    y_true: [H], q_forecasts: [Q, H], q_levels: [Q]
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    q_forecasts = np.asarray(q_forecasts, dtype=float)
    q = np.asarray(q_levels, dtype=float)
    assert q_forecasts.shape[0] == q.shape[0]
    diff = y_true[None, :] - q_forecasts
    pinball = np.maximum(q[:, None] * diff, (q[:, None] - 1) * diff)
    scale = np.sum(np.abs(y_true)) + 1e-12
    wql = np.mean(2.0 * np.sum(pinball, axis=1) / scale)
    return float(wql)
