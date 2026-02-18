# Copyright (c) 2026 takotime808

"""Shared utilities for conformal prediction."""

from typing import Optional, Tuple

import numpy as np


def absolute_residual_score(
    y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    """Nonconformity score: absolute residual |y - y_hat|."""
    return np.abs(y_true - y_pred)


def normalized_residual_score(
    y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray
) -> np.ndarray:
    """Normalized nonconformity score: |y - y_hat| / sigma.

    Produces adaptive (heteroscedastic) intervals when used with models
    that provide uncertainty estimates via return_std=True.
    """
    y_std = np.clip(y_std, 1e-12, None)
    return np.abs(y_true - y_pred) / y_std


def intervals_from_scores(
    y_pred: np.ndarray,
    conformal_quantile: float,
    y_std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct intervals from point predictions and a conformal quantile.

    If y_std is provided (normalized scoring), intervals are adaptive:
        [y_pred - q * y_std, y_pred + q * y_std]
    Otherwise, constant-width intervals:
        [y_pred - q, y_pred + q]
    """
    if y_std is not None:
        half_width = conformal_quantile * y_std
    else:
        half_width = conformal_quantile
    return y_pred - half_width, y_pred + half_width
