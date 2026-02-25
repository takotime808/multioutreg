# Copyright (c) 2025 takotime808

from __future__ import annotations

import numpy as np
from typing import Sequence, Tuple


def gaussian_quantiles(
    mean: np.ndarray,
    std: np.ndarray,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> np.ndarray:
    """Convert (mean, std) arrays to a quantile array via Gaussian approximation.

    Parameters
    ----------
    mean : np.ndarray, shape (horizon,)
    std  : np.ndarray, shape (horizon,)
    quantiles : sequence of Q quantile levels in (0, 1)

    Returns
    -------
    np.ndarray, shape (Q, horizon)
        Estimated quantile values assuming Gaussian predictive distribution.
    """
    from scipy.special import erfinv

    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    q = np.asarray(quantiles, dtype=float)

    # Gaussian quantile: mu + sigma * sqrt(2) * erfinv(2p - 1)
    z = np.sqrt(2.0) * erfinv(2.0 * q - 1.0)          # shape (Q,)
    return mean[np.newaxis, :] + z[:, np.newaxis] * std[np.newaxis, :]


def conformal_interval_from_residuals(
    point_pred: np.ndarray,
    cal_residuals: np.ndarray,
    alpha: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute split-conformal prediction intervals from calibration residuals.

    Uses the standard split-conformal quantile:
        q = ceil((n + 1) * (1 - alpha)) / n -th quantile of |residuals|

    Parameters
    ----------
    point_pred : np.ndarray, shape (horizon,)
        Point predictions for each future step.
    cal_residuals : np.ndarray, shape (n_cal,)
        Absolute residuals on the held-out calibration set.
    alpha : float, default 0.1
        Desired miscoverage level (e.g. 0.1 → 90% coverage).

    Returns
    -------
    lower, upper : np.ndarray, each shape (horizon,)
    """
    abs_resid = np.abs(np.asarray(cal_residuals, dtype=float))
    n = len(abs_resid)
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    q_hat = float(np.quantile(abs_resid, level))
    point_pred = np.asarray(point_pred, dtype=float)
    return point_pred - q_hat, point_pred + q_hat


def propagate_uncertainty_recursive(
    single_step_std: float | np.ndarray,
    horizon: int,
    correlation: float = 0.0,
) -> np.ndarray:
    """Propagate a single-step standard deviation over multiple horizons.

    Assumes Gaussian errors with optional first-order autocorrelation.

    For white-noise errors (correlation=0):
        sigma_h = sigma_1 * sqrt(h)

    For AR(1) errors with coefficient `rho`:
        sigma_h^2 = sigma_1^2 * sum_{k=0}^{h-1} rho^{2k}
                  = sigma_1^2 * (1 - rho^{2h}) / (1 - rho^2)   [rho != 1]

    Parameters
    ----------
    single_step_std : float or np.ndarray
        Standard deviation at horizon h=1.
    horizon : int
        Number of steps to propagate.
    correlation : float, default 0.0
        First-order autocorrelation coefficient of the errors (rho in AR(1)).

    Returns
    -------
    np.ndarray, shape (horizon,)
        Standard deviations at horizons 1, 2, ..., horizon.
    """
    sigma1 = float(np.asarray(single_step_std).ravel()[0])
    rho = float(correlation)
    stds = np.zeros(horizon)

    for h in range(1, horizon + 1):
        if abs(rho) < 1e-9:
            var_h = sigma1 ** 2 * h
        elif abs(rho - 1.0) < 1e-9:
            # Random walk limit
            var_h = sigma1 ** 2 * h
        else:
            var_h = sigma1 ** 2 * (1.0 - rho ** (2 * h)) / (1.0 - rho ** 2)
        stds[h - 1] = np.sqrt(max(0.0, var_h))

    return stds
