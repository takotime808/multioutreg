# Copyright (c) 2026 takotime808
"""
Normalized optimization and global-accuracy errors (Mainini et al. 2022).

The AVT-331 L1 benchmark paper (arXiv:2204.07867, Sec. 5) scores a method by
how close its estimated optimum is to the true one and by how well its
surrogate matches the high-fidelity function everywhere. All four errors are
dimensionless, so they compare across problems:

- design_error(x_hat, x_star, bounds=None): ``||x_hat - x*|| / sqrt(D)`` on the unit cube
- goal_error(f_at_x_hat, f_min, f_max): ``(f(x_hat) - f_min) / (f_max - f_min)``
- total_error(e_x, e_f): ``sqrt((e_x^2 + e_f^2) / 2)``
- normalized_rmse(y_true, y_pred, f_min, f_max): RMSE over the range ``f_max - f_min``
- optimum_errors(...): ``{"Ex", "Ef", "Et"}`` in one call

Notes
-----
* ``f_at_x_hat`` must be the *true* (high-fidelity) value at the estimated
  optimum, not the surrogate's prediction there.
* ``x_star`` may hold several minimizers (one per row); the nearest one counts.
  For optimal sets that are not finite (a face, a curve), pass ``distance``
  instead: a callable giving the unit-cube distance from each row of its
  input to the set.
* ``f_min`` and ``f_max`` are the function's true extremes over the domain,
  supplied with the benchmark, not the extremes of the training data.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np


def _unit(X, bounds) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if bounds is None:
        return X
    bounds = np.asarray(bounds, dtype=np.float64)
    return (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def _check_range(f_min: float, f_max: float) -> float:
    span = float(f_max) - float(f_min)
    if not span > 0:
        raise ValueError(f"f_max ({f_max}) must be greater than f_min ({f_min}).")
    return span


def design_error(x_hat, x_star=None, bounds=None, distance: Optional[Callable] = None) -> float:
    """Distance from ``x_hat`` to the nearest true minimizer, scaled to the unit cube, over ``sqrt(D)``.

    ``x_hat`` and ``x_star`` are in the problem's own coordinates when
    ``bounds`` (shape ``(D, 2)``) is given, otherwise already in the unit cube.
    ``distance``, if given, replaces ``x_star`` and receives the unit-cube point.
    """
    u_hat = _unit(x_hat, bounds)
    if u_hat.shape[0] != 1:
        raise ValueError("x_hat must be a single point.")
    D = u_hat.shape[1]
    if distance is not None:
        d = float(np.asarray(distance(u_hat)).reshape(-1)[0])
    elif x_star is not None:
        u_star = _unit(x_star, bounds)
        if u_star.shape[1] != D:
            raise ValueError(f"x_star has {u_star.shape[1]} coordinates, x_hat has {D}.")
        d = float(np.min(np.linalg.norm(u_star - u_hat, axis=1)))
    else:
        raise ValueError("Pass x_star or distance.")
    return d / np.sqrt(D)


def goal_error(f_at_x_hat: float, f_min: float, f_max: float) -> float:
    """``(f(x_hat) - f_min) / (f_max - f_min)``; 0 at the optimum, 1 at the worst point."""
    return (float(f_at_x_hat) - float(f_min)) / _check_range(f_min, f_max)


def total_error(e_x: float, e_f: float) -> float:
    """Aggregate of design and goal errors (Serani et al. 2016): ``sqrt((e_x^2 + e_f^2) / 2)``."""
    return float(np.sqrt((e_x ** 2 + e_f ** 2) / 2))


def normalized_rmse(y_true, y_pred, f_min: float, f_max: float) -> float:
    """RMSE between the true function and a surrogate, divided by ``f_max - f_min``."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true has {y_true.size} values, y_pred has {y_pred.size}.")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) / _check_range(f_min, f_max)


def optimum_errors(
    x_hat,
    f_at_x_hat: float,
    f_min: float,
    f_max: float,
    x_star=None,
    bounds=None,
    distance: Optional[Callable] = None,
) -> Dict[str, float]:
    """``{"Ex": design_error, "Ef": goal_error, "Et": total_error}`` for one estimated optimum."""
    e_x = design_error(x_hat, x_star, bounds=bounds, distance=distance)
    e_f = goal_error(f_at_x_hat, f_min, f_max)
    return {"Ex": e_x, "Ef": e_f, "Et": total_error(e_x, e_f)}
