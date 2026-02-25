# Copyright (c) 2026 takotime808

"""Quantile Regression surrogate — asymmetric, heteroscedastic prediction intervals."""

from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import QuantileRegressor

from multioutreg.surrogates.conformal_mixin import ConformalMixin


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF via erfinv (no scipy required)."""
    # erfinv(2p - 1) = norm.ppf(p) / sqrt(2)
    return math.sqrt(2.0) * _erfinv(2.0 * p - 1.0)


def _erfinv(x: float) -> float:
    """Rational approximation of erfinv for |x| < 1."""
    # Abramowitz & Stegun approximation, adequate for our use
    sign = 1.0 if x >= 0.0 else -1.0
    x = abs(x)
    a = 0.147
    t = math.sqrt(-math.log((1.0 - x * x) / 2.0 + 1e-300))
    t2 = (2.0 / (math.pi * a) + math.log((1.0 - x * x) / 2.0 + 1e-300) / 2.0)
    result = sign * math.sqrt(math.sqrt(t2 * t2 - math.log((1.0 - x * x) / 2.0 + 1e-300) / a) - t2)
    return result


class QuantileRegressionSurrogate(ConformalMixin):
    """Quantile Regression surrogate with native asymmetric prediction intervals.

    Fits three :class:`sklearn.linear_model.QuantileRegressor` models per output:
    the median (q=0.5) for point predictions, and lower / upper quantiles
    ``(miscoverage/2, 1 - miscoverage/2)`` for heteroscedastic intervals.
    Produces asymmetric intervals that adapt to the local noise level without
    assuming a symmetric error distribution — complementary to (not a replacement
    for) conformal wrapping.

    Use :meth:`predict_intervals` to retrieve raw (lower, upper) quantile bounds
    directly.  :meth:`predict(return_std=True)` returns a pseudo-std approximated
    as ``(upper - lower) / (2 * z_alpha)`` where ``z_alpha`` is the normal quantile,
    enabling drop-in compatibility with the standard surrogate uncertainty API.

    Parameters
    ----------
    alpha : float, default 1.0
        L1 regularisation strength for ``QuantileRegressor`` (larger → more
        shrinkage).  Corresponds to ``QuantileRegressor.alpha``.
    miscoverage : float, default 0.1
        Target miscoverage rate.  ``miscoverage=0.1`` fits q=0.05 / q=0.5 / q=0.95
        and targets 90 % prediction intervals.
    solver : str, default "highs"
        Linear programming solver (``"highs"`` is the recommended default).
    solver_options : dict | None, default None
        Additional options forwarded to the LP solver (e.g. ``{"max_iter": 1000}``).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        miscoverage: float = 0.1,
        solver: str = "highs",
        solver_options: "dict | None" = None,
    ):
        self.alpha = alpha
        self.miscoverage = miscoverage
        self.solver = solver
        self.solver_options = solver_options

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "QuantileRegressionSurrogate":
        """Fit median, lower, and upper quantile models per output.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        Y : np.ndarray, shape (n_samples,) or (n_samples, n_outputs)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        self.n_outputs_ = Y.shape[1]
        q_lo = self.miscoverage / 2.0
        q_hi = 1.0 - self.miscoverage / 2.0

        self._models_median: list[QuantileRegressor] = []
        self._models_lo: list[QuantileRegressor] = []
        self._models_hi: list[QuantileRegressor] = []

        def _make_qr(q: float) -> QuantileRegressor:
            return QuantileRegressor(
                quantile=q,
                alpha=self.alpha,
                solver=self.solver,
                solver_options=self.solver_options,
            )

        for j in range(self.n_outputs_):
            y_j = Y[:, j]
            m = _make_qr(0.5)
            m.fit(X, y_j)
            self._models_median.append(m)

            lo = _make_qr(q_lo)
            lo.fit(X, y_j)
            self._models_lo.append(lo)

            hi = _make_qr(q_hi)
            hi.fit(X, y_j)
            self._models_hi.append(hi)

        return self

    def predict(
        self, X: np.ndarray, return_std: bool = False
    ) -> "np.ndarray | tuple[np.ndarray, np.ndarray]":
        """Predict median outputs, optionally returning pseudo-std.

        Parameters
        ----------
        X : np.ndarray
        return_std : bool, default False
            If True, returns a pseudo-std approximated from the interval width
            as ``(q_hi - q_lo) / (2 * z_alpha)``.  This allows the surrogate
            to participate in the standard ``predict(return_std=True)`` contract.

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples, n_outputs)
        y_std  : np.ndarray, shape (n_samples, n_outputs)  [only if return_std]
        """
        if not hasattr(self, "_models_median"):
            raise AttributeError(
                "QuantileRegressionSurrogate is not fitted. Call fit() first."
            )

        y_pred = np.column_stack([m.predict(X) for m in self._models_median])

        if not return_std:
            return y_pred

        y_lo = np.column_stack([m.predict(X) for m in self._models_lo])
        y_hi = np.column_stack([m.predict(X) for m in self._models_hi])

        z = _norm_ppf(1.0 - self.miscoverage / 2.0)
        pseudo_std = np.maximum((y_hi - y_lo) / (2.0 * z), 0.0)
        return y_pred, pseudo_std

    def predict_intervals(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return raw quantile interval bounds (y_lower, y_upper).

        Unlike ``predict(return_std=True)`` which converts width to a pseudo-std,
        this method returns the actual fitted quantile predictions.

        Returns
        -------
        y_lower : np.ndarray, shape (n_samples, n_outputs)
        y_upper : np.ndarray, shape (n_samples, n_outputs)
        """
        if not hasattr(self, "_models_lo"):
            raise AttributeError(
                "QuantileRegressionSurrogate is not fitted. Call fit() first."
            )
        y_lo = np.column_stack([m.predict(X) for m in self._models_lo])
        y_hi = np.column_stack([m.predict(X) for m in self._models_hi])
        return y_lo, y_hi

    def _conformal_point_predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        preds = np.asarray(preds)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return preds

    def get_params(self, deep: bool = True) -> dict:
        return {
            "alpha": self.alpha,
            "miscoverage": self.miscoverage,
            "solver": self.solver,
            "solver_options": self.solver_options,
        }

    def set_params(self, **params) -> "QuantileRegressionSurrogate":
        for key, value in params.items():
            setattr(self, key, value)
        return self
