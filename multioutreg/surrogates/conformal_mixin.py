# Copyright (c) 2026 takotime808

"""Shared conformal prediction mixin for surrogate models."""

import numpy as np


class ConformalMixin:
    """Mixin providing conformal prediction calibration and inference.

    Any surrogate that implements ``fit(X, Y)`` and ``predict(X)`` can
    inherit this mixin to gain ``wrap_conformal`` and ``conformal_predict``
    without depending on sklearn's ``MultiOutputRegressor``.

    Subclasses must implement ``_conformal_point_predict(X)`` that returns
    an ``(n_samples, n_outputs)`` array. By default this calls
    ``self.predict(X)`` with no keyword arguments.
    """

    def _conformal_point_predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        preds = np.asarray(preds)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return preds

    def wrap_conformal(self, X_cal: np.ndarray, y_cal: np.ndarray, **kwargs) -> "ConformalMixin":
        """Calibrate conformal prediction intervals on held-out data.

        Parameters
        ----------
        X_cal : np.ndarray
            Calibration features (must NOT overlap with training data).
        y_cal : np.ndarray
            Calibration targets.

        Returns
        -------
        self
        """
        y_cal = np.asarray(y_cal)
        if y_cal.ndim == 1:
            y_cal = y_cal.reshape(-1, 1)

        y_cal_pred = self._conformal_point_predict(X_cal)

        self._conformal_residuals = np.abs(y_cal - y_cal_pred)
        self._conformal_n_outputs = y_cal.shape[1]
        return self

    def conformal_predict(self, X: np.ndarray, alpha: float = 0.1):
        """Return conformal prediction intervals.

        Parameters
        ----------
        X : np.ndarray
        alpha : float
            Miscoverage level. Intervals target 1-alpha coverage.

        Returns
        -------
        y_lower, y_upper : np.ndarray
        """
        if not hasattr(self, "_conformal_residuals"):
            raise AttributeError(
                "No conformal predictor attached. Call wrap_conformal() first."
            )
        from multioutreg.conformal.base import BaseConformalPredictor

        y_pred = self._conformal_point_predict(X)

        q = np.array([
            BaseConformalPredictor._conformal_quantile(
                self._conformal_residuals[:, j], alpha
            )
            for j in range(self._conformal_n_outputs)
        ])

        y_lower = y_pred - q[np.newaxis, :]
        y_upper = y_pred + q[np.newaxis, :]
        return y_lower, y_upper
