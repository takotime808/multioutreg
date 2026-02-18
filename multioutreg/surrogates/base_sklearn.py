# Copyright (c) 2025 takotime808

import numpy as np
from sklearn.multioutput import MultiOutputRegressor

class BaseSurrogate:
    """Base class for multi-output surrogates."""
    def __init__(self, base_estimator):
        self.model = MultiOutputRegressor(base_estimator)

    def fit(self, X, Y):
        self.model.fit(X, Y)
        return self

    def predict(self, X, return_std=False):
        if return_std:
            try:
                return self.model.predict(X, return_std=True)
            except TypeError:
                preds = self.model.predict(X)
                std = np.zeros_like(preds)
                return preds, std
        return self.model.predict(X)

    def wrap_conformal(self, X_cal, y_cal, **kwargs):
        """Calibrate conformal prediction intervals on held-out data.

        Computes absolute residuals on the calibration set using the
        already-fitted model for distribution-free prediction intervals.

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

        y_cal_pred = self.model.predict(X_cal)
        y_cal_pred = np.asarray(y_cal_pred)
        if y_cal_pred.ndim == 1:
            y_cal_pred = y_cal_pred.reshape(-1, 1)

        self._conformal_residuals = np.abs(y_cal - y_cal_pred)
        self._conformal_n_outputs = y_cal.shape[1]
        return self

    def conformal_predict(self, X, alpha=0.1):
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
        if not hasattr(self, '_conformal_residuals'):
            raise AttributeError(
                "No conformal predictor attached. Call wrap_conformal() first."
            )
        from multioutreg.conformal.base import BaseConformalPredictor

        y_pred = self.model.predict(X)
        y_pred = np.asarray(y_pred)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        q = np.array([
            BaseConformalPredictor._conformal_quantile(
                self._conformal_residuals[:, j], alpha
            )
            for j in range(self._conformal_n_outputs)
        ])

        y_lower = y_pred - q[np.newaxis, :]
        y_upper = y_pred + q[np.newaxis, :]
        return y_lower, y_upper