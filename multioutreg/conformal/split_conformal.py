# Copyright (c) 2026 takotime808

"""Split (inductive) conformal prediction."""

from typing import Optional, Tuple, Union

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from multioutreg.conformal.base import BaseConformalPredictor
from multioutreg.conformal.utils import absolute_residual_score, normalized_residual_score


class SplitConformalPredictor(BaseConformalPredictor):
    """Split (inductive) conformal prediction.

    Splits training data into a proper training set and a calibration set.
    The model is fitted on the training set, and nonconformity scores
    (absolute residuals) are computed on the calibration set. At prediction
    time, the conformal quantile of these scores determines the interval width.

    Coverage guarantee: P(Y in [y_lower, y_upper]) >= 1 - alpha
    for exchangeable data.

    Parameters
    ----------
    estimator : sklearn-compatible regressor
        Any regressor with fit/predict. Will be cloned internally.
    calibration_size : float
        Fraction of training data reserved for calibration (default 0.2).
    adaptive : bool
        If True, uses normalized residuals |y - y_hat| / y_std for
        locally-adaptive intervals. Requires the estimator to support
        predict(X, return_std=True).
    random_state : int, optional
        Seed for reproducible calibration splits.
    """

    def __init__(
        self,
        estimator,
        calibration_size: float = 0.2,
        adaptive: bool = False,
        random_state: Optional[int] = None,
    ):
        super().__init__(estimator, random_state)
        self.calibration_size = calibration_size
        self.adaptive = adaptive

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SplitConformalPredictor":
        """Fit model on training split and compute calibration scores.

        Parameters
        ----------
        X : (n_samples, n_features)
        y : (n_samples,) or (n_samples, n_outputs)
        """
        y = self._ensure_2d(y)
        self._n_outputs = y.shape[1]

        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y,
            test_size=self.calibration_size,
            random_state=self.random_state,
        )

        self.model_ = clone(self.estimator)
        if self._n_outputs == 1:
            self.model_.fit(X_train, y_train.ravel())
        else:
            self.model_.fit(X_train, y_train)

        # Compute nonconformity scores on calibration set
        if self.adaptive:
            y_cal_pred, y_cal_std = self.model_.predict(X_cal, return_std=True)
            y_cal_pred = self._ensure_2d(y_cal_pred)
            y_cal_std = self._ensure_2d(y_cal_std)
            self.residuals_ = normalized_residual_score(y_cal, y_cal_pred, y_cal_std)
        else:
            y_cal_pred = self._ensure_2d(self.model_.predict(X_cal))
            self.residuals_ = absolute_residual_score(y_cal, y_cal_pred)

        return self

    def predict_interval(
        self,
        X: np.ndarray,
        alpha: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute prediction intervals.

        Parameters
        ----------
        X : (n_samples, n_features)
        alpha : float
            Miscoverage level. Intervals target 1-alpha coverage.

        Returns
        -------
        y_lower, y_upper : np.ndarray
            Shape (n_samples,) for single output or (n_samples, n_outputs).
        """
        if not hasattr(self, "model_"):
            raise AttributeError("Predictor not fitted. Call fit() first.")

        if self.adaptive:
            y_pred, y_std = self.model_.predict(X, return_std=True)
            y_pred = self._ensure_2d(y_pred)
            y_std = self._ensure_2d(y_std)
        else:
            y_pred = self._ensure_2d(self.model_.predict(X))
            y_std = None

        # Compute conformal quantile per output
        q = np.array([
            self._conformal_quantile(self.residuals_[:, j], alpha)
            for j in range(self._n_outputs)
        ])

        if self.adaptive:
            y_lower = y_pred - q[np.newaxis, :] * y_std
            y_upper = y_pred + q[np.newaxis, :] * y_std
        else:
            y_lower = y_pred - q[np.newaxis, :]
            y_upper = y_pred + q[np.newaxis, :]

        if self._n_outputs == 1:
            return y_lower.ravel(), y_upper.ravel()
        return y_lower, y_upper
