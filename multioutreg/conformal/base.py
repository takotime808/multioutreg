# Copyright (c) 2026 takotime808

"""Abstract base class for conformal prediction wrappers."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator


class BaseConformalPredictor(ABC):
    """Abstract base for conformal prediction wrappers.

    All conformal predictors follow a two-phase protocol:
    1. fit(X, y) -- fits the underlying model AND calibrates nonconformity scores.
    2. predict_interval(X, alpha) -- returns (y_lower, y_upper) with
       guaranteed marginal coverage >= 1-alpha.

    For multi-output regression, conformal calibration is applied independently
    per output (marginal coverage guarantee per target).
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        random_state: Optional[int] = None,
    ):
        self.estimator = estimator
        self.random_state = random_state

    @abstractmethod
    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> "BaseConformalPredictor":
        """Fit the model and compute calibration scores."""
        ...

    @abstractmethod
    def predict_interval(
        self,
        X: np.ndarray,
        alpha: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute prediction intervals with coverage >= 1-alpha.

        Returns
        -------
        y_lower, y_upper : np.ndarray
            Shape (n_samples,) for single output or (n_samples, n_outputs).
        """
        ...

    def predict(
        self,
        X: np.ndarray,
        alpha: Optional[float] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Point predictions. If alpha is given, returns (y_pred, y_lower, y_upper)."""
        if not hasattr(self, "model_"):
            raise AttributeError("Predictor not fitted. Call fit() first.")
        y_pred = self.model_.predict(X)
        if alpha is not None:
            y_lower, y_upper = self.predict_interval(X, alpha)
            return y_pred, y_lower, y_upper
        return y_pred

    @staticmethod
    def _ensure_2d(y: np.ndarray) -> np.ndarray:
        """Ensure y is 2D (n_samples, n_outputs)."""
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return y

    @staticmethod
    def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
        """Compute the conformal quantile with finite-sample correction.

        Uses q = ceil((n+1)(1-alpha)) / n, which guarantees
        P(Y in interval) >= 1-alpha for exchangeable data.
        """
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        q_level = min(q_level, 1.0)
        return float(np.quantile(scores, q_level))
