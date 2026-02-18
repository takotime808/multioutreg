# Copyright (c) 2025 takotime808

"""CV+ (cross-validation+) conformal prediction (Barber et al., 2021)."""

from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold

from multioutreg.conformal.base import BaseConformalPredictor
from multioutreg.conformal.utils import absolute_residual_score


class CVPlusConformalPredictor(BaseConformalPredictor):
    """CV+ conformal prediction (Barber et al., 2021).

    Uses K-fold cross-validation to compute out-of-fold residuals for every
    training point, avoiding the data-splitting inefficiency of split conformal.
    All data is used for both training and calibration.

    Parameters
    ----------
    estimator : sklearn-compatible regressor
        Any regressor with fit/predict. Will be cloned for each fold.
    n_folds : int
        Number of cross-validation folds (default 5).
    random_state : int, optional
        Seed for reproducible fold splits.
    """

    def __init__(
        self,
        estimator,
        n_folds: int = 5,
        random_state: Optional[int] = None,
    ):
        super().__init__(estimator, random_state)
        self.n_folds = n_folds

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CVPlusConformalPredictor":
        """Fit fold models and compute out-of-fold calibration scores.

        Parameters
        ----------
        X : (n_samples, n_features)
        y : (n_samples,) or (n_samples, n_outputs)
        """
        y = self._ensure_2d(y)
        self._n_outputs = y.shape[1]
        n_samples = X.shape[0]

        kf = KFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        self.residuals_ = np.full((n_samples, self._n_outputs), np.nan)
        self.fold_models_: List = []

        for train_idx, val_idx in kf.split(X):
            model_k = clone(self.estimator)
            if self._n_outputs == 1:
                model_k.fit(X[train_idx], y[train_idx].ravel())
            else:
                model_k.fit(X[train_idx], y[train_idx])

            y_val_pred = self._ensure_2d(model_k.predict(X[val_idx]))
            self.residuals_[val_idx] = absolute_residual_score(y[val_idx], y_val_pred)
            self.fold_models_.append(model_k)

        # Final model on all data for point predictions
        self.model_ = clone(self.estimator)
        if self._n_outputs == 1:
            self.model_.fit(X, y.ravel())
        else:
            self.model_.fit(X, y)

        return self

    def predict_interval(
        self,
        X: np.ndarray,
        alpha: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """CV+ prediction intervals.

        Uses the mean of fold model predictions combined with the conformal
        quantile of out-of-fold residuals (jackknife+ style).

        Parameters
        ----------
        X : (n_samples, n_features)
        alpha : float
            Miscoverage level. Intervals target 1-alpha coverage.

        Returns
        -------
        y_lower, y_upper : np.ndarray
        """
        if not hasattr(self, "model_"):
            raise AttributeError("Predictor not fitted. Call fit() first.")

        # Aggregate predictions from all fold models
        fold_preds = np.stack([
            self._ensure_2d(m.predict(X)) for m in self.fold_models_
        ], axis=0)  # (K, n_test, n_outputs)

        y_pred = np.mean(fold_preds, axis=0)  # (n_test, n_outputs)

        # Conformal quantile per output
        q = np.array([
            self._conformal_quantile(self.residuals_[:, j], alpha)
            for j in range(self._n_outputs)
        ])

        y_lower = y_pred - q[np.newaxis, :]
        y_upper = y_pred + q[np.newaxis, :]

        if self._n_outputs == 1:
            return y_lower.ravel(), y_upper.ravel()
        return y_lower, y_upper
