# Copyright (c) 2026 takotime808

"""CV+ (cross-validation+) conformal prediction (Barber et al., 2021)."""

import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold

from multioutreg.conformal.base import BaseConformalPredictor
from multioutreg.conformal.utils import absolute_residual_score


class CVPlusConformalPredictor(BaseConformalPredictor):
    """CV+ conformal prediction (Barber et al., 2021).

    Uses K-fold cross-validation to compute out-of-fold residuals R_i for every
    training point, avoiding the data-splitting inefficiency of split conformal.
    All data is used for both training and calibration.

    For a test point x, with mu_{-k(i)} the model fitted without the fold
    containing training point i, the interval is::

        lower = floor(alpha (n+1))-th smallest of { mu_{-k(i)}(x) - R_i }
        upper = ceil((1-alpha)(n+1))-th smallest of { mu_{-k(i)}(x) + R_i }

    Coverage guarantee: P(Y in interval) >= 1 - 2*alpha for exchangeable data
    (in practice usually close to 1 - alpha). Intervals are unbounded when n is
    too small for the requested alpha.

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
        # fold_ids_[i] = index into fold_models_ of the model that never saw point i
        self.fold_ids_ = np.full(n_samples, -1, dtype=int)

        for k, (train_idx, val_idx) in enumerate(kf.split(X)):
            model_k = clone(self.estimator)
            if self._n_outputs == 1:
                model_k.fit(X[train_idx], y[train_idx].ravel())
            else:
                model_k.fit(X[train_idx], y[train_idx])

            y_val_pred = self._ensure_2d(model_k.predict(X[val_idx]))
            self.residuals_[val_idx] = absolute_residual_score(y[val_idx], y_val_pred)
            self.fold_ids_[val_idx] = k
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

        Parameters
        ----------
        X : (n_samples, n_features)
        alpha : float
            Miscoverage level. Guaranteed coverage is >= 1 - 2*alpha.

        Returns
        -------
        y_lower, y_upper : np.ndarray
        """
        if not hasattr(self, "model_"):
            raise AttributeError("Predictor not fitted. Call fit() first.")
        self._check_alpha(alpha)

        fold_preds = np.stack([
            self._ensure_2d(m.predict(X)) for m in self.fold_models_
        ], axis=0)  # (K, n_test, n_outputs)

        n = self.residuals_.shape[0]
        k_lo = int(np.floor(alpha * (n + 1)))       # 1-indexed rank
        k_hi = int(np.ceil((1 - alpha) * (n + 1)))  # 1-indexed rank
        if k_lo < 1 or k_hi > n:
            warnings.warn(
                f"{n} training points are too few for CV+ at alpha={alpha}; "
                "returning infinite interval bounds.",
                UserWarning,
                stacklevel=2,
            )

        n_test = fold_preds.shape[1]
        y_lower = np.empty((n_test, self._n_outputs))
        y_upper = np.empty((n_test, self._n_outputs))
        for j in range(self._n_outputs):
            # mu[i, t] = prediction at test point t from the model that left out point i
            mu = fold_preds[self.fold_ids_, :, j]          # (n, n_test)
            r = self.residuals_[:, j][:, np.newaxis]       # (n, 1)
            if k_lo >= 1:
                y_lower[:, j] = np.partition(mu - r, k_lo - 1, axis=0)[k_lo - 1]
            else:
                y_lower[:, j] = -np.inf
            if k_hi <= n:
                y_upper[:, j] = np.partition(mu + r, k_hi - 1, axis=0)[k_hi - 1]
            else:
                y_upper[:, j] = np.inf

        if self._n_outputs == 1:
            return y_lower.ravel(), y_upper.ravel()
        return y_lower, y_upper
