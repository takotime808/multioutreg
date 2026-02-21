# Copyright (c) 2026 takotime808

"""Random Fourier Feature Gaussian Process surrogate.

Approximates an RBF-kernel GP using Bochner's theorem (Rahimi & Recht, 2007).
Training cost is O(D·n) instead of the O(n³) required by an exact GP, where D
is the number of random Fourier features.  Uncertainty estimates are derived
from the Bayesian Ridge posterior in the lifted feature space -- analytic,
no sampling required.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import BayesianRidge

from multioutreg.surrogates.conformal_mixin import ConformalMixin


class _RFFEstimator(BaseEstimator, RegressorMixin):
    """Single-output RFF-GP estimator (sklearn-compatible, used by GridSearchCV)."""

    def __init__(self, n_components=500, length_scale=1.0, random_state=None):
        self.n_components = n_components
        self.length_scale = length_scale
        self.random_state = random_state

    def _rff_transform(self, X):
        D = self.n_components
        return np.sqrt(2.0 / D) * np.cos(X @ self.omega_ + self.bias_)

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n_features = X.shape[1]
        D = self.n_components
        # Sample frequencies from the spectral density of the RBF kernel: N(0, 1/l²)
        self.omega_ = rng.normal(0.0, 1.0 / self.length_scale, size=(n_features, D))
        self.bias_ = rng.uniform(0.0, 2.0 * np.pi, size=(D,))
        Z = self._rff_transform(X)
        self.regressor_ = BayesianRidge().fit(Z, y)
        return self

    def predict(self, X, return_std=False):
        Z = self._rff_transform(X)
        return self.regressor_.predict(Z, return_std=return_std)


class RFFGPSurrogate(ConformalMixin):
    """Random Fourier Feature Gaussian Process surrogate.

    Fits one :class:`_RFFEstimator` per output column.  Compared to the exact
    GP, this model:

    * Trains in O(D·n) instead of O(n³)
    * Predicts in O(D) instead of O(n)
    * Uses zero external dependencies beyond numpy and sklearn
    * Preserves the kernel structure of an RBF GP

    Parameters
    ----------
    n_components : int
        Number of random Fourier features D. Higher values give a better
        kernel approximation; 500 is usually sufficient.
    length_scale : float
        Length scale of the RBF kernel being approximated.
    random_state : int or None
        Seed for the random frequency draw, for reproducibility.
    """

    def __init__(self, n_components=500, length_scale=1.0, random_state=None):
        self.n_components = n_components
        self.length_scale = length_scale
        self.random_state = random_state

    def fit(self, X, Y):
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.n_outputs_ = Y.shape[1]
        self.estimators_ = [
            _RFFEstimator(
                n_components=self.n_components,
                length_scale=self.length_scale,
                random_state=(
                    None if self.random_state is None else self.random_state + j
                ),
            ).fit(X, Y[:, j])
            for j in range(self.n_outputs_)
        ]
        return self

    def predict(self, X, return_std=False):
        preds, stds = [], []
        for est in self.estimators_:
            if return_std:
                p, s = est.predict(X, return_std=True)
                preds.append(p)
                stds.append(s)
            else:
                preds.append(est.predict(X))
        preds = np.column_stack(preds)
        if return_std:
            return preds, np.column_stack(stds)
        return preds

    def _conformal_point_predict(self, X):
        preds = np.asarray(self.predict(X))
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return preds
