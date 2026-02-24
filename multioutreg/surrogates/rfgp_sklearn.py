# Copyright (c) 2026 takotime808

"""Random Fourier Feature Gaussian Process surrogate.

Approximates a stationary kernel GP using Bochner's theorem (Rahimi & Recht,
2007).  Training cost is O(D·n) instead of the O(n³) required by an exact GP,
where D is the number of random Fourier features.  Uncertainty estimates are
derived from the Bayesian Ridge posterior in the lifted feature space —
analytic, no sampling required.

Supported kernels
-----------------
* ``'rbf'``      — Radial Basis Function / squared-exponential.
                   Spectral density: N(0, 1/l²).
* ``'matern32'`` — Matérn ν=3/2.  Spectral density: multivariate Student-t
                   with 2ν=3 degrees of freedom.
                   Sampled as z·√(ν/u)/l, z~N(0,I), u~Gamma(ν,1).
* ``'matern52'`` — Matérn ν=5/2.  Spectral density: multivariate Student-t
                   with 2ν=5 degrees of freedom.
                   Sampled as z·√(ν/u)/l, z~N(0,I), u~Gamma(ν,1).

The Matérn kernels are less smooth than RBF and are often a better prior for
physical engineering surrogates that have finite differentiability.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import BayesianRidge

from multioutreg.surrogates.conformal_mixin import ConformalMixin

_VALID_KERNELS = ("rbf", "matern32", "matern52")


class _RFFEstimator(BaseEstimator, RegressorMixin):
    """Single-output RFF-GP estimator (sklearn-compatible, used by GridSearchCV)."""

    def __init__(
        self,
        n_components=500,
        length_scale=1.0,
        kernel="rbf",
        random_state=None,
    ):
        self.n_components = n_components
        self.length_scale = length_scale
        self.kernel = kernel
        self.random_state = random_state

    def _sample_frequencies(self, rng, n_features: int, D: int) -> np.ndarray:
        """Sample spectral frequencies ω according to the chosen kernel."""
        z = rng.standard_normal((n_features, D))
        if self.kernel == "rbf":
            return z / self.length_scale
        elif self.kernel in ("matern32", "matern52"):
            nu = 1.5 if self.kernel == "matern32" else 2.5
            # u ~ Gamma(ν, 1); shape=(1,D) broadcasts over n_features
            u = rng.gamma(nu, 1.0, size=(1, D))
            return z * np.sqrt(nu / u) / self.length_scale
        else:
            raise ValueError(
                f"Unknown kernel '{self.kernel}'. "
                f"Choose from {_VALID_KERNELS}."
            )

    def _rff_transform(self, X: np.ndarray) -> np.ndarray:
        D = self.n_components
        return np.sqrt(2.0 / D) * np.cos(X @ self.omega_ + self.bias_)

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        D = self.n_components
        self.omega_ = self._sample_frequencies(rng, X.shape[1], D)
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
    * Preserves the kernel structure of the chosen stationary kernel

    Parameters
    ----------
    n_components : int
        Number of random Fourier features D. Higher values give a better
        kernel approximation; 500 is usually sufficient.
    length_scale : float
        Length scale of the kernel being approximated.
    kernel : str
        Kernel type.  One of ``'rbf'``, ``'matern32'``, ``'matern52'``.
        Matérn kernels use Student-t spectral sampling and are less smooth
        than RBF, making them a better prior for many physical models.
    random_state : int or None
        Seed for the random frequency draw, for reproducibility.
    """

    def __init__(
        self,
        n_components=500,
        length_scale=1.0,
        kernel="rbf",
        random_state=None,
    ):
        self.n_components = n_components
        self.length_scale = length_scale
        self.kernel = kernel
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
                kernel=self.kernel,
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
