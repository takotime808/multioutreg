# Copyright (c) 2026 takotime808

"""Nyström Sparse GP surrogate.

Approximates a kernel GP using the Nyström method (Williams & Seeger, 2001):
m data-adaptive landmark points are selected at fit time, the kernel is
approximated as K ≈ K_nm K_mm⁻¹ K_mn, then BayesianRidge is fit in the
m-dimensional lifted space.

Training cost is O(m²n + m³), prediction O(m) — the same asymptotic class
as RFGP but with data-driven landmarks rather than random frequencies.
This makes it superior to RFGP when m << n and the data has structure.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import BayesianRidge

from multioutreg.surrogates.conformal_mixin import ConformalMixin


class _NystroemEstimator(BaseEstimator, RegressorMixin):
    """Single-output Nyström GP estimator (sklearn-compatible, used by GridSearchCV)."""

    def __init__(self, n_components=100, gamma=None, kernel="rbf", random_state=None):
        self.n_components = n_components
        self.gamma = gamma
        self.kernel = kernel
        self.random_state = random_state

    def fit(self, X, y):
        n_comp = min(self.n_components, X.shape[0])
        self._nystroem = Nystroem(
            kernel=self.kernel,
            gamma=self.gamma,
            n_components=n_comp,
            random_state=self.random_state,
        )
        Z = self._nystroem.fit_transform(X)
        self._br = BayesianRidge()
        self._br.fit(Z, y)
        return self

    def predict(self, X, return_std=False):
        Z = self._nystroem.transform(X)
        return self._br.predict(Z, return_std=return_std)


class NystroemGPSurrogate(ConformalMixin):
    """Nyström Sparse GP surrogate.

    Fits one :class:`_NystroemEstimator` per output column.  Data-adaptive
    landmark points give a better kernel approximation than random Fourier
    features when the number of landmarks m is small relative to n.

    Parameters
    ----------
    n_components : int
        Number of landmark points m. Higher = better approximation.
    gamma : float or None
        Kernel bandwidth parameter. None uses sklearn's default (1/n_features).
    kernel : str
        Kernel type passed to :class:`sklearn.kernel_approximation.Nystroem`.
        Defaults to ``"rbf"``.
    random_state : int or None
        Seed for reproducible landmark selection.
    """

    def __init__(self, n_components=100, gamma=None, kernel="rbf", random_state=None):
        self.n_components = n_components
        self.gamma = gamma
        self.kernel = kernel
        self.random_state = random_state

    def fit(self, X, Y):
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.n_outputs_ = Y.shape[1]
        self.estimators_ = [
            _NystroemEstimator(
                n_components=self.n_components,
                gamma=self.gamma,
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
