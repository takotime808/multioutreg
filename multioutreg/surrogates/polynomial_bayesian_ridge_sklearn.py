# Copyright (c) 2026 takotime808

"""Polynomial Bayesian Ridge surrogate.

Lifts inputs to a polynomial feature space then fits BayesianRidge, giving
an analytic posterior over a nonlinear function class at no extra sampling cost.
Training is O((p^d)³) in the lifted feature dimension, so it stays cheap for
low-dimensional inputs (p < ~15) at degree d=2.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures

from multioutreg.surrogates.base_sklearn import BaseSurrogate


class _PBREstimator(BaseEstimator, RegressorMixin):
    """Single-output Polynomial Bayesian Ridge estimator (sklearn-compatible)."""

    def __init__(self, degree=2, interaction_only=False):
        self.degree = degree
        self.interaction_only = interaction_only

    def fit(self, X, y):
        self._poly = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=False,
        )
        X_poly = self._poly.fit_transform(X)
        self._br = BayesianRidge()
        self._br.fit(X_poly, y)
        return self

    def predict(self, X, return_std=False):
        X_poly = self._poly.transform(X)
        return self._br.predict(X_poly, return_std=return_std)


class PolynomialBayesianRidgeSurrogate(BaseSurrogate):
    """Polynomial Bayesian Ridge surrogate with analytic posterior uncertainty.

    Applies a polynomial feature expansion to the inputs before fitting
    BayesianRidge.  This gives an analytic Gaussian posterior over a
    nonlinear hypothesis class without any sampling overhead.

    Parameters
    ----------
    degree : int
        Degree of the polynomial expansion (default 2).
    interaction_only : bool
        If True, only interaction terms are produced (no x²-style terms).
        Reduces the feature count significantly for higher-degree expansions.
    """

    def __init__(self, degree=2, interaction_only=False, **kwargs):
        super().__init__(_PBREstimator(degree=degree, interaction_only=interaction_only))
        self.degree = degree
        self.interaction_only = interaction_only

    def predict(self, X, return_std=False):
        if not return_std:
            return self.model.predict(X)

        preds, stds = [], []
        for est in self.model.estimators_:
            p, s = est.predict(X, return_std=True)
            preds.append(p)
            stds.append(s)
        return np.column_stack(preds), np.column_stack(stds)
