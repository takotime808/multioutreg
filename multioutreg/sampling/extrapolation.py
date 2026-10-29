# Copyright (c) 2026 takotime808

"""Flag prediction points that lie outside the training design.

A surrogate's prediction far from its training data is an extrapolation, and
each model family extrapolates differently: trees hold the boundary value,
GPs revert to their mean, networks are unconstrained. Their std does not
always warn about it either. :class:`ExtrapolationDetector` scores how far a
point is from the training inputs and flags points beyond a threshold taken
from the training data themselves.

Methods (inputs are standardized first):

``"knn"`` (default)
    Mean distance to the ``n_neighbors`` nearest training points. Follows the
    actual shape of the design, including holes inside its bounding box.
``"mahalanobis"``
    Mahalanobis distance from the training mean. Cheap; assumes a roughly
    elliptical design.
``"hull"``
    Outside the convex hull of the training points (score 1) or inside
    (score 0), via a linear program per point. Exact, but slower for large
    designs and conservative in high dimensions, where most new points are
    outside the hull.
``"box"``
    Outside the per-feature training range; the score is the largest
    standardized distance beyond it.

For ``"knn"`` and ``"mahalanobis"`` the threshold is the ``quantile`` of the
training points' own scores, computed leave-one-out for ``"knn"`` so a point
is not its own neighbour.
"""

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import cKDTree

_METHODS = ("knn", "mahalanobis", "hull", "box")


class ExtrapolationDetector:
    """Score and flag points by their distance from the training inputs.

    Parameters
    ----------
    method : {"knn", "mahalanobis", "hull", "box"}
    n_neighbors : int
        Neighbours averaged by ``"knn"``.
    quantile : float in (0, 1]
        Training-score quantile used as the threshold for ``"knn"`` and
        ``"mahalanobis"``. Default 0.99.

    Attributes
    ----------
    threshold_ : float
        Scores above this are flagged.
    train_scores_ : np.ndarray
        Training points' own scores (leave-one-out for ``"knn"``).
    """

    def __init__(self, method="knn", n_neighbors=5, quantile=0.99):
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
        if not 0.0 < quantile <= 1.0:
            raise ValueError(f"quantile must be in (0, 1], got {quantile}")
        self.method = method
        self.n_neighbors = n_neighbors
        self.quantile = quantile

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self._mean = X.mean(axis=0)
        scale = X.std(axis=0)
        self._scale = np.where(scale == 0, 1.0, scale)
        Xs = (X - self._mean) / self._scale
        self._X = Xs

        if self.method == "knn":
            self._k = max(1, min(self.n_neighbors, len(Xs) - 1))
            self._tree = cKDTree(Xs)
            # k + 1 neighbours, dropping the point itself (distance 0).
            dist, _ = self._tree.query(Xs, k=self._k + 1)
            self.train_scores_ = np.atleast_2d(dist)[:, 1:].mean(axis=1)
        elif self.method == "mahalanobis":
            self._prec = np.linalg.pinv(np.atleast_2d(np.cov(Xs, rowvar=False)))
            self.train_scores_ = self._mahalanobis(Xs)
        elif self.method == "box":
            self._lo, self._hi = Xs.min(axis=0), Xs.max(axis=0)
            self.train_scores_ = np.zeros(len(Xs))
        else:  # hull
            self.train_scores_ = np.zeros(len(Xs))

        if self.method in ("knn", "mahalanobis"):
            self.threshold_ = float(np.quantile(self.train_scores_, self.quantile))
        else:
            self.threshold_ = 0.0
        return self

    def _mahalanobis(self, Xs):
        return np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", Xs, self._prec, Xs), 0.0))

    def _in_hull(self, x):
        # x is in the hull iff some lambda >= 0 with sum 1 has X^T lambda = x.
        n = len(self._X)
        A_eq = np.vstack([self._X.T, np.ones((1, n))])
        b_eq = np.append(x, 1.0)
        res = linprog(np.zeros(n), A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
        return res.status == 0

    def score(self, X):
        """Distance-from-design score per row; larger is further out."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Xs = (X - self._mean) / self._scale
        if self.method == "knn":
            dist, _ = self._tree.query(Xs, k=self._k)
            return np.atleast_2d(dist.reshape(len(Xs), -1)).mean(axis=1)
        if self.method == "mahalanobis":
            return self._mahalanobis(Xs)
        if self.method == "box":
            beyond = np.maximum(self._lo - Xs, 0.0) + np.maximum(Xs - self._hi, 0.0)
            return beyond.max(axis=1)
        return np.array([0.0 if self._in_hull(x) else 1.0 for x in Xs])

    def is_extrapolation(self, X):
        """Boolean mask: ``True`` where the point is outside the training design."""
        return self.score(X) > self.threshold_

    def fit_flag(self, X_train, X):
        """Fit on ``X_train`` and return the flags for ``X``."""
        return self.fit(X_train).is_extrapolation(X)
