# Copyright (c) 2026 takotime808

"""Reduced-rank regression: a linear model whose coefficient matrix has rank r.

Independent per-output linear models estimate a ``p x m`` coefficient matrix
with no shared structure. When the outputs are driven by a few common linear
combinations of the inputs, constraining the matrix to rank ``r < min(p, m)``
removes most of the estimation noise (Izenman 1975). The fit is closed form:
fit ridge regression on standardized inputs and outputs, take the top ``r``
right singular vectors ``V_r`` of the fitted values, and project the
coefficients onto them, ``B_r = B V_r V_r^T``.

``rank="auto"`` picks ``r`` by K-fold cross-validated mean squared error
(smallest rank among ties).

The predictive std is the ordinary-least-squares prediction interval with the
residual variance of each output on ``n - p - 1`` degrees of freedom::

    var_j(x) = s_j^2 * (1 + 1/n + x^T (X^T X + alpha I)^-1 x)

on centred, standardized inputs.
"""

import numpy as np
from sklearn.model_selection import KFold

from multioutreg.surrogates.conformal_mixin import ConformalMixin


def _ridge_and_directions(Xs, Ys, alpha):
    """Ridge coefficients and the right singular vectors of the fitted values."""
    gram = Xs.T @ Xs + alpha * np.eye(Xs.shape[1])
    B = np.linalg.solve(gram, Xs.T @ Ys)
    _, _, Vt = np.linalg.svd(Xs @ B, full_matrices=False)
    return B, Vt


def _project(B, Vt, rank):
    V = Vt[:rank].T
    return B @ V @ V.T


class ReducedRankRegressionSurrogate(ConformalMixin):
    """Joint multi-output linear surrogate with a rank-constrained coefficient matrix.

    Parameters
    ----------
    rank : int | "auto"
        Rank of the coefficient matrix, clamped to ``min(n_features, n_outputs)``.
        ``"auto"`` chooses it by K-fold cross-validation.
    alpha : float
        Ridge penalty on the standardized inputs; the small default only keeps
        the solve well conditioned.
    cv : int
        Folds used when ``rank="auto"``.
    random_state : int | None
        Seed for the shuffled folds.

    Attributes
    ----------
    rank_ : int
    coef_ : np.ndarray
        ``(n_features, n_outputs)`` coefficients on the standardized inputs, in
        output units.
    cv_mse_ : np.ndarray | None
        Cross-validated MSE (standardized outputs) for ranks ``1..len(cv_mse_)``
        when ``rank="auto"``.
    residual_var_ : np.ndarray
        Per-output residual variance.
    """

    _multi_output = True  # signals AutoDetect to evaluate on full Y

    def __init__(self, rank="auto", alpha=1e-6, cv=5, random_state=0):
        self.rank = rank
        self.alpha = alpha
        self.cv = cv
        self.random_state = random_state

    def _select_rank(self, Xs, Ys, max_rank):
        n = Xs.shape[0]
        folds = KFold(n_splits=max(2, min(self.cv, n)), shuffle=True,
                      random_state=self.random_state)
        sse = np.zeros(max_rank)
        for train, test in folds.split(Xs):
            B, Vt = _ridge_and_directions(Xs[train], Ys[train], self.alpha)
            for r in range(1, max_rank + 1):
                sse[r - 1] += np.sum((Ys[test] - Xs[test] @ _project(B, Vt, r)) ** 2)
        self.cv_mse_ = sse / Ys.size
        return int(np.argmin(self.cv_mse_)) + 1

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        n, p = X.shape
        m = Y.shape[1]

        self.x_mean_ = X.mean(axis=0)
        self.x_scale_ = X.std(axis=0)
        self.x_scale_[self.x_scale_ == 0] = 1.0
        self.y_mean_ = Y.mean(axis=0)
        y_scale = Y.std(axis=0)
        y_scale[y_scale == 0] = 1.0
        Xs = (X - self.x_mean_) / self.x_scale_
        Ys = (Y - self.y_mean_) / y_scale

        max_rank = min(p, m)
        self.cv_mse_ = None
        if self.rank == "auto":
            self.rank_ = self._select_rank(Xs, Ys, max_rank) if n >= 4 else max_rank
        else:
            if int(self.rank) < 1:
                raise ValueError(f"rank must be >= 1 or 'auto', got {self.rank!r}")
            self.rank_ = min(int(self.rank), max_rank)

        B, Vt = _ridge_and_directions(Xs, Ys, self.alpha)
        self.coef_ = _project(B, Vt, self.rank_) * y_scale

        resid = Y - self.y_mean_ - Xs @ self.coef_
        self.residual_var_ = np.sum(resid ** 2, axis=0) / max(n - p - 1, 1)
        self.gram_inv_ = np.linalg.inv(Xs.T @ Xs + self.alpha * np.eye(p))
        self.n_train_ = n
        return self

    def predict(self, X, return_std=False):
        Xs = (np.asarray(X, dtype=float) - self.x_mean_) / self.x_scale_
        mean = self.y_mean_ + Xs @ self.coef_
        if not return_std:
            return mean
        leverage = 1.0 / self.n_train_ + np.einsum("ij,jk,ik->i", Xs, self.gram_inv_, Xs)
        return mean, np.sqrt(np.outer(1.0 + leverage, self.residual_var_))
