# Copyright (c) 2026 takotime808

"""Gaussian copula wrapper: add output correlation to any per-output surrogate.

Most surrogates in the package fit one model per output, so their predictive
distribution treats the outputs as independent even when the outputs move
together. This wrapper keeps the wrapped surrogate's marginals (mean and std
per output) and couples them with a correlation matrix ``R`` estimated from
its residuals::

    Cov(Y | x) = diag(std(x)) R diag(std(x))

``R`` is estimated from out-of-fold residuals, since in-sample residuals of
an interpolating model (a GP, a deep forest) are close to zero and say nothing
about its errors on new points. Each residual is standardized by the
predicted std and then rank-transformed to normal scores, so ``R`` is the
Gaussian-copula correlation and is not thrown off by heavy-tailed residuals.
"""

import numpy as np
from scipy.stats import norm
from sklearn.model_selection import KFold

from multioutreg.surrogates.conformal_mixin import ConformalMixin


def _normal_scores(Z):
    """Column-wise rank transform to standard-normal scores."""
    n = Z.shape[0]
    ranks = np.argsort(np.argsort(Z, axis=0), axis=0) + 1
    return norm.ppf(ranks / (n + 1))


class GaussianCopulaSurrogate(ConformalMixin):
    """Couple a surrogate's per-output predictions with a residual correlation.

    Parameters
    ----------
    surrogate_cls : type | None
        Surrogate supplying the marginals; must accept ``fit(X, Y)`` and
        ``predict(X, return_std=True)``. Defaults to
        ``GaussianProcessSurrogate``.
    cv : int
        Folds used to collect out-of-fold residuals for ``R`` (at least 2).
    shrinkage : float in [0, 1]
        Shrink ``R`` towards the identity: ``(1 - s) R + s I``. Useful when
        there are few rows per output. Default 0.
    random_state : int | None
        Seed for the fold split.
    **surrogate_kwargs
        Passed to ``surrogate_cls``.

    Attributes
    ----------
    correlation_ : np.ndarray, shape (n_outputs, n_outputs)
        Estimated copula correlation ``R``.
    surrogate_ : object
        The wrapped surrogate refitted on all rows.
    """

    def __init__(self, surrogate_cls=None, cv=5, shrinkage=0.0, random_state=None, **surrogate_kwargs):
        self.surrogate_cls = surrogate_cls
        self.cv = cv
        self.shrinkage = shrinkage
        self.random_state = random_state
        self.surrogate_kwargs = surrogate_kwargs

    def _make_surrogate(self):
        cls = self.surrogate_cls
        if cls is None:
            from multioutreg.surrogates.gp_sklearn import GaussianProcessSurrogate

            cls = GaussianProcessSurrogate
        return cls(**self.surrogate_kwargs)

    @staticmethod
    def _predict_mean_std(model, X):
        mean, std = model.predict(X, return_std=True)
        mean = np.asarray(mean, dtype=float).reshape(len(X), -1)
        std = np.asarray(std, dtype=float).reshape(len(X), -1)
        return mean, std

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        n, m = Y.shape
        self.n_outputs_ = m
        if not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError(f"shrinkage must be in [0, 1], got {self.shrinkage}")
        n_splits = min(int(self.cv), n)
        if n_splits < 2:
            raise ValueError(f"cv must be >= 2 (and at most the number of rows); got {self.cv}")

        Z = np.empty_like(Y)
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        for train, test in folds.split(X):
            model = self._make_surrogate().fit(X[train], Y[train])
            mean, std = self._predict_mean_std(model, X[test])
            scale = np.where(std > 0, std, 1.0)
            Z[test] = (Y[test] - mean) / scale

        R = np.corrcoef(_normal_scores(Z), rowvar=False) if m > 1 else np.ones((1, 1))
        R = np.nan_to_num(np.atleast_2d(R), nan=0.0)  # a constant residual column has no correlation
        np.fill_diagonal(R, 1.0)
        R = (1.0 - self.shrinkage) * R + self.shrinkage * np.eye(m)
        # Clip to the nearest PSD matrix with unit diagonal, in case of rounding.
        w, V = np.linalg.eigh(R)
        R = (V * np.clip(w, 0.0, None)) @ V.T
        d = np.sqrt(np.diag(R))
        self.correlation_ = R / np.outer(d, d)
        self.oof_scores_ = Z

        self.surrogate_ = self._make_surrogate().fit(X, Y)
        return self

    def predict(self, X, return_std=False, return_cov=False):
        """Predict outputs; ``return_cov=True`` adds the coupled covariance.

        The mean and std are the wrapped surrogate's; only the covariance
        (shape ``(n_samples, n_outputs, n_outputs)``) changes.
        """
        X = np.asarray(X, dtype=float)
        if not (return_std or return_cov):
            return np.asarray(self.surrogate_.predict(X), dtype=float).reshape(len(X), -1)
        mean, std = self._predict_mean_std(self.surrogate_, X)
        if return_cov:
            cov = std[:, :, None] * self.correlation_[None, :, :] * std[:, None, :]
            return mean, cov
        return mean, std

    def sample(self, X, n_samples=500, seed=None):
        """Draws from the joint predictive distribution, shape ``(n, n_samples, n_outputs)``."""
        from multioutreg.metrics.joint_scores import gaussian_samples

        mean, cov = self.predict(X, return_cov=True)
        return gaussian_samples(mean, cov, n_samples=n_samples, seed=seed)

    def _conformal_point_predict(self, X):
        return self.predict(X)
