# Copyright (c) 2026 takotime808

"""Partial least squares surrogate: one joint linear model for all outputs.

``PLSRegression`` projects the inputs onto ``k`` latent directions chosen to
covary with the outputs, then regresses every output on the same ``k`` scores.
The outputs share their latent structure, so it suits many correlated outputs
and collinear or high-dimensional inputs (``p`` close to or above ``n``).

The predictive std treats the fitted model as ordinary least squares on the
latent scores ``T`` (``n x k``)::

    var_j(x) = s_j^2 * (1 + 1/n + t(x)^T (T^T T)^-1 t(x))

where ``s_j^2`` is the residual variance of output ``j`` on ``n - k - 1``
degrees of freedom and ``t(x)`` are the scores of ``x``. This ignores the
uncertainty in the latent directions themselves, so it is slightly optimistic
on small designs.
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from multioutreg.surrogates.conformal_mixin import ConformalMixin


class PLSSurrogate(ConformalMixin):
    """Joint multi-output partial least squares surrogate.

    Parameters
    ----------
    n_components : int
        Number of latent components, clamped to ``min(n_features, n_samples - 1)``.
    scale : bool
        Scale inputs and outputs to unit variance (``PLSRegression`` default).
    **kwargs
        Passed to :class:`sklearn.cross_decomposition.PLSRegression`.

    Attributes
    ----------
    pls_ : PLSRegression
    n_components_ : int
    residual_var_ : np.ndarray
        Per-output residual variance ``s_j^2``.
    """

    _multi_output = True  # signals AutoDetect to evaluate on full Y

    def __init__(self, n_components=2, scale=True, **kwargs):
        self.n_components = n_components
        self.scale = scale
        self.pls_kwargs = kwargs

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        n, p = X.shape
        self.n_components_ = max(1, min(int(self.n_components), p, n - 1))
        self.pls_ = PLSRegression(
            n_components=self.n_components_, scale=self.scale, **self.pls_kwargs
        ).fit(X, Y)

        resid = Y - self._mean(X)
        dof = max(n - self.n_components_ - 1, 1)
        self.residual_var_ = np.sum(resid ** 2, axis=0) / dof
        T = self.pls_.x_scores_
        self.score_gram_inv_ = np.linalg.pinv(T.T @ T)
        self.n_train_ = n
        return self

    def _mean(self, X):
        return np.asarray(self.pls_.predict(X)).reshape(len(X), -1)

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=float)
        mean = self._mean(X)
        if not return_std:
            return mean
        t = self.pls_.transform(X)
        leverage = 1.0 / self.n_train_ + np.einsum("ij,jk,ik->i", t, self.score_gram_inv_, t)
        std = np.sqrt(np.outer(1.0 + leverage, self.residual_var_))
        return mean, std
