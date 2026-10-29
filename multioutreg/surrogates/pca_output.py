# Copyright (c) 2026 takotime808

"""Output-reduction wrapper: fit any surrogate on the principal components of Y.

Field outputs (spectra, pressure distributions, time series sampled on a grid)
have hundreds of strongly correlated columns. Fitting one model per column is
slow and ignores the correlation. This wrapper standardizes ``Y``, projects it
onto its top ``k`` principal components, fits the wrapped surrogate on the
``k`` component scores, and maps predictions back to the original outputs.

Uncertainty is mapped back the same way. With component scores
``z_k ~ N(mu_k, sigma_k^2)`` treated as independent and loadings ``V``
(``k x m``), the predictive covariance of the outputs is::

    Cov(Y) = S V^T diag(sigma^2) V S + diag(r)

where ``S`` is the diagonal matrix of output scales and ``r`` is the per-output
variance of the training reconstruction error (the part of ``Y`` the discarded
components carry). ``predict(X, return_std=True)`` returns the square root of
its diagonal; ``predict(X, return_cov=True)`` returns the full matrix, so the
outputs come out correlated even when the wrapped surrogate is per-output.
"""

import numpy as np
from sklearn.decomposition import PCA

from multioutreg.surrogates.conformal_mixin import ConformalMixin


class PCAOutputSurrogate(ConformalMixin):
    """Fit a surrogate on the top principal components of the outputs.

    Parameters
    ----------
    surrogate_cls : type | None
        Surrogate fitted on the component scores. It must accept
        ``fit(X, Z)`` and ``predict(X, return_std=True)``. Defaults to
        ``GaussianProcessSurrogate``.
    n_components : int | float
        Number of components to keep, or (if a float in ``(0, 1]``) the
        fraction of output variance to retain, as in
        :class:`sklearn.decomposition.PCA`. Clamped to the number of outputs
        and training rows.
    standardize : bool
        Scale each output to unit variance before PCA, so outputs in large
        units do not dominate the components. Default ``True``.
    include_truncation_error : bool
        Add the per-output variance of the training reconstruction error to
        the predictive variance. Without it the discarded components count
        as known exactly and the std is too small. Default ``True``.
    **surrogate_kwargs
        Passed to ``surrogate_cls``.

    Attributes
    ----------
    pca_ : sklearn.decomposition.PCA
    n_components_ : int
        Number of components kept.
    explained_variance_ratio_ : np.ndarray
        Fraction of standardized output variance each kept component explains.
    truncation_var_ : np.ndarray
        Per-output variance of the reconstruction error on the training data,
        in the original output units.
    """

    def __init__(
        self,
        surrogate_cls=None,
        n_components=0.99,
        standardize=True,
        include_truncation_error=True,
        **surrogate_kwargs,
    ):
        self.surrogate_cls = surrogate_cls
        self.n_components = n_components
        self.standardize = standardize
        self.include_truncation_error = include_truncation_error
        self.surrogate_kwargs = surrogate_kwargs

    def _make_surrogate(self):
        cls = self.surrogate_cls
        if cls is None:
            from multioutreg.surrogates.gp_sklearn import GaussianProcessSurrogate

            cls = GaussianProcessSurrogate
        return cls(**self.surrogate_kwargs)

    def _resolve_n_components(self, n_samples, n_outputs):
        limit = min(n_samples, n_outputs)
        n = self.n_components
        if isinstance(n, (int, np.integer)) and not isinstance(n, bool):
            if n < 1:
                raise ValueError(f"n_components must be >= 1, got {n}")
            return min(int(n), limit)
        n = float(n)
        if not 0.0 < n <= 1.0:
            raise ValueError(f"a float n_components must be in (0, 1], got {n}")
        # A fraction of 1.0 keeps every component; PCA treats the float 1.0 as
        # "one component", so pass the integer instead.
        return limit if n == 1.0 else n

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.n_outputs_ = Y.shape[1]

        self.y_mean_ = Y.mean(axis=0)
        if self.standardize:
            scale = Y.std(axis=0)
            scale[scale == 0] = 1.0
        else:
            scale = np.ones(self.n_outputs_)
        self.y_scale_ = scale
        Y_std = (Y - self.y_mean_) / self.y_scale_

        n_comp = self._resolve_n_components(*Y.shape)
        # The data are already centred, so PCA's own mean is ~0; keep it anyway
        # so transform/inverse_transform stay exact.
        self.pca_ = PCA(n_components=n_comp, svd_solver="full").fit(Y_std)
        self.n_components_ = self.pca_.n_components_
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_

        Z = self.pca_.transform(Y_std)
        recon = self.y_mean_ + self.y_scale_ * self.pca_.inverse_transform(Z)
        self.truncation_var_ = np.var(Y - recon, axis=0)

        self.surrogate_ = self._make_surrogate().fit(X, Z)
        return self

    def _component_predict(self, X, return_std):
        out = self.surrogate_.predict(X, return_std=return_std)
        if not return_std:
            return np.asarray(out).reshape(len(X), -1), None
        z_mean, z_std = out
        return (
            np.asarray(z_mean).reshape(len(X), -1),
            np.asarray(z_std).reshape(len(X), -1),
        )

    def predict_components(self, X, return_std=False):
        """Predicted principal-component scores, shape ``(n, n_components_)``."""
        X = np.asarray(X, dtype=float)
        z_mean, z_std = self._component_predict(X, return_std)
        return (z_mean, z_std) if return_std else z_mean

    def predict(self, X, return_std=False, return_cov=False):
        """Predict outputs, optionally with per-output std or the full covariance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        return_std : bool
            Also return the per-output std, shape ``(n_samples, n_outputs)``.
        return_cov : bool
            Also return the predictive covariance of the outputs for each row,
            shape ``(n_samples, n_outputs, n_outputs)``. Takes precedence over
            ``return_std``.
        """
        X = np.asarray(X, dtype=float)
        need_unc = return_std or return_cov
        z_mean, z_std = self._component_predict(X, need_unc)

        mean = self.y_mean_ + self.y_scale_ * self.pca_.inverse_transform(z_mean)
        if not need_unc:
            return mean

        # Loadings in original output units: row k is component k scaled per output.
        L = self.pca_.components_ * self.y_scale_  # (k, m)
        noise = self.truncation_var_ if self.include_truncation_error else 0.0
        z_var = np.square(z_std)  # (n, k)
        if return_cov:
            cov = np.einsum("nk,ki,kj->nij", z_var, L, L)
            cov[:, np.arange(self.n_outputs_), np.arange(self.n_outputs_)] += noise
            return mean, cov
        var = z_var @ np.square(L) + noise
        return mean, np.sqrt(np.maximum(var, 0.0))

    def _conformal_point_predict(self, X):
        return np.asarray(self.predict(X)).reshape(len(X), -1)
