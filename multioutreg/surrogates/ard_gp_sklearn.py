# Copyright (c) 2026 takotime808

"""ARD Gaussian Process surrogate.

Fits a Gaussian Process Regressor with one length scale per input feature
(Automatic Relevance Determination, ARD).  The standard RBF kernel uses a
single shared length scale across all dimensions; ARD allows the optimizer
to assign small scales to informative features and large scales to
irrelevant ones, effectively performing soft feature selection during GP
hyperparameter optimisation.

Zero new dependencies — uses only ``sklearn.gaussian_process``.

Training cost is O(n³) — same as the standard GP.  The kernel parameter
count grows with input dimensionality (p extra length scales), so the MLE
optimisation is slightly more expensive than the scalar-kernel GP for large p.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from multioutreg.surrogates.conformal_mixin import ConformalMixin


class _ARDGPEstimator(BaseEstimator, RegressorMixin):
    """Single-output ARD-GP estimator (sklearn-compatible, used by GridSearchCV).

    The ARD kernel is constructed at ``fit()`` time because the number of
    features (and hence the length of the length-scale vector) is unknown
    until training data is provided.
    """

    def __init__(self, alpha=1e-6, n_restarts_optimizer=0):
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer

    def fit(self, X, y):
        kernel = RBF(length_scale=np.ones(X.shape[1]))
        self._gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
        )
        self._gpr.fit(X, y)
        return self

    def predict(self, X, return_std=False):
        return self._gpr.predict(X, return_std=return_std)


class ARDGPSurrogate(ConformalMixin):
    """ARD Gaussian Process surrogate.

    Fits one :class:`_ARDGPEstimator` per output column.  Unlike the
    standard :class:`~multioutreg.surrogates.GaussianProcessSurrogate`
    which uses a scalar RBF length scale, this surrogate assigns an
    independent length scale to each input feature so that the GP
    hyperparameter optimisation can automatically down-weight irrelevant
    inputs.

    Training cost is O(n³) — same gate as the standard GP.

    Parameters
    ----------
    alpha : float
        Noise regularisation added to the diagonal of the kernel matrix
        (equivalent to observation noise variance).
    n_restarts_optimizer : int
        Number of multi-start optimizer restarts for kernel hyperparameter
        MLE.  Set to 0 for a single optimisation run (fastest).
    """

    def __init__(self, alpha=1e-6, n_restarts_optimizer=0):
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer

    def fit(self, X, Y):
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.n_outputs_ = Y.shape[1]
        self.estimators_ = [
            _ARDGPEstimator(
                alpha=self.alpha,
                n_restarts_optimizer=self.n_restarts_optimizer,
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
