# Copyright (c) 2026 takotime808

"""GPX surrogate — Rust-accelerated Kriging via SMT/egobox.

Wraps the ``GPX`` model from the SMT Surrogate Modeling Toolbox, which
reimplements sklearn-style Kriging (KRG) in compiled Rust via the egobox
library.  The probabilistic model is identical to a standard GP/Kriging
but training and prediction are 10–100× faster for the same dataset size.

Installation
------------
    pip install smt[gpx]

If the optional dependency is not installed this module defines placeholder
classes that raise ``ImportError`` at instantiation so the rest of the
package continues to work without it.
"""

import numpy as np

try:
    from smt.surrogate_models.gpx import GPX as _SMTGPX
    _GPX_AVAILABLE = True
except ImportError:
    _SMTGPX = None
    _GPX_AVAILABLE = False

from sklearn.base import BaseEstimator, RegressorMixin
from multioutreg.surrogates.conformal_mixin import ConformalMixin


def _check_available():
    if not _GPX_AVAILABLE:
        raise ImportError(
            "GPXSurrogate requires the smt package with the gpx extra. "
            "Install it with: pip install smt[gpx]"
        )


class _GPXEstimator(BaseEstimator, RegressorMixin):
    """Single-output GPX estimator (sklearn-compatible, used by GridSearchCV)."""

    def __init__(self, corr="squar_exp", poly="constant", n_start=10):
        self.corr = corr
        self.poly = poly
        self.n_start = n_start

    def fit(self, X, y):
        _check_available()
        self._model = _SMTGPX(
            corr=self.corr,
            poly=self.poly,
            n_start=self.n_start,
            print_global=False,
        )
        self._model.set_training_values(X, y.reshape(-1, 1))
        self._model.train()
        return self

    def predict(self, X, return_std=False):
        _check_available()
        y_pred = self._model.predict_values(X).ravel()
        if not return_std:
            return y_pred
        var = self._model.predict_variances(X).ravel()
        std = np.sqrt(np.maximum(var, 0.0))
        return y_pred, std


class GPXSurrogate(ConformalMixin):
    """Rust-accelerated Kriging surrogate via SMT/egobox GPX.

    Fits one :class:`_GPXEstimator` per output column using the SMT ``GPX``
    backend, which compiles the Kriging likelihood and gradient to native
    Rust for a 10–100× speedup over sklearn's ``GaussianProcessRegressor``
    at the same O(n³) asymptotic cost.

    Requires ``pip install smt[gpx]``.  The dependency is optional; if not
    installed an ``ImportError`` is raised at instantiation time.

    Parameters
    ----------
    corr : str
        Correlation function: ``"squar_exp"`` (RBF), ``"abs_exp"``,
        ``"matern52"``, or ``"matern32"``.
    poly : str
        Regression/trend term: ``"constant"``, ``"linear"``, or
        ``"quadratic"``.
    n_start : int
        Number of multistart optimizer runs for hyperparameter MLE.
    """

    def __init__(self, corr="squar_exp", poly="constant", n_start=10):
        _check_available()
        self.corr = corr
        self.poly = poly
        self.n_start = n_start

    def fit(self, X, Y):
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.n_outputs_ = Y.shape[1]
        self.estimators_ = [
            _GPXEstimator(
                corr=self.corr,
                poly=self.poly,
                n_start=self.n_start,
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
