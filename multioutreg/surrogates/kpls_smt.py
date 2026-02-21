# Copyright (c) 2026 takotime808

"""KPLS surrogate — Kriging with Partial Least Squares dimensionality reduction.

Wraps the ``KPLS`` model from the SMT Surrogate Modeling Toolbox.  KPLS
reduces the effective input dimensionality with a PLS projection before
fitting a Kriging (GP) model, making it well-suited for high-dimensional
inputs (p >> n) where standard Kriging would otherwise be impractical.

The probabilistic model gives the same Kriging mean and variance as a
standard GP in the projected space.

Installation
------------
    pip install smt          # core SMT (includes KPLS)
    # or
    pip install smt[gpx]     # also installs Rust-accelerated GPX backend

If the optional dependency is not installed this module defines placeholder
classes that raise ``ImportError`` at instantiation so the rest of the
package continues to work without it.
"""

import numpy as np

try:
    from smt.surrogate_models import KPLS as _SMTKPLS
    _KPLS_AVAILABLE = True
except ImportError:
    _SMTKPLS = None
    _KPLS_AVAILABLE = False

from sklearn.base import BaseEstimator, RegressorMixin
from multioutreg.surrogates.conformal_mixin import ConformalMixin


def _check_available():
    if not _KPLS_AVAILABLE:
        raise ImportError(
            "KPLSSurrogate requires the smt package. "
            "Install it with: pip install smt"
        )


class _KPLSEstimator(BaseEstimator, RegressorMixin):
    """Single-output KPLS estimator (sklearn-compatible, used by GridSearchCV)."""

    def __init__(self, n_comp=2, corr="squar_exp", poly="constant"):
        self.n_comp = n_comp
        self.corr = corr
        self.poly = poly

    def fit(self, X, y):
        _check_available()
        n_comp = min(self.n_comp, X.shape[1])
        self._model = _SMTKPLS(
            n_comp=n_comp,
            corr=self.corr,
            poly=self.poly,
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


class KPLSSurrogate(ConformalMixin):
    """Kriging with Partial Least Squares (KPLS) surrogate via SMT.

    KPLS reduces input dimensionality with PLS before fitting a Kriging model.
    This makes it well-suited for high-dimensional problems (p >> n) where
    standard GP/Kriging would be slow or numerically unstable due to the
    large covariance matrix.

    Requires ``pip install smt``.  The dependency is optional; if not
    installed an ``ImportError`` is raised at instantiation time.

    Parameters
    ----------
    n_comp : int
        Number of PLS components.  Defaults to 2.  Will be clamped to
        ``min(n_comp, n_features)`` at fit time.
    corr : str
        Correlation function: ``"squar_exp"`` (RBF), ``"abs_exp"``,
        ``"matern52"``, or ``"matern32"``.
    poly : str
        Regression/trend term: ``"constant"``, ``"linear"``, or
        ``"quadratic"``.
    """

    def __init__(self, n_comp=2, corr="squar_exp", poly="constant"):
        _check_available()
        self.n_comp = n_comp
        self.corr = corr
        self.poly = poly

    def fit(self, X, Y):
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.n_outputs_ = Y.shape[1]
        self.estimators_ = [
            _KPLSEstimator(
                n_comp=self.n_comp,
                corr=self.corr,
                poly=self.poly,
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
