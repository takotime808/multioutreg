# Copyright (c) 2025 takotime808

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from multioutreg.surrogates.conformal_mixin import ConformalMixin


def scaled(estimator):
    """``StandardScaler`` + *estimator*, for models whose fit depends on input units."""
    return make_pipeline(StandardScaler(), estimator)


class BaseSurrogate(ConformalMixin):
    """Base class for multi-output surrogates.

    ``scale_inputs=True`` standardises X (fitted on the training data) before the
    estimator; distance/kernel/penalty-based models (GP, SVR, KNN, MLP, Lasso,
    ElasticNet) otherwise depend on the units of the inputs.
    """

    def __init__(self, base_estimator, scale_inputs=False):
        self.scale_inputs = scale_inputs
        self.model = self._make_multi(base_estimator)

    def _make_multi(self, estimator):
        if getattr(self, "scale_inputs", False):
            estimator = scaled(estimator)
        return MultiOutputRegressor(estimator)

    def fit(self, X, Y):
        self.model.fit(X, Y)
        return self

    def predict(self, X, return_std=False):
        if return_std:
            # MultiOutputRegressor.predict does not accept return_std, so query
            # each per-output estimator directly. Estimators without native
            # uncertainty fall back to zero std for that output.
            preds, stds = [], []
            for est in self.model.estimators_:
                try:
                    p, s = est.predict(X, return_std=True)
                except TypeError:
                    p = est.predict(X)
                    s = np.zeros_like(p)
                preds.append(np.asarray(p).ravel())
                stds.append(np.asarray(s).ravel())
            return np.column_stack(preds), np.column_stack(stds)
        return self.model.predict(X)

    def _conformal_point_predict(self, X):
        preds = np.asarray(self.model.predict(X))
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return preds