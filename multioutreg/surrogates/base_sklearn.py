# Copyright (c) 2025 takotime808

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from multioutreg.surrogates.conformal_mixin import ConformalMixin


class BaseSurrogate(ConformalMixin):
    """Base class for multi-output surrogates."""

    def __init__(self, base_estimator):
        self.model = MultiOutputRegressor(base_estimator)

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