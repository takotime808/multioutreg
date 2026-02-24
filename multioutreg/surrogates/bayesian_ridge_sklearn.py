# Copyright (c) 2026 takotime808

import numpy as np
from sklearn.linear_model import BayesianRidge
from multioutreg.surrogates.base_sklearn import BaseSurrogate


class BayesianRidgeSurrogate(BaseSurrogate):
    """Bayesian Ridge Regression surrogate with analytic posterior uncertainty.

    Wraps sklearn's BayesianRidge, which computes an exact Gaussian posterior
    over weights.  Training is O(p³) in feature dimension -- orders of magnitude
    cheaper than a GP for large datasets while still providing principled
    uncertainty estimates without any sampling.
    """

    def __init__(self, **kwargs):
        super().__init__(BayesianRidge(**kwargs))

    def predict(self, X, return_std=False):
        if not return_std:
            return self.model.predict(X)

        preds, stds = [], []
        for est in self.model.estimators_:
            y_pred, y_std = est.predict(X, return_std=True)
            preds.append(y_pred)
            stds.append(y_std)

        return np.column_stack(preds), np.column_stack(stds)
