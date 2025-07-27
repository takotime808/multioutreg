# Copyright (c) 2025 takotime808

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from multioutreg.surrogates.base_sklearn import BaseSurrogate

class RandomForestSurrogate(BaseSurrogate):
    """Random Forest surrogate using ensemble uncertainty."""
    def __init__(self, **kwargs):
        super().__init__(RandomForestRegressor(**kwargs))

    def predict(self, X, return_std=False):
        preds = self.model.predict(X)
        if not return_std:
            return preds

        # Estimate std across trees for each output dimension
        stds = []
        for i, estimator in enumerate(self.model.estimators_):
            # each estimator corresponds to one output dimension
            output_preds = np.array([tree.predict(X) for tree in estimator.estimators_])
            stds.append(output_preds.std(axis=0))
        std = np.column_stack(stds)
        return preds, std