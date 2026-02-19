# Copyright (c) 2026 takotime808

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from multioutreg.surrogates.base_sklearn import BaseSurrogate


class ExtraTreesRegressorSurrogate(BaseSurrogate):
    """Extra-Trees surrogate using ensemble variance for uncertainty."""

    def __init__(self, **kwargs):
        super().__init__(ExtraTreesRegressor(**kwargs))

    def predict(self, X, return_std=False):
        preds = self.model.predict(X)
        if not return_std:
            return preds

        # Estimate std across trees for each output dimension
        stds = []
        for estimator in self.model.estimators_:
            output_preds = np.array([tree.predict(X) for tree in estimator.estimators_])
            stds.append(output_preds.std(axis=0))
        std = np.column_stack(stds)
        return preds, std
