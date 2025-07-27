# Copyright (c) 2025 takotime808

import numpy as np
from sklearn.multioutput import MultiOutputRegressor

class BaseSurrogate:
    """Base class for multi-output surrogates."""
    def __init__(self, base_estimator):
        self.model = MultiOutputRegressor(base_estimator)

    def fit(self, X, Y):
        self.model.fit(X, Y)
        return self

    def predict(self, X, return_std=False):
        if return_std:
            try:
                return self.model.predict(X, return_std=True)
            except TypeError:
                preds = self.model.predict(X)
                std = np.zeros_like(preds)
                return preds, std
        return self.model.predict(X)