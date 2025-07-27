# Copyright (c) 2025 takotime808

"""Utilities for automatically selecting the best regressor per output."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import GridSearchCV


class AutoDetectMultiOutputRegressor(BaseEstimator, RegressorMixin):
    """Fit a separate estimator per output choosing the best via grid search."""

    def __init__(self, estimators: Sequence[BaseEstimator], param_spaces: Sequence[dict], cv: int = 3, scoring: str = "neg_mean_squared_error") -> None:
        if len(estimators) != len(param_spaces):
            raise ValueError("Each estimator must have a corresponding param space")
        self.estimators = list(estimators)
        self.param_spaces = list(param_spaces)
        self.cv = cv
        self.scoring = scoring

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AutoDetectMultiOutputRegressor":
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.models_ = []
        for i in range(y.shape[1]):
            best_score = -np.inf
            best_est = None
            for est, params in zip(self.estimators, self.param_spaces):
                gs = GridSearchCV(est, params, cv=self.cv, scoring=self.scoring)
                gs.fit(X, y[:, i])
                if gs.best_score_ > best_score:
                    best_score = gs.best_score_
                    best_est = clone(gs.best_estimator_)
            if best_est is None:
                raise RuntimeError("No valid estimator found")
            best_est.fit(X, y[:, i])
            self.models_.append(best_est)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "models_"):
            raise AttributeError("Estimator not fitted")
        preds = [model.predict(X) for model in self.models_]
        return np.column_stack(preds)