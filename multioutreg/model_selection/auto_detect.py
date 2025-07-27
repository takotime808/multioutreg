# Copyright (c) 2025 takotime808

"""Utilities for automatically selecting the best regressor per output."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from multioutreg.surrogates import (
    LinearRegressionSurrogate,
    GaussianProcessSurrogate,
    RandomForestSurrogate,
    GradientBoostingSurrogate,
    SVRSurrogate,
    KNeighborsSurrogate,
    DecisionTreeRegressorSurrogate,
)

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
            best_idx = None
            best_params = None
            for idx, (est, params) in enumerate(zip(self.estimators, self.param_spaces)):
                gs = GridSearchCV(est, params, cv=self.cv, scoring=self.scoring)
                gs.fit(X, y[:, i])
                if gs.best_score_ > best_score:
                    best_score = gs.best_score_
                    best_est = clone(gs.best_estimator_)
                    best_idx = idx
                    best_params = gs.best_params_
            if best_est is None:
                raise RuntimeError("No valid estimator found")

            if hasattr(self, "_surrogate_constructors") and best_idx is not None:
                surrogate = self._surrogate_constructors[best_idx](**best_params)
                surrogate.fit(X, y[:, [i]])
                self.models_.append(surrogate)
            else:
                best_est.fit(X, y[:, i])
                self.models_.append(best_est)

        # expose base estimators for compatibility with plotting utilities
        self.estimators_ = []
        for model in self.models_:
            if hasattr(model, "model") and hasattr(model.model, "estimators_"):
                # vendored surrogate using MultiOutputRegressor internally
                self.estimators_.append(model.model.estimators_[0])
            elif hasattr(model, "estimators_"):
                self.estimators_.append(model.estimators_[0])
            else:
                self.estimators_.append(model)

        return self

    def predict(self, X: np.ndarray, return_std: bool = False) -> np.ndarray:
        if not hasattr(self, "models_"):
            raise AttributeError("Estimator not fitted")
        preds = []
        stds = []
        for model in self.models_:
            if return_std:
                try:
                    pred, std = model.predict(X, return_std=True)
                except TypeError:
                    pred = model.predict(X)
                    std = np.zeros_like(pred)
            else:
                pred = model.predict(X)
                std = None

            pred = np.asarray(pred)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            preds.append(pred)

            if return_std:
                std = np.asarray(std)
                if std.ndim == 1:
                    std = std.reshape(-1, 1)
                stds.append(std)

        pred_mat = np.column_stack(preds)
        if return_std:
            std_mat = np.column_stack(stds)
            return pred_mat, std_mat
        return pred_mat

    @classmethod
    def with_vendored_surrogates(
        cls, cv: int = 3, scoring: str = "neg_mean_squared_error"
    ) -> "AutoDetectMultiOutputRegressor":
        """Return instance configured to search all vendored surrogates."""

        estimators = [
            LinearRegression(),
            GaussianProcessRegressor(),
            RandomForestRegressor(),
            GradientBoostingRegressor(),
            SVR(),
            KNeighborsRegressor(),
            DecisionTreeRegressor(),
        ]

        param_spaces = [
            {},
            {"alpha": [1e-10, 1e-2]},
            {"n_estimators": [50, 100], "max_depth": [3, 5, None]},
            {"n_estimators": [50, 100], "max_depth": [3, 5]},
            {"C": [1.0, 10.0], "gamma": ["scale", "auto"]},
            {"n_neighbors": [3, 5, 7]},
            {"max_depth": [1, None]},
        ]

        instance = cls(estimators, param_spaces, cv=cv, scoring=scoring)

        instance._surrogate_constructors = [
            LinearRegressionSurrogate,
            GaussianProcessSurrogate,
            RandomForestSurrogate,
            GradientBoostingSurrogate,
            SVRSurrogate,
            KNeighborsSurrogate,
            DecisionTreeRegressorSurrogate,
        ]
        return instance