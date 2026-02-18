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
from sklearn.neural_network import MLPRegressor

from multioutreg.surrogates import (
    LinearRegressionSurrogate,
    GaussianProcessSurrogate,
    RandomForestSurrogate,
    GradientBoostingSurrogate,
    SVRSurrogate,
    KNeighborsSurrogate,
    DecisionTreeRegressorSurrogate,
    ConformalPredictionNetworkSurrogate,
    MultiFidelitySurrogate,
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
                if isinstance(surrogate, MultiFidelitySurrogate):
                    surrogate.fit((X, y[:, [i]]))
                else:
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
        cls,
        cv: int = 3,
        scoring: str = "neg_mean_squared_error",
        fidelity_levels: Sequence[str] | None = None,
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
            MLPRegressor(max_iter=500),
        ]

        param_spaces = [
            {},
            {"alpha": [1e-10, 1e-2]},
            {"n_estimators": [50, 100], "max_depth": [3, 5, None]},
            {"n_estimators": [50, 100], "max_depth": [3, 5]},
            {"C": [1.0, 10.0], "gamma": ["scale", "auto"]},
            {"n_neighbors": [3, 5, 7]},
            {"max_depth": [1, None]},
            {"hidden_layer_sizes": [(64,), (128,)], "alpha": [1e-4, 1e-3]},
        ]

        instance = cls(estimators, param_spaces, cv=cv, scoring=scoring)

        if fidelity_levels is None:
            instance._surrogate_constructors = [
                LinearRegressionSurrogate,
                GaussianProcessSurrogate,
                RandomForestSurrogate,
                GradientBoostingSurrogate,
                SVRSurrogate,
                KNeighborsSurrogate,
                DecisionTreeRegressorSurrogate,
                ConformalPredictionNetworkSurrogate,
            ]
        else:
            def wrap(cls_sur):
                return lambda **p: MultiFidelitySurrogate(
                    lambda: cls_sur(**p), fidelity_levels
                )

            instance._surrogate_constructors = [
                wrap(LinearRegressionSurrogate),
                wrap(GaussianProcessSurrogate),
                wrap(RandomForestSurrogate),
                wrap(GradientBoostingSurrogate),
                wrap(SVRSurrogate),
                wrap(KNeighborsSurrogate),
                wrap(DecisionTreeRegressorSurrogate),
                wrap(ConformalPredictionNetworkSurrogate),
            ]
            instance.fidelity_levels = list(fidelity_levels)

        return instance

    def calibrate_conformal(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        **kwargs,
    ) -> "AutoDetectMultiOutputRegressor":
        """Calibrate conformal prediction intervals using held-out data.

        Computes absolute residuals on the calibration set using the
        already-fitted model. These residuals are used at prediction time
        to construct distribution-free intervals with coverage guarantees.

        Parameters
        ----------
        X_cal : np.ndarray
            Calibration features (must NOT overlap with training data).
        y_cal : np.ndarray
            Calibration targets.

        Returns
        -------
        self
        """
        from multioutreg.conformal.base import BaseConformalPredictor

        if not hasattr(self, "models_"):
            raise AttributeError("Estimator not fitted. Call fit() first.")

        y_cal = np.asarray(y_cal)
        if y_cal.ndim == 1:
            y_cal = y_cal.reshape(-1, 1)

        y_cal_pred = self.predict(X_cal)
        if y_cal_pred.ndim == 1:
            y_cal_pred = y_cal_pred.reshape(-1, 1)

        self._conformal_residuals = np.abs(y_cal - y_cal_pred)
        self._conformal_n_outputs = y_cal.shape[1]
        return self

    def predict_interval(
        self, X: np.ndarray, alpha: float = 0.1
    ) -> tuple:
        """Return conformal prediction intervals.

        Parameters
        ----------
        X : np.ndarray
        alpha : float
            Miscoverage level. Intervals target 1-alpha coverage.

        Returns
        -------
        y_lower, y_upper : np.ndarray
        """
        if not hasattr(self, "_conformal_residuals"):
            raise AttributeError(
                "No conformal predictor calibrated. Call calibrate_conformal() first."
            )
        from multioutreg.conformal.base import BaseConformalPredictor

        y_pred = self.predict(X)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        q = np.array([
            BaseConformalPredictor._conformal_quantile(
                self._conformal_residuals[:, j], alpha
            )
            for j in range(self._conformal_n_outputs)
        ])

        y_lower = y_pred - q[np.newaxis, :]
        y_upper = y_pred + q[np.newaxis, :]
        return y_lower, y_upper