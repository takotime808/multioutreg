# Copyright (c) 2025 takotime808

"""Utilities for automatically selecting the best regressor per output."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
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
    ExtraTreesRegressorSurrogate,
    NGBoostSurrogate,
    MultiFidelitySurrogate,
    BayesianRidgeSurrogate,
    RFFGPSurrogate,
)
from multioutreg.surrogates.rfgp_sklearn import _RFFEstimator

try:
    from ngboost import NGBRegressor as _NGBRegressor
    _NGBOOST_AVAILABLE = True
except ImportError:
    _NGBOOST_AVAILABLE = False

class AutoDetectMultiOutputRegressor(BaseEstimator, RegressorMixin):
    """Fit a separate estimator per output choosing the best via grid search.

    Parameters
    ----------
    estimators : sequence of BaseEstimator
        Candidate sklearn estimators to evaluate.
    param_spaces : sequence of dict
        Parameter grids corresponding to each estimator.
    cv : int
        Number of cross-validation folds used in grid search.
    scoring : str
        Sklearn scoring metric for GridSearchCV.
    pre_screen : bool, default False
        When True, run statistical pre-screening tests before fitting each
        output and skip models that are either computationally expensive
        without justification or unlikely to improve on a linear baseline.
        See :class:`~multioutreg.model_selection.screening.ModelScreener`.
    """

    def __init__(
        self,
        estimators: Sequence[BaseEstimator],
        param_spaces: Sequence[dict],
        cv: int = 3,
        scoring: str = "neg_mean_squared_error",
        pre_screen: bool = False,
    ) -> None:
        if len(estimators) != len(param_spaces):
            raise ValueError("Each estimator must have a corresponding param space")
        self.estimators = list(estimators)
        self.param_spaces = list(param_spaces)
        self.cv = cv
        self.scoring = scoring
        self.pre_screen = pre_screen

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AutoDetectMultiOutputRegressor":
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Run statistical pre-screening once across all outputs
        _screener = None
        if self.pre_screen and hasattr(self, "_model_names"):
            from multioutreg.model_selection.screening import ModelScreener
            _screener = ModelScreener().fit(X, y)

        # Pre-evaluate joint multi-output surrogate candidates (models with
        # _multi_output = True that cannot be wrapped in MultiOutputRegressor).
        # These are scored per output so they can compete in the per-output loop.
        _mo_scores: list[tuple] = []  # (per_output_scores, fitted_surrogate)
        for mo_surrogate in getattr(self, "_multi_output_candidates", []):
            mo_surrogate.fit(X, y)
            mo_preds = np.asarray(mo_surrogate.predict(X))
            if mo_preds.ndim == 1:
                mo_preds = mo_preds.reshape(-1, 1)
            # Use neg-MSE to match GridSearchCV default scoring
            per_output_mse = [
                -np.mean((y[:, j] - mo_preds[:, j]) ** 2)
                for j in range(y.shape[1])
            ]
            _mo_scores.append((per_output_mse, mo_surrogate))

        self.models_ = []
        # _model_output_col[i] = which column of models_[i].predict(X) gives output i.
        # Multi-output models store the full (n_samples, n_outputs) matrix and we
        # slice a single column per slot; single-output models always return column 0.
        self._model_output_col: list[int] = []
        for i in range(y.shape[1]):
            # Determine which estimator indices to evaluate for this output
            if _screener is not None:
                eligible = set(_screener.eligible_indices_for_output(
                    i, self._model_names))
            else:
                eligible = set(range(len(self.estimators)))

            best_score = -np.inf
            best_est = None
            best_idx = None
            best_params = None
            for idx, (est, params) in enumerate(zip(self.estimators, self.param_spaces)):
                if idx not in eligible:
                    continue
                gs = GridSearchCV(est, params, cv=self.cv, scoring=self.scoring)
                gs.fit(X, y[:, i])
                if gs.best_score_ > best_score:
                    best_score = gs.best_score_
                    best_est = clone(gs.best_estimator_)
                    best_idx = idx
                    best_params = gs.best_params_

            # Check if any joint multi-output surrogate beats the single-output winner
            best_mo_surrogate = None
            for per_output_scores, mo_surrogate in _mo_scores:
                if per_output_scores[i] > best_score:
                    best_score = per_output_scores[i]
                    best_mo_surrogate = mo_surrogate
                    best_est = None  # signal that multi-output model won

            if best_mo_surrogate is not None:
                self.models_.append(best_mo_surrogate)
                self._model_output_col.append(i)  # column i of the joint prediction
            elif best_est is None:
                raise RuntimeError("No valid estimator found")
            elif hasattr(self, "_surrogate_constructors") and best_idx is not None:
                surrogate = self._surrogate_constructors[best_idx](**best_params)
                if isinstance(surrogate, MultiFidelitySurrogate):
                    surrogate.fit((X, y[:, [i]]))
                else:
                    surrogate.fit(X, y[:, [i]])
                self.models_.append(surrogate)
                self._model_output_col.append(0)  # single-output, always column 0
            else:
                best_est.fit(X, y[:, i])
                self.models_.append(best_est)
                self._model_output_col.append(0)  # single-output, always column 0

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

    def register_multi_output_candidates(
        self, surrogates: list
    ) -> "AutoDetectMultiOutputRegressor":
        """Register joint multi-output surrogate candidates.

        These surrogates predict all outputs simultaneously (i.e. they set
        ``_multi_output = True``). They are evaluated once on the full Y matrix
        during ``fit()`` and each output's score is compared against the
        per-output GridSearchCV winners.

        Parameters
        ----------
        surrogates : list
            Fitted or unfitted surrogate instances with ``fit(X, Y)`` and
            ``predict(X)`` that return an ``(n_samples, n_outputs)`` array.

        Returns
        -------
        self
        """
        self._multi_output_candidates = list(surrogates)
        return self

    def predict(self, X: np.ndarray, return_std: bool = False) -> np.ndarray:
        if not hasattr(self, "models_"):
            raise AttributeError("Estimator not fitted")
        preds = []
        stds = []
        # Cache full prediction matrices keyed by model id so that multi-output
        # models shared across several output slots are only called once.
        # _model_output_col[i] stores which column of models_[i]'s prediction
        # corresponds to output i (set during fit()).
        _cache: dict[int, tuple] = {}  # id(model) -> (pred_2d, std_2d_or_None)
        _model_cols = getattr(self, "_model_output_col", [0] * len(self.models_))
        for model, col in zip(self.models_, _model_cols):
            mid = id(model)
            if mid not in _cache:
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
                if return_std and std is not None:
                    std = np.asarray(std)
                    if std.ndim == 1:
                        std = std.reshape(-1, 1)
                _cache[mid] = (pred, std)

            full_pred, full_std = _cache[mid]
            preds.append(full_pred[:, col : col + 1])
            if return_std:
                stds.append(
                    full_std[:, col : col + 1]
                    if full_std is not None
                    else np.zeros((X.shape[0], 1))
                )

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
        pre_screen: bool = False,
    ) -> "AutoDetectMultiOutputRegressor":
        """Return instance configured to search all vendored surrogates.

        Parameters
        ----------
        pre_screen : bool, default False
            When True, statistical tests are run at the start of ``fit()``
            to skip expensive models that are unlikely to help on this data.
            Computationally heavy models (GP, SVR, MLP, heteroscedastic
            ensembles) are only run when the data characteristics justify them.
        """

        estimators = [
            LinearRegression(),
            GaussianProcessRegressor(),
            RandomForestRegressor(),
            ExtraTreesRegressor(),
            GradientBoostingRegressor(),
            SVR(),
            KNeighborsRegressor(),
            DecisionTreeRegressor(),
            MLPRegressor(max_iter=500),
            BayesianRidge(),
            _RFFEstimator(n_components=500, length_scale=1.0),
        ]

        param_spaces = [
            {},
            {"alpha": [1e-10, 1e-2]},
            {"n_estimators": [50, 100], "max_depth": [3, 5, None]},
            {"n_estimators": [50, 100], "max_depth": [3, 5, None]},
            {"n_estimators": [50, 100], "max_depth": [3, 5]},
            {"C": [1.0, 10.0], "gamma": ["scale", "auto"]},
            {"n_neighbors": [3, 5, 7]},
            {"max_depth": [1, None]},
            {"hidden_layer_sizes": [(64,), (128,)], "alpha": [1e-4, 1e-3]},
            {},
            {"n_components": [100, 500], "length_scale": [0.1, 1.0, 10.0]},
        ]

        model_names = ["linear", "gp", "rf", "et", "gb", "svr", "knn", "dt", "mlp",
                       "bayesian_ridge", "rfgp"]
        surrogate_constructors = [
            LinearRegressionSurrogate,
            GaussianProcessSurrogate,
            RandomForestSurrogate,
            ExtraTreesRegressorSurrogate,
            GradientBoostingSurrogate,
            SVRSurrogate,
            KNeighborsSurrogate,
            DecisionTreeRegressorSurrogate,
            ConformalPredictionNetworkSurrogate,
            BayesianRidgeSurrogate,
            RFFGPSurrogate,
        ]

        if _NGBOOST_AVAILABLE:
            estimators.append(_NGBRegressor(n_estimators=200, verbose=False))
            param_spaces.append({"n_estimators": [100, 200], "learning_rate": [0.01, 0.05]})
            model_names.append("ngboost")
            surrogate_constructors.append(NGBoostSurrogate)

        instance = cls(estimators, param_spaces, cv=cv, scoring=scoring,
                       pre_screen=pre_screen)
        # Short names used by ModelScreener.eligible_indices_for_output()
        instance._model_names = model_names

        if fidelity_levels is None:
            instance._surrogate_constructors = surrogate_constructors
        else:
            def wrap(cls_sur):
                return lambda **p: MultiFidelitySurrogate(
                    lambda: cls_sur(**p), fidelity_levels
                )

            instance._surrogate_constructors = [wrap(c) for c in surrogate_constructors]
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