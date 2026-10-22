# Copyright (c) 2026 takotime808

"""Neural-network surrogate with split-conformal uncertainty estimates."""

import numpy as np
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from multioutreg.surrogates.base_sklearn import BaseSurrogate


class ConformalPredictionNetworkSurrogate(BaseSurrogate):
    """MLP surrogate with calibrated residual-based predictive uncertainty.

    Inputs are standardised by default (``scale_inputs=True``); the MLP's
    optimisation and L2 penalty otherwise depend on the inputs' units.
    """

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        alpha=0.0001,
        max_iter=500,
        calibration_fraction=0.2,
        conformal_quantile=0.9,
        random_state=None,
        scale_inputs=True,
    ):
        if not 0.0 < calibration_fraction < 1.0:
            raise ValueError("calibration_fraction must be in (0, 1)")
        if not 0.0 < conformal_quantile < 1.0:
            raise ValueError("conformal_quantile must be in (0, 1)")

        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.calibration_fraction = calibration_fraction
        self.conformal_quantile = conformal_quantile
        self.random_state = random_state
        self.scale_inputs = scale_inputs

        self.model = self._make_model()

    def get_params(self, deep=True):
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "calibration_fraction": self.calibration_fraction,
            "conformal_quantile": self.conformal_quantile,
            "random_state": self.random_state,
            "scale_inputs": self.scale_inputs,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self.model = self._make_model()
        return self

    def _make_model(self):
        return self._make_multi(
            MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                alpha=self.alpha,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
        )

    def fit(self, X, Y):
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        X_train, X_cal, Y_train, Y_cal = train_test_split(
            X,
            Y,
            test_size=self.calibration_fraction,
            random_state=self.random_state,
        )

        self.model.fit(X_train, Y_train)

        cal_preds = np.asarray(self.model.predict(X_cal))
        if cal_preds.ndim == 1:
            cal_preds = cal_preds.reshape(-1, 1)

        residuals = np.abs(Y_cal - cal_preds)
        self._conformal_scale = np.quantile(residuals, self.conformal_quantile, axis=0)
        return self

    def predict(self, X, return_std=False):
        preds = np.asarray(self.model.predict(X))
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        if not return_std:
            return preds

        if not hasattr(self, "_conformal_scale"):
            raise AttributeError("Estimator not fitted")

        # _conformal_scale is the q-quantile of |residual|, i.e. z * sigma for
        # Gaussian residuals with z = Phi^-1((1 + q) / 2); divide by z to get sigma.
        z = norm.ppf(0.5 + self.conformal_quantile / 2)
        std = np.broadcast_to(self._conformal_scale.reshape(1, -1) / z, preds.shape).copy()
        return preds, std

class _CPNEstimator(RegressorMixin, BaseEstimator):
    """Single-output adapter so GridSearchCV scores the surrogate that is deployed.

    Scoring a plain ``MLPRegressor`` trained on 100% of each fold, then deploying
    a :class:`ConformalPredictionNetworkSurrogate` that trains on 80% (the rest
    calibrates), compared a different model from the one used.
    """

    def __init__(self, hidden_layer_sizes=(100,), alpha=0.0001, max_iter=500,
                 random_state=0):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        self.surrogate_ = ConformalPredictionNetworkSurrogate(
            hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
            max_iter=self.max_iter, random_state=self.random_state,
        ).fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self.surrogate_.predict(X)).ravel()
