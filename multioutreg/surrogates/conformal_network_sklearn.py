# Copyright (c) 2026 takotime808

"""Neural-network surrogate with split-conformal uncertainty estimates."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from multioutreg.surrogates.base_sklearn import BaseSurrogate


class ConformalPredictionNetworkSurrogate(BaseSurrogate):
    """MLP surrogate with calibrated residual-based predictive uncertainty."""

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        alpha=0.0001,
        max_iter=500,
        calibration_fraction=0.2,
        conformal_quantile=0.9,
        random_state=None,
        **kwargs,
    ):
        if not 0.0 < calibration_fraction < 1.0:
            raise ValueError("calibration_fraction must be in (0, 1)")
        if not 0.0 < conformal_quantile < 1.0:
            raise ValueError("conformal_quantile must be in (0, 1)")

        self.calibration_fraction = calibration_fraction
        self.conformal_quantile = conformal_quantile
        self.random_state = random_state

        super().__init__(
            MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=alpha,
                max_iter=max_iter,
                random_state=random_state,
                **kwargs,
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

        std = np.broadcast_to(self._conformal_scale.reshape(1, -1), preds.shape).copy()
        return preds, std