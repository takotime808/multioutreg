# Copyright (c) 2025 takotime808

import numpy as np
from typing import Tuple, Union
from sklearn.base import BaseEstimator, RegressorMixin, clone


class PerTargetRegressorWithStd(BaseEstimator, RegressorMixin):
    """
    A wrapper for fitting separate regressors for each target in a multi-output regression task,
    with optional support for uncertainty estimates via `return_std`.

    This class fits one estimator per output column in the target matrix `y`. During prediction,
    it aggregates the outputs from each fitted estimator and optionally returns standard deviations
    if supported by the base models.

    This is useful for modeling heterogeneous outputs with different estimators or when some
    models support predictive uncertainty.

    Parameters
    ----------
    estimators : list of estimators
        A list of sklearn-compatible regressors, one for each target output.
        Each estimator must implement `.fit()` and `.predict()`. If `return_std=True` is used
        during prediction, the estimator should support `predict(X, return_std=True)` or will fallback
        to NaN standard deviations.

    Attributes
    ----------
    estimators_ : list of estimators
        The list of fitted regressors after calling `.fit()`.

    Methods
    -------
    fit(X, y)
        Fits each estimator to its corresponding target column in `y`.
        If `y` has shape (n_samples,), it is reshaped to (n_samples, 1).

    predict(X, return_std=False)
        Predicts using each fitted estimator. If `return_std=True`, also returns
        per-target standard deviation estimates when supported, otherwise returns NaNs.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> model = PerTargetRegressorWithStd([LinearRegression(), LinearRegression()])
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100, 2)
    >>> model.fit(X, y)
    >>> y_pred, y_std = model.predict(X, return_std=True)
    """
    def __init__(self, estimators):
        self.estimators = list(estimators)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PerTargetRegressorWithStd":
        """
        Fit separate estimators for each output column.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Multi-output target matrix.

        Returns
        -------
        PerTargetRegressorWithStd
            The fitted model.
        """
        self.estimators_ = []
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        for est, col in zip(self.estimators, y.T):
            est_fitted = clone(est)
            est_fitted.fit(X, col)
            self.estimators_.append(est_fitted)
        return self

    def predict(
            self,
            X: np.ndarray,
            return_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict using individual estimators for each target dimension.

        Parameters
        ----------
        X : np.ndarray
            Input feature array.
        return_std : bool, optional
            Whether to return standard deviation.

        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            Mean predictions or (mean, std) tuple.
        """
        preds, stds = [], []
        for est in self.estimators_:
            if return_std:
                try:
                    pred, std = est.predict(X, return_std=True)
                except TypeError:
                    pred = est.predict(X)
                    std = np.full(pred.shape, np.nan)
                preds.append(pred.reshape(-1, 1))
                stds.append(std.reshape(-1, 1))
            else:
                pred = est.predict(X)
                preds.append(np.asarray(pred).reshape(-1, 1))
        if return_std:
            return np.hstack(preds), np.hstack(stds)
        return np.hstack(preds)