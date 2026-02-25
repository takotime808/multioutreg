# Copyright (c) 2026 takotime808

"""Sparse GP surrogate (SGPR) — optional dependency ``pip install gpytorch``."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from multioutreg.surrogates.conformal_mixin import ConformalMixin

try:
    import torch
    import gpytorch
    _GPYTORCH_AVAILABLE = True
except ImportError:
    _GPYTORCH_AVAILABLE = False


def _require_gpytorch() -> None:
    if not _GPYTORCH_AVAILABLE:
        raise ImportError(
            "gpytorch is required for SparseGPSurrogate. "
            "Install it with: pip install gpytorch"
        )


class _SGPRModel(  # type: ignore[misc]
    gpytorch.models.ExactGP if _GPYTORCH_AVAILABLE else object  # type: ignore[misc]
):
    """Single-output sparse GP with inducing points (Titsias 2009, SGPR)."""

    def __init__(self, train_x, train_y, likelihood, inducing_points):
        if not _GPYTORCH_AVAILABLE:
            raise RuntimeError("gpytorch not available")
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.InducingPointKernel(
            base_kernel,
            inducing_points=inducing_points,
            likelihood=likelihood,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _select_inducing_points(X: np.ndarray, n_inducing: int) -> np.ndarray:
    """Select inducing points via random subsampling (without replacement)."""
    n = X.shape[0]
    idx = np.random.choice(n, size=min(n_inducing, n), replace=False)
    return X[idx]


class SparseGPSurrogate(ConformalMixin):
    """Sparse Gaussian Process surrogate with inducing points (SGPR).

    Extends exact GP inference to larger datasets using a set of ``n_inducing``
    inducing points (Titsias 2009).  Scales as O(n·m²) where m is the number
    of inducing points, unlocking GP-quality posterior uncertainty in the
    n = 300–10 000 regime where exact GP (O(n³)) becomes infeasible.

    One GP model is trained independently per output column; predictions include
    a calibrated posterior standard deviation.

    Requires the optional ``gpytorch`` package::

        pip install gpytorch

    Parameters
    ----------
    n_inducing : int, default 50
        Number of inducing points.  More points → better approximation but
        O(m²) memory and compute.  Typical range: 20–200.
    max_iter : int, default 100
        Adam optimiser steps for hyperparameter training.
    learning_rate : float, default 0.1
    random_state : int | None, default None
    """

    def __init__(
        self,
        n_inducing: int = 50,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        random_state: int | None = None,
    ):
        _require_gpytorch()
        self.n_inducing = n_inducing
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "SparseGPSurrogate":
        """Fit one SGPR per output on (X, Y).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        Y : np.ndarray, shape (n_samples,) or (n_samples, n_outputs)

        Returns
        -------
        self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # Normalise inputs for stable kernel training
        self._x_mean = X.mean(axis=0)
        self._x_std = np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
        X_s = (X - self._x_mean) / self._x_std

        self.n_outputs_ = Y.shape[1]
        self.estimators_: list[tuple] = []  # (model, likelihood) per output

        for j in range(self.n_outputs_):
            y_j = torch.tensor(Y[:, j], dtype=torch.float32)
            X_t = torch.tensor(X_s, dtype=torch.float32)
            inducing = torch.tensor(
                _select_inducing_points(X_s, self.n_inducing),
                dtype=torch.float32,
            )

            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = _SGPRModel(X_t, y_j, likelihood, inducing)

            model.train()
            likelihood.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            for _ in range(self.max_iter):
                optimizer.zero_grad()
                output = model(X_t)
                loss = -mll(output, y_j)
                loss.backward()
                optimizer.step()

            model.eval()
            likelihood.eval()
            self.estimators_.append((model, likelihood))

        return self

    def predict(
        self, X: np.ndarray, return_std: bool = False
    ) -> "np.ndarray | tuple[np.ndarray, np.ndarray]":
        """Predict outputs for X.

        Parameters
        ----------
        X : np.ndarray
        return_std : bool, default False
            If True, also return posterior standard deviation.

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples, n_outputs)
        y_std  : np.ndarray, shape (n_samples, n_outputs)  [only if return_std]
        """
        if not hasattr(self, "estimators_"):
            raise AttributeError("SparseGPSurrogate is not fitted. Call fit() first.")

        X = np.asarray(X, dtype=np.float32)
        X_s = (X - self._x_mean) / self._x_std
        X_t = torch.tensor(X_s, dtype=torch.float32)

        preds, stds = [], []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for model, likelihood in self.estimators_:
                pred_dist = likelihood(model(X_t))
                preds.append(pred_dist.mean.numpy())
                if return_std:
                    stds.append(pred_dist.variance.clamp_min(0.0).sqrt().numpy())

        y_pred = np.column_stack(preds)
        if not return_std:
            return y_pred
        return y_pred, np.column_stack(stds)

    def _conformal_point_predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        preds = np.asarray(preds)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return preds

    def get_params(self, deep: bool = True) -> dict:
        return {
            "n_inducing": self.n_inducing,
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "SparseGPSurrogate":
        for key, value in params.items():
            setattr(self, key, value)
        return self


class _SGPREstimator(BaseEstimator, RegressorMixin):
    """Single-output SGPR estimator (sklearn-compatible, used by GridSearchCV).

    Thin wrapper so that ``AutoDetectMultiOutputRegressor.with_vendored_surrogates``
    can include SGPR in its per-output grid search without exposing the
    full multi-output ``SparseGPSurrogate`` API.
    """

    def __init__(self, n_inducing: int = 50, max_iter: int = 100, learning_rate: float = 0.1):
        self.n_inducing = n_inducing
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SGPREstimator":
        _require_gpytorch()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        x_mean = X.mean(axis=0)
        x_std = np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
        X_s = (X - x_mean) / x_std
        self._x_mean = x_mean
        self._x_std = x_std

        X_t = torch.tensor(X_s, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        inducing = torch.tensor(
            _select_inducing_points(X_s, self.n_inducing), dtype=torch.float32
        )

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = _SGPRModel(X_t, y_t, likelihood, inducing)
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for _ in range(self.max_iter):
            optimizer.zero_grad()
            out = model(X_t)
            loss = -mll(out, y_t)
            loss.backward()
            optimizer.step()

        model.eval()
        likelihood.eval()
        self._model = model
        self._likelihood = likelihood
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        X_s = (X - self._x_mean) / self._x_std
        X_t = torch.tensor(X_s, dtype=torch.float32)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self._likelihood(self._model(X_t))
        return pred.mean.numpy()
