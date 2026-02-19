# Copyright (c) 2026 takotime808

"""Bayesian Neural Network surrogate using MC Dropout uncertainty."""

from __future__ import annotations

import numpy as np

from multioutreg.surrogates._torch_utils import (
    TorchStandardScaler,
    require_torch,
    to_tensor,
    train_loop,
)
from multioutreg.surrogates.conformal_mixin import ConformalMixin

# _multi_output sentinel: this surrogate predicts all outputs jointly
_MULTI_OUTPUT = True


class BNNSurrogate(ConformalMixin):
    """Bayesian Neural Network surrogate with MC Dropout uncertainty.

    Trains a fully-connected network with dropout and approximates the
    posterior predictive distribution at inference by running ``n_mc_samples``
    stochastic forward passes with dropout active.

    Parameters
    ----------
    hidden_layer_sizes : tuple[int, ...], default (128, 64)
    dropout_p : float, default 0.1
        Dropout probability applied after every hidden layer.
    n_mc_samples : int, default 30
        Number of stochastic forward passes used to estimate predictive std.
        Set to 1 for deterministic point-estimate mode.
    max_epochs : int, default 500
    learning_rate : float, default 1e-3
    batch_size : int, default 64
    patience : int, default 30
        Early-stopping patience (epochs without improvement).
    random_state : int | None, default None
    """

    _multi_output = True  # signals AutoDetect to evaluate on full Y

    def __init__(
        self,
        hidden_layer_sizes: tuple = (128, 64),
        dropout_p: float = 0.1,
        n_mc_samples: int = 30,
        max_epochs: int = 500,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        patience: int = 30,
        random_state: int | None = None,
    ):
        require_torch()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_p = dropout_p
        self.n_mc_samples = n_mc_samples
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "BNNSurrogate":
        """Fit the BNN on training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        Y : np.ndarray, shape (n_samples,) or (n_samples, n_outputs)

        Returns
        -------
        self
        """
        import torch
        import torch.nn as nn
        from multioutreg.surrogates.bnn_network import MCDropoutNet

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        self.n_outputs_ = Y.shape[1]

        self.x_scaler_ = TorchStandardScaler().fit(X)
        self.y_scaler_ = TorchStandardScaler().fit(Y)

        X_s = self.x_scaler_.transform(X).astype(np.float32)
        Y_s = self.y_scaler_.transform(Y).astype(np.float32)

        X_t = to_tensor(X_s)
        Y_t = to_tensor(Y_s)

        self.network_ = MCDropoutNet(
            n_features=X.shape[1],
            n_outputs=self.n_outputs_,
            hidden_layer_sizes=self.hidden_layer_sizes,
            dropout_p=self.dropout_p,
        )

        optimizer = torch.optim.Adam(
            self.network_.parameters(), lr=self.learning_rate
        )
        loss_fn = nn.MSELoss()

        self.training_losses_ = train_loop(
            model=self.network_,
            optimizer=optimizer,
            loss_fn=loss_fn,
            X_tensor=X_t,
            Y_tensor=Y_t,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            patience=self.patience,
        )
        return self

    def predict(
        self, X: np.ndarray, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict outputs for X.

        Parameters
        ----------
        X : np.ndarray
        return_std : bool, default False
            If True, also return predictive std estimated via MC Dropout.

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples, n_outputs)
        y_std : np.ndarray, shape (n_samples, n_outputs)  [only if return_std=True]
        """
        import torch

        if not hasattr(self, "network_"):
            raise AttributeError("BNNSurrogate is not fitted. Call fit() first.")

        X = np.asarray(X, dtype=np.float32)
        X_s = self.x_scaler_.transform(X).astype(np.float32)
        X_t = to_tensor(X_s)

        if not return_std or self.n_mc_samples == 1:
            self.network_.eval()
            with torch.no_grad():
                y_s = self.network_(X_t).numpy()
            y_pred = self.y_scaler_.inverse_transform(y_s)
            if not return_std:
                return y_pred
            return y_pred, np.zeros_like(y_pred)

        # MC Dropout: keep dropout active, run n_mc_samples stochastic passes
        self.network_.eval()
        self.network_.enable_dropout()
        samples = []
        with torch.no_grad():
            for _ in range(self.n_mc_samples):
                y_s = self.network_(X_t).numpy()
                samples.append(self.y_scaler_.inverse_transform(y_s))

        samples = np.stack(samples, axis=0)  # (n_mc_samples, n_samples, n_outputs)
        y_pred = samples.mean(axis=0)
        y_std = samples.std(axis=0)
        return y_pred, y_std

    def _conformal_point_predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        preds = np.asarray(preds)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return preds

    def get_params(self, deep: bool = True) -> dict:
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "dropout_p": self.dropout_p,
            "n_mc_samples": self.n_mc_samples,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "BNNSurrogate":
        for key, value in params.items():
            setattr(self, key, value)
        return self
