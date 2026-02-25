# Copyright (c) 2026 takotime808

"""Deep Ensemble surrogate — gold-standard UQ via N independent networks."""

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


def _build_network(n_features: int, n_outputs: int, hidden_layer_sizes: tuple):
    """Build a simple deterministic feed-forward network (no dropout)."""
    require_torch()
    import torch.nn as nn

    layers = []
    in_dim = n_features
    for h in hidden_layer_sizes:
        layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
        in_dim = h
    layers.append(nn.Linear(in_dim, n_outputs))
    return nn.Sequential(*layers)


class DeepEnsembleSurrogate(ConformalMixin):
    """Deep Ensemble surrogate (Lakshminarayanan et al. 2017).

    Trains ``n_estimators`` independent feed-forward networks from different
    random seeds and uses their prediction disagreement as epistemic uncertainty.
    This approach consistently outperforms single-model uncertainty methods
    (MC Dropout, conformal wrappers) on calibration benchmarks.

    Each network is trained with MSE loss.  At inference the ensemble mean is
    the point estimate; predictive std is the cross-network standard deviation.

    Requires PyTorch (core dependency)::

        pip install torch

    Parameters
    ----------
    n_estimators : int, default 5
        Number of independent networks in the ensemble.  5 gives a good
        variance estimate; 10+ improves calibration at extra compute cost.
    hidden_layer_sizes : tuple[int, ...], default (128, 64)
        Width of each hidden layer.
    max_epochs : int, default 500
    learning_rate : float, default 1e-3
    batch_size : int, default 64
    patience : int, default 30
        Early-stopping patience (epochs without improvement).
    random_state : int | None, default None
        Base seed; individual networks use ``random_state + i`` so each
        gets a distinct initialisation.
    """

    _multi_output = True  # signals AutoDetect to evaluate on full Y

    def __init__(
        self,
        n_estimators: int = 5,
        hidden_layer_sizes: tuple = (128, 64),
        max_epochs: int = 500,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        patience: int = 30,
        random_state: int | None = None,
    ):
        require_torch()
        self.n_estimators = n_estimators
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "DeepEnsembleSurrogate":
        """Train all ensemble members on (X, Y).

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

        self.networks_: list[nn.Module] = []
        for i in range(self.n_estimators):
            seed = (self.random_state + i) if self.random_state is not None else None
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)

            net = _build_network(X.shape[1], self.n_outputs_, self.hidden_layer_sizes)
            optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
            train_loop(
                model=net,
                optimizer=optimizer,
                loss_fn=nn.MSELoss(),
                X_tensor=X_t,
                Y_tensor=Y_t,
                max_epochs=self.max_epochs,
                batch_size=self.batch_size,
                patience=self.patience,
            )
            net.eval()
            self.networks_.append(net)

        return self

    def predict(
        self, X: np.ndarray, return_std: bool = False
    ) -> "np.ndarray | tuple[np.ndarray, np.ndarray]":
        """Predict outputs for X.

        Parameters
        ----------
        X : np.ndarray
        return_std : bool, default False
            If True, also return predictive std estimated from ensemble spread.

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples, n_outputs)
        y_std  : np.ndarray, shape (n_samples, n_outputs)  [only if return_std]
        """
        import torch

        if not hasattr(self, "networks_"):
            raise AttributeError(
                "DeepEnsembleSurrogate is not fitted. Call fit() first."
            )

        X = np.asarray(X, dtype=np.float32)
        X_s = self.x_scaler_.transform(X).astype(np.float32)
        X_t = to_tensor(X_s)

        member_preds = []
        with torch.no_grad():
            for net in self.networks_:
                y_s = net(X_t).numpy()
                member_preds.append(self.y_scaler_.inverse_transform(y_s))

        # stack: (n_estimators, n_samples, n_outputs)
        stack = np.stack(member_preds, axis=0)
        y_pred = stack.mean(axis=0)

        if not return_std:
            return y_pred

        y_std = stack.std(axis=0)
        return y_pred, y_std

    def _conformal_point_predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        preds = np.asarray(preds)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return preds

    def get_params(self, deep: bool = True) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "DeepEnsembleSurrogate":
        for key, value in params.items():
            setattr(self, key, value)
        return self
