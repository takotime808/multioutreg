# Copyright (c) 2026 takotime808

"""Deep Ensemble surrogate — gold-standard UQ via N independent networks."""

from __future__ import annotations

import numpy as np

from multioutreg.surrogates._torch_utils import (
    TorchStandardScaler,
    gaussian_nll_loss,
    mixture_mean_std,
    require_torch,
    split_mean_var,
    to_tensor,
    train_loop,
)

_LOSSES = ("nll", "mse")
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
    random seeds.

    With ``loss="nll"`` (default) each network has a mean and a variance head
    and is trained on the Gaussian negative log-likelihood, as in the paper.
    The prediction is the equal-weight mixture of the members: predictive
    variance = mean of the member noise variances (aleatoric) + variance of
    the member means (epistemic).

    With ``loss="mse"`` the networks predict the mean only and the predictive
    std is the cross-network spread. That is epistemic only: it ignores the
    noise in y, so its intervals under-cover on noisy data (often far below
    nominal). Use it only for noise-free data or wrap it with conformal
    prediction.

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
    loss : {"nll", "mse"}, default "nll"
        ``"nll"`` learns the noise variance (aleatoric + epistemic std);
        ``"mse"`` gives the epistemic-only ensemble spread.
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
        loss: str = "nll",
    ):
        require_torch()
        if loss not in _LOSSES:
            raise ValueError(f"loss must be one of {_LOSSES}, got {loss!r}")
        self.loss = loss
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

        if self.loss == "nll":
            n_heads = 2 * self.n_outputs_
            loss_fn = gaussian_nll_loss(self.n_outputs_)
        else:
            n_heads = self.n_outputs_
            loss_fn = nn.MSELoss()

        self.networks_: list[nn.Module] = []
        for i in range(self.n_estimators):
            seed = (self.random_state + i) if self.random_state is not None else None
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)

            net = _build_network(X.shape[1], n_heads, self.hidden_layer_sizes)
            optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
            train_loop(
                model=net,
                optimizer=optimizer,
                loss_fn=loss_fn,
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
            If True, also return the predictive std (see the class docstring).

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

        nll = getattr(self, "loss", "mse") == "nll"
        member_means, member_vars = [], []
        with torch.no_grad():
            for net in self.networks_:
                out = net(X_t)
                if nll:
                    mu_s, var_s = split_mean_var(out, self.n_outputs_)
                    member_vars.append(var_s.numpy() * self.y_scaler_.scale_ ** 2)
                else:
                    mu_s = out
                member_means.append(self.y_scaler_.inverse_transform(mu_s.numpy()))

        # stack: (n_estimators, n_samples, n_outputs)
        means = np.stack(member_means, axis=0)
        if nll:
            y_pred, y_std = mixture_mean_std(means, np.stack(member_vars, axis=0))
        else:
            y_pred, y_std = means.mean(axis=0), means.std(axis=0)

        if not return_std:
            return y_pred
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
            "loss": self.loss,
        }

    def set_params(self, **params) -> "DeepEnsembleSurrogate":
        for key, value in params.items():
            setattr(self, key, value)
        return self
