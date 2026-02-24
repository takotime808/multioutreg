# Copyright (c) 2026 takotime808

"""Shared PyTorch training utilities for surrogate models."""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for this surrogate. "
            "Install it with: pip install torch"
        )


class TorchStandardScaler:
    """StandardScaler for numpy arrays used before/after PyTorch forward passes."""

    def fit(self, X: np.ndarray) -> "TorchStandardScaler":
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.scale_ + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def train_loop(
    model: "nn.Module",
    optimizer: "torch.optim.Optimizer",
    loss_fn,
    X_tensor: "torch.Tensor",
    Y_tensor: "torch.Tensor",
    max_epochs: int = 500,
    batch_size: int = 64,
    patience: int = 20,
    min_delta: float = 1e-6,
    lr_scheduler: "torch.optim.lr_scheduler._LRScheduler | None" = None,
) -> list[float]:
    """Standard mini-batch training loop with early stopping.

    Parameters
    ----------
    model : nn.Module
    optimizer : torch.optim.Optimizer
    loss_fn : callable
        Called as ``loss_fn(y_pred, y_true)`` and returns a scalar tensor.
    X_tensor, Y_tensor : torch.Tensor
        Full training data (will be batched internally).
    max_epochs : int
    batch_size : int
    patience : int
        Number of epochs without improvement before stopping.
    min_delta : float
        Minimum absolute improvement to reset patience counter.
    lr_scheduler : optional
        Called as ``lr_scheduler.step()`` after each epoch.

    Returns
    -------
    losses : list[float]
        Training loss per epoch.
    """
    require_torch()
    n = X_tensor.shape[0]
    losses = []
    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            X_batch = X_tensor[idx]
            Y_batch = Y_tensor[idx]
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, Y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= max(n_batches, 1)
        losses.append(epoch_loss)

        if lr_scheduler is not None:
            lr_scheduler.step()

        if best_loss - epoch_loss > min_delta:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    return losses


def to_tensor(X: np.ndarray, dtype=None) -> "torch.Tensor":
    """Convert numpy array to float32 tensor."""
    require_torch()
    if dtype is None:
        dtype = torch.float32
    return torch.tensor(X, dtype=dtype)
