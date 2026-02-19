# Copyright (c) 2026 takotime808

"""PyTorch network definitions for BNNSurrogate (MC Dropout variant)."""

from multioutreg.surrogates._torch_utils import require_torch

require_torch()

import torch
import torch.nn as nn


class MCDropoutNet(nn.Module):
    """Fully-connected network with dropout kept active at inference time.

    Running N stochastic forward passes with dropout active approximates
    the posterior predictive distribution (MC Dropout, Gal & Ghahramani 2016).

    Parameters
    ----------
    n_features : int
    n_outputs : int
    hidden_layer_sizes : tuple[int, ...]
    dropout_p : float
        Dropout probability applied after every hidden layer.
    """

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        hidden_layer_sizes: tuple = (128, 64),
        dropout_p: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_layer_sizes:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def enable_dropout(self):
        """Set all Dropout layers to training mode (active at inference)."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
