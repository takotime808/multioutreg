# Copyright (c) 2025 takotime808
"""Multi-fidelity wrapper for surrogate models."""
from typing import Dict, Iterable, Tuple, Union

import numpy as np

from multioutreg.surrogates.base_sklearn import BaseSurrogate


class MultiFidelitySurrogate:
    """Manage surrogates for different fidelity levels."""

    def __init__(self, surrogate_cls, fidelity_levels: Iterable[str]):
        """Initialize the multi-fidelity surrogate.

        Parameters
        ----------
        surrogate_cls : Callable[[], BaseSurrogate]
            Class or callable used to instantiate a surrogate for each fidelity
            level.
        fidelity_levels : Iterable[str]
            Ordered collection of fidelity level names.
        """
        self.fidelity_levels = list(fidelity_levels)
        self.models: Dict[str, BaseSurrogate] = {
            level: surrogate_cls() for level in self.fidelity_levels
        }

    def fit(
        self,
        data: Union[
            Dict[str, Tuple[np.ndarray, np.ndarray]],
            Tuple[np.ndarray, np.ndarray],
        ],
    ) -> "MultiFidelitySurrogate":
        """Fit surrogates for each fidelity level."""
        if isinstance(data, dict):
            for level, (X, Y) in data.items():
                if level not in self.models:
                    raise ValueError(f"Unknown fidelity level: {level}")
                self.models[level].fit(X, Y)
        else:
            if len(self.fidelity_levels) != 1:
                raise ValueError("Data for multiple fidelities must be a dict")
            X, Y = data
            self.models[self.fidelity_levels[0]].fit(X, Y)
        return self

    def predict(
        self,
        X: np.ndarray,
        level: str | None = None,
        return_std: bool = False,
    ):
        """Predict with the surrogate corresponding to ``level``."""
        if level is None:
            level = self.fidelity_levels[0]
        if level not in self.models:
            raise ValueError(f"Unknown fidelity level: {level}")
        return self.models[level].predict(X, return_std=return_std)