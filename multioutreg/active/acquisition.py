# Copyright (c) 2026 takotime808
"""Acquisition functions for active learning on multi-output surrogates.

Every acquisition takes ``mean`` and ``std`` arrays of shape
``(n_candidates, n_outputs)`` and returns a score of shape ``(n_candidates,)``
where larger is better. See ``docs/active_learning.md`` for the design.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
from scipy.stats import norm


def _as_2d(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(-1, 1) if a.ndim == 1 else a


# ---------------------------------------------------------------------------
# Uncertainty extraction
# ---------------------------------------------------------------------------


class CommitteeSurrogate:
    """Query-by-committee uncertainty for any surrogate.

    Fits ``n_members`` copies of ``model_factory()`` on bootstrap resamples and
    predicts the member mean, with the spread across members as ``std``. This
    lets models without native uncertainty (KNN, linear, plain trees) take part
    in variance-based active learning (Burbidge et al. 2007).

    Parameters
    ----------
    model_factory : callable
        Zero-argument callable returning an unfitted surrogate with
        ``fit(X, Y)`` and ``predict(X)``.
    n_members : int, default 5
    random_state : int or None, default 0
    """

    def __init__(self, model_factory: Callable, n_members: int = 5, random_state=0):
        if n_members < 2:
            raise ValueError("A committee needs at least 2 members.")
        self.model_factory = model_factory
        self.n_members = n_members
        self.random_state = random_state

    def fit(self, X, Y) -> "CommitteeSurrogate":
        X = np.asarray(X, dtype=np.float64)
        Y = _as_2d(Y)
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        self.members_ = []
        for _ in range(self.n_members):
            idx = rng.integers(0, n, size=n)
            self.members_.append(self.model_factory().fit(X[idx], Y[idx]))
        return self

    def predict(self, X, return_std: bool = False):
        preds = np.stack([_as_2d(m.predict(X)) for m in self.members_])
        mean = preds.mean(axis=0)
        if return_std:
            return mean, preds.std(axis=0, ddof=1)
        return mean


def predict_mean_std(model, X, allow_zero_std: bool = False, **predict_kwargs):
    """Return ``(mean, std)`` as 2-D arrays, refusing silent zero uncertainty.

    ``BaseSurrogate`` and ``AutoDetectMultiOutputRegressor`` fill ``std`` with
    zeros for outputs whose model has no native uncertainty. A variance-based
    acquisition would read that as "perfectly known" and never sample for that
    output, so this raises unless ``allow_zero_std`` is set.
    """
    mean, std = model.predict(X, return_std=True, **predict_kwargs)
    mean, std = _as_2d(mean), _as_2d(std)
    if not allow_zero_std and X.shape[0] > 1:
        dead = np.flatnonzero(np.all(std == 0, axis=0))
        if dead.size:
            raise ValueError(
                f"Model returned std == 0 everywhere for output(s) {dead.tolist()}; "
                "it has no native uncertainty. Wrap it in CommitteeSurrogate, or "
                "pass allow_zero_std=True."
            )
    return mean, std


# ---------------------------------------------------------------------------
# Output standardization and aggregation
# ---------------------------------------------------------------------------


def output_scale(Y) -> np.ndarray:
    """Per-output standard deviation of observed values (1 where degenerate)."""
    s = _as_2d(Y).std(axis=0)
    return np.where(s > 0, s, 1.0)


def aggregate(values, how: str = "sum", weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """Combine per-output scores ``(n, n_outputs)`` into one score ``(n,)``."""
    values = _as_2d(values)
    if weights is not None:
        values = values * np.asarray(weights, dtype=np.float64)
    if how == "sum":
        return values.sum(axis=1)
    if how == "mean":
        return values.mean(axis=1)
    if how == "max":
        return values.max(axis=1)
    raise ValueError(f"Unknown aggregation {how!r}; use 'sum', 'mean' or 'max'.")


def _scalarize(mean, std, weights):
    """Weighted sum of outputs; std assumes independent outputs."""
    mean, std = _as_2d(mean), _as_2d(std)
    w = np.ones(mean.shape[1]) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (mean.shape[1],):
        raise ValueError(f"weights must have length {mean.shape[1]}, got {w.shape}.")
    return mean @ w, np.sqrt((std ** 2) @ (w ** 2))


# ---------------------------------------------------------------------------
# Acquisition functions
# ---------------------------------------------------------------------------


def max_variance(mean, std, y_scale=None, how: str = "sum", weights=None) -> np.ndarray:
    """Standardized predictive variance, aggregated over outputs (ALM).

    ``y_scale`` (per-output std of the training targets, see
    :func:`output_scale`) keeps large-scale outputs from dominating.
    """
    std = _as_2d(std)
    if y_scale is not None:
        std = std / np.asarray(y_scale, dtype=np.float64)
    return aggregate(std ** 2, how=how, weights=weights)


def upper_confidence_bound(mean, std, beta: float = 4.0, weights=None, minimize: bool = True) -> np.ndarray:
    """GP-UCB (Srinivas et al. 2010) on the scalarized objective.

    With ``minimize=True`` this is the negated lower confidence bound, so that
    larger is still better.
    """
    if beta < 0:
        raise ValueError(f"beta must be >= 0, got {beta}.")
    mu, sigma = _scalarize(mean, std, weights)
    sign = -1.0 if minimize else 1.0
    return sign * mu + np.sqrt(beta) * sigma


def expected_improvement(mean, std, best: float, xi: float = 0.0, weights=None, minimize: bool = True) -> np.ndarray:
    """Expected improvement over ``best`` (Jones et al. 1998)."""
    mu, sigma = _scalarize(mean, std, weights)
    improvement = (best - mu - xi) if minimize else (mu - best - xi)
    out = np.maximum(improvement, 0.0)
    pos = sigma > 0
    z = improvement[pos] / sigma[pos]
    out[pos] = improvement[pos] * norm.cdf(z) + sigma[pos] * norm.pdf(z)
    return out


def probability_of_improvement(mean, std, best: float, xi: float = 0.0, weights=None, minimize: bool = True) -> np.ndarray:
    """Probability of improving on ``best`` by at least ``xi`` (Kushner 1964)."""
    mu, sigma = _scalarize(mean, std, weights)
    improvement = (best - mu - xi) if minimize else (mu - best - xi)
    out = (improvement > 0).astype(np.float64)
    pos = sigma > 0
    out[pos] = norm.cdf(improvement[pos] / sigma[pos])
    return out


ACQUISITIONS = {
    "max_variance": max_variance,
    "ucb": upper_confidence_bound,
    "ei": expected_improvement,
    "pi": probability_of_improvement,
}
