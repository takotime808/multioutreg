# Copyright (c) 2026 takotime808
"""Diverse batch selection from scored candidate pools."""

from __future__ import annotations

from typing import Optional

import numpy as np


def default_length_scale(n_points: int, n_features: int) -> float:
    """Typical spacing of a space-filling design of ``n_points`` in the unit cube."""
    return float(max(n_points, 1) ** (-1.0 / n_features))


def select_batch(
    U_cand: np.ndarray,
    scores: np.ndarray,
    batch_size: int,
    U_train: Optional[np.ndarray] = None,
    length_scale: Optional[float] = None,
    penalize_train: bool = False,
) -> np.ndarray:
    """Greedy top-k with a Gaussian distance penalty around picked points.

    After each pick, every candidate's score is multiplied by
    ``1 - exp(-d^2 / (2 l^2))`` with ``d`` its distance to the pick, so a batch
    spreads across the acquisition surface instead of clustering on one peak
    (local penalization, González et al. 2016).

    Parameters
    ----------
    U_cand : array, shape (n_candidates, n_features)
        Candidates in the unit cube (distances are measured there).
    scores : array, shape (n_candidates,)
        Acquisition values; larger is better. Shifted to be non-negative.
    batch_size : int
    U_train : array, optional
        Existing design in the unit cube; sets the default length scale and,
        with ``penalize_train``, is penalized like a pick.
    length_scale : float, optional
        Defaults to :func:`default_length_scale` of ``n_train + batch_size``.

    Returns
    -------
    np.ndarray of int, shape (batch_size,)
        Indices into ``U_cand``.
    """
    U_cand = np.asarray(U_cand, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n, d = U_cand.shape
    if batch_size < 0:
        raise ValueError(f"batch_size must be >= 0, got {batch_size}.")
    if batch_size > n:
        raise ValueError(f"batch_size={batch_size} exceeds the {n} candidates.")
    n_train = 0 if U_train is None else len(U_train)
    if length_scale is None:
        length_scale = default_length_scale(n_train + batch_size, d)

    s = scores - np.nanmin(scores)
    s = np.nan_to_num(s, nan=0.0)
    # Ties (e.g. all-zero EI far from data) still need distinct, spread picks.
    s = s + 1e-12 * (s.max() + 1.0)

    def penalty(points):
        d2 = ((U_cand[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
        return np.prod(1.0 - np.exp(-d2 / (2.0 * length_scale ** 2)), axis=1)

    if penalize_train and n_train:
        s = s * penalty(np.asarray(U_train, dtype=np.float64))

    available = np.ones(n, dtype=bool)
    chosen = []
    for _ in range(batch_size):
        i = int(np.argmax(np.where(available, s, -np.inf)))
        chosen.append(i)
        available[i] = False
        s = s * penalty(U_cand[i : i + 1])
    return np.asarray(chosen, dtype=int)
