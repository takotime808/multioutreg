# Copyright (c) 2025 takotime808

"""Infer how a design of experiments was sampled (grid, LHS, Sobol, random).

The observed design is summarised by three statistics, and each is compared
with the same statistics computed on designs simulated by every candidate
method with the same ``n`` and ``d`` (a parametric bootstrap / Monte Carlo
null). This replaces ad-hoc absolute thresholds, which cannot be valid for
every ``n`` and ``d``.

Statistics (on inputs mapped to the unit cube):

- ``strata``: mean over dimensions of the fraction of the ``n`` equal-width
  strata that contain a point. LHS fills all of them; i.i.d. uniform fills
  about ``1 - 1/e`` (0.63).
- ``log_cd``: log10 of the centered L2 discrepancy (``scipy.stats.qmc``), which
  measures joint uniformity. Sobol < LHS < random. Skipped for large ``n``
  (cost is O(n^2 d)).
- ``nn``: mean nearest-neighbour distance. Space-filling designs spread their
  points out more than random ones do.

For each candidate the statistics are combined into one Mahalanobis distance
from the simulated designs' mean (using their covariance), and the Monte Carlo
p-value is the fraction of simulated designs at least as far away. A method is
"consistent" with the data when its p-value is at least ``alpha``; the reported method is the one with the largest p-value,
or ``"Uncertain"`` when no candidate is consistent (e.g. observational data or
non-uniform marginals).

Limitations: the tests assume the design was drawn on a box with uniform
marginals. When bounds are not given they are estimated from the data's
min/max (the simulated designs go through the same transform, so the null is
matched). Designs with other marginals must be mapped to the unit cube first,
e.g. via their CDFs (see ``mor_cli/use_ks_for_sample_bias.py``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

CANDIDATES: Tuple[str, ...] = ("Random", "LHS", "Sobol")
# Centered discrepancy is O(n^2 d); above this many points it is skipped.
MAX_N_DISCREPANCY = 2000


@dataclass
class SamplingDetection:
    """Result of :func:`detect_sampling_method`."""

    method: str
    explanation: str
    statistics: Dict[str, float] = field(default_factory=dict)
    p_values: Dict[str, float] = field(default_factory=dict)
    n_sim: int = 0
    columns_used: Optional[Sequence[int]] = None


def to_unit_cube(X: np.ndarray, bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
    """Map ``X`` to [0, 1]^d.

    With ``bounds=(lower, upper)`` this is a plain affine map. Without bounds the
    sample min/max are placed at ``0.5/n`` and ``1 - 0.5/n``, i.e. the centres of
    the outermost strata, which is where an LHS puts them on average.
    """
    X = np.asarray(X, dtype=float)
    if bounds is not None:
        lo, hi = (np.broadcast_to(np.asarray(b, dtype=float), X.shape[1:]) for b in bounds)
        return np.clip((X - lo) / (hi - lo), 0.0, 1.0)
    n = X.shape[0]
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return (X - lo) / span * (n - 1) / n + 0.5 / n


def design_statistics(U: np.ndarray) -> Dict[str, float]:
    """Statistics of a design already on the unit cube (see module docstring)."""
    n, d = U.shape
    bins = np.minimum((U * n).astype(int), n - 1)
    strata = float(np.mean([np.unique(bins[:, j]).size / n for j in range(d)]))
    stats = {"strata": strata}
    if n <= MAX_N_DISCREPANCY:
        stats["log_cd"] = float(np.log10(max(qmc.discrepancy(U, method="CD"), 1e-300)))
    nn = cKDTree(U).query(U, k=2)[0][:, 1]
    stats["nn"] = float(np.mean(nn))
    return stats


def _simulate(method: str, n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    if method == "Random":
        return rng.random((n, d))
    if method == "LHS":
        return qmc.LatinHypercube(d=d, seed=rng).random(n)
    if method == "Sobol":
        with warnings.catch_warnings():
            # n not a power of 2 loses balance properties; that is what we want to mimic.
            warnings.simplefilter("ignore", UserWarning)
            return qmc.Sobol(d=d, scramble=True, seed=rng).random(n)
    raise ValueError(f"Unknown sampling method: {method}")


def _mc_p_value(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Monte Carlo p-value of a statistic vector against simulated ones.

    Distance is Mahalanobis from the simulated mean; the covariance gets a small
    ridge because some statistics are (nearly) constant under a method, e.g. an
    LHS on known bounds always fills every stratum. Uses the (count + 1) / (B + 1)
    correction, so the smallest attainable p-value is 1 / (B + 1).
    """
    mu = simulated.mean(axis=0)
    cov = np.atleast_2d(np.cov(simulated, rowvar=False))
    scale = np.maximum(np.diag(cov), (1e-3 * np.abs(mu)) ** 2 + 1e-12)
    cov = cov + np.diag(1e-2 * scale)
    prec = np.linalg.inv(cov)

    def dist(v: np.ndarray) -> np.ndarray:
        diff = np.atleast_2d(v) - mu
        return np.einsum("ij,jk,ik->i", diff, prec, diff)

    d_obs = dist(observed)[0]
    d_sim = dist(simulated)
    return float((np.sum(d_sim >= d_obs) + 1) / (simulated.shape[0] + 1))


def grid_structure(X: np.ndarray) -> Tuple[bool, bool, np.ndarray]:
    """Check for grid structure.

    Returns ``(grid_like, full_factorial, levels)``: ``grid_like`` when every
    column takes at most ``n / 2`` distinct values (continuous designs have
    ``n``); ``full_factorial`` when the rows are exactly the Cartesian product
    of those levels.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    levels = np.array([np.unique(X[:, j]).size for j in range(X.shape[1])])
    grid_like = bool(np.all((levels >= 2) & (levels <= n / 2)))
    full = False
    if grid_like and int(np.prod(levels.astype(float))) == n:
        full = np.unique(X, axis=0).shape[0] == n
    return grid_like, bool(full), levels


def detect_sampling_method(
    X: np.ndarray,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    n_sim: int = 199,
    alpha: float = 0.05,
    candidates: Sequence[str] = CANDIDATES,
    random_state: Optional[int] = 0,
) -> SamplingDetection:
    """Infer the sampling method of a design ``X`` (rows = points, columns = inputs).

    Pass only the *input* columns; responses are not part of the design and
    distort every statistic. ``bounds=(lower, upper)`` should be given when the
    design region is known (e.g. ``(0, 1)`` after a probability integral
    transform); otherwise it is estimated from the data.

    The result is a statistical statement, not proof: "consistent with LHS"
    means LHS-simulated designs of the same size look like this one, and with
    few points several methods can be consistent at once.
    """
    if X is None:
        raise ValueError("X is required")
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] < 4 or X.shape[1] == 0:
        raise ValueError(f"Need at least 4 points and 1 column, got shape {X.shape}")
    finite = np.all(np.isfinite(X), axis=1)
    if not finite.all():
        X = X[finite]
    n = X.shape[0]

    grid_like, full, levels = grid_structure(X)
    if grid_like:
        kind = "full-factorial grid" if full else "grid-like design (few repeated levels per input)"
        return SamplingDetection(
            method="Grid",
            explanation=(
                f"Every input takes only {levels.min()}-{levels.max()} distinct values over {n} points: "
                f"{kind} -> Grid"
            ),
            statistics={"levels_min": float(levels.min()), "levels_max": float(levels.max())},
        )

    # Inputs with few repeated levels (discrete / categorical) break the
    # continuous statistics; test the continuous inputs only.
    continuous = np.where(levels > n / 2)[0]
    if continuous.size == 0:
        return SamplingDetection(
            method="Uncertain",
            explanation="No continuous input columns to test -> Uncertain",
        )
    Xc = X[:, continuous]
    if bounds is not None:
        lo, hi = (np.broadcast_to(np.asarray(b, dtype=float), (X.shape[1],))[continuous] for b in bounds)
        cbounds = (lo, hi)
    else:
        cbounds = None
    d = Xc.shape[1]
    observed = design_statistics(to_unit_cube(Xc, cbounds))

    rng = np.random.default_rng(random_state)
    p_values: Dict[str, float] = {}
    for method in candidates:
        sims = []
        for _ in range(n_sim):
            S = _simulate(method, n, d, rng)
            if cbounds is None:
                S = to_unit_cube(S)
            sims.append(design_statistics(S))
        keys = list(observed)
        p_values[method] = _mc_p_value(
            np.array([observed[k] for k in keys]),
            np.array([[s[k] for k in keys] for s in sims]),
        )

    best = max(p_values, key=p_values.get)
    consistent = [m for m in sorted(p_values, key=p_values.get, reverse=True) if p_values[m] >= alpha]
    p_txt = ", ".join(f"{m} p={p_values[m]:.3f}" for m in sorted(p_values, key=p_values.get, reverse=True))
    stat_txt = f"strata filled {observed['strata']:.2f}"
    if "log_cd" in observed:
        stat_txt += f", centered discrepancy {10 ** observed['log_cd']:.2e}"
    note = "" if continuous.size == X.shape[1] else f" (tested {d} of {X.shape[1]} inputs; others are discrete)"

    if not consistent:
        method = "Uncertain"
        explanation = (
            f"No candidate design matches ({p_txt}; {stat_txt}){note}. "
            "Data may be observational or have non-uniform marginals -> Uncertain"
        )
    else:
        method = best
        also = f"; also consistent with {', '.join(consistent[1:])}" if len(consistent) > 1 else ""
        explanation = (
            f"Monte Carlo test vs {n_sim} simulated designs ({p_txt}; {stat_txt}){also}{note} -> {method}"
        )
    return SamplingDetection(
        method=method,
        explanation=explanation,
        statistics=observed,
        p_values=p_values,
        n_sim=n_sim,
        columns_used=continuous.tolist(),
    )
