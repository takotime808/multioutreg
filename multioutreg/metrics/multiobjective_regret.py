# Copyright (c) 2025 takotime808
"""
Multi-objective regret utilities for evaluation and optimization loops.

Provides:
- pareto_mask(Y, minimize=True)
- hypervolume_regret(Y_true, Y_pred, reference_point=None, minimize=True)
- scalarized_regret(Y_true, Y_pred, weights, minimize=True, reduce="mean")
- epsilon_regret(Y_true, Y_pred, minimize=True)
- RegretTracker(...) for iterative workflows (e.g., BO).

Notes
-----
* Assumes objectives are columns in Y (shape: [n_samples, n_obj]).
* Accepts 1-D or 2-D arrays; 1-D is treated as (n, 1).
* By default, everything is oriented to **minimization**. If you have
  maximization objectives, flip their sign before calling, or pass
  minimize=[True/False,...] per objective (broadcasted inside).
* Hypervolume regret := HV(true_front) - HV(pred_front).
  (Ref point must be "worse" than all points; for minimization,
   larger coordinate values are worse.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple, Union, Dict

import numpy as np


# --------------------------- helpers & core ops ---------------------------

def _ensure_2d(Y: np.ndarray) -> np.ndarray:
    """Coerce Y to shape (n, m). If 1-D, treat as (n, 1)."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        return Y.reshape(-1, 1)
    if Y.ndim != 2:
        raise ValueError("Y must be a 1-D or 2-D array of objective values.")
    return Y


def _as_minimization(Y: np.ndarray, minimize: Union[bool, Iterable[bool]]) -> np.ndarray:
    """Return objectives in minimization convention (flip signs for maximization)."""
    Y = _ensure_2d(Y)
    if isinstance(minimize, bool):
        return Y if minimize else -Y
    minimize = np.asarray(minimize, dtype=bool)
    if minimize.ndim != 1 or minimize.size != Y.shape[1]:
        raise ValueError("`minimize` must be bool or a 1D bool array with size == n_obj.")
    signs = np.where(minimize, 1.0, -1.0)
    return Y * signs


def pareto_mask(Y: np.ndarray, minimize: Union[bool, Iterable[bool]] = True) -> np.ndarray:
    """
    Return a boolean mask for non-dominated points (Pareto set) under minimization convention.

    Parameters
    ----------
    Y : array, shape (n, m) or (n,)
        Objective values.
    minimize : bool or array-like of bool
        True for objectives to minimize, False for maximize.

    Returns
    -------
    mask : array, shape (n,)
        True if Y[i] is Pareto-efficient (non-dominated).
    """
    Y_ = _as_minimization(Y, minimize)  # (n, m)
    n = Y_.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    if n == 1:
        return np.array([True], dtype=bool)

    # Pairwise dominance matrix under minimization:
    # i dominates j if i <= j componentwise and i < j in at least one component
    le = np.all(Y_[:, None, :] <= Y_[None, :, :], axis=-1)   # (n, n)
    lt = np.any(Y_[:, None, :] <  Y_[None, :, :], axis=-1)   # (n, n)
    dominates = le & lt                                       # (n, n)

    # A point j is dominated if there exists any i that dominates j
    dominated = np.any(dominates, axis=0)                     # (n,)
    mask = ~dominated
    return mask


def _default_reference_point(Ys: List[np.ndarray]) -> np.ndarray:
    """
    Pick a safe minimization reference point slightly worse than all given points.
    """
    arrs = [ _ensure_2d(a) for a in Ys ]
    allY = np.vstack(arrs)
    worst = np.max(allY, axis=0)
    span = np.ptp(allY, axis=0)
    # 10% beyond the worst observed (handles near-constant dims too)
    return worst + np.where(span > 0, 0.1 * span, 1.0)


def _hypervolume_1d_min(F: np.ndarray, ref: np.ndarray) -> float:
    """
    Exact 1D hypervolume for minimization.
    F is non-dominated front, shape (k,1). HV is max(0, ref - min(F)).
    """
    if F.size == 0:
        return 0.0
    fmin = float(np.min(F[:, 0]))
    return float(max(0.0, ref[0] - fmin))


def _hv_2d_min(F: np.ndarray, ref: np.ndarray) -> float:
    """
    Exact 2D hypervolume for minimization. Assumes F is Pareto non-dominated.
    Algorithm: sort by x ascending (so y decreases); sum rectangles.
    """
    if F.size == 0:
        return 0.0
    idx = np.argsort(F[:, 0])
    F = F[idx]
    hv = 0.0
    prev_y = ref[1]
    for x, y in F:
        width = max(0.0, ref[0] - x)
        height = max(0.0, prev_y - y)
        hv += width * height
        prev_y = min(prev_y, y)  # ensure monotone decrease
    return float(max(0.0, hv))


def _hypervolume_min(F: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Compute hypervolume (minimization) of a non-dominated front F given reference_point.
    Tries pymoo if available; otherwise uses exact 1D/2D or MC fallback for m>2.
    """
    F = _ensure_2d(F)
    m = F.shape[1]

    # 1D exact
    if m == 1:
        return _hypervolume_1d_min(F, reference_point)

    # Try pymoo for general m
    try:
        from pymoo.indicators.hv import HV  # type: ignore
        return float(HV(ref_point=np.asarray(reference_point, float)).do(F))
    except Exception:
        pass

    # 2D exact
    if m == 2:
        return _hv_2d_min(F, reference_point)

    # MC fallback for m>2 (rough but safe)
    rng = np.random.default_rng(123)
    best = np.min(F, axis=0)
    lows = np.minimum(best, reference_point)
    highs = np.maximum(best, reference_point)
    widths = np.maximum(highs - lows, 1e-8)
    num = 20000
    samples = lows + rng.random((num, m)) * widths
    # A sample is dominated by the front if exists f in F with f <= s (minimization)
    dominated = []
    for s in samples:
        dominated.append(np.any(np.all(F <= s[None, :], axis=1)))
    frac = float(np.mean(dominated))
    vol = float(np.prod(widths)) * frac
    return vol


# --------------------------- regret metrics ---------------------------

def hypervolume_regret(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    reference_point: Optional[np.ndarray] = None,
    minimize: Union[bool, Iterable[bool]] = True,
) -> float:
    """
    Hypervolume regret: HV(true_front) - HV(pred_front)  (>= 0 ideally).

    Y_true, Y_pred: arrays of shape (n, m) or (n,) with objective values.
    """
    Yt = _as_minimization(Y_true, minimize)
    Yp = _as_minimization(Y_pred, minimize)

    mt = pareto_mask(Yt, True)
    mp = pareto_mask(Yp, True)
    Ft = Yt[mt]
    Fp = Yp[mp]

    if reference_point is None:
        reference_point = _default_reference_point([Yt, Yp])
    reference_point = _ensure_2d(reference_point).reshape(-1)
    if reference_point.size != Yt.shape[1]:
        raise ValueError("reference_point dimensionality must match number of objectives.")

    hv_t = _hypervolume_min(Ft, reference_point)
    hv_p = _hypervolume_min(Fp, reference_point)
    return float(max(0.0, hv_t - hv_p))


def scalarized_regret(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    weights: np.ndarray,
    minimize: Union[bool, Iterable[bool]] = True,
    reduce: str = "mean",
) -> Union[float, np.ndarray, Dict[str, float]]:
    """
    Scalarized regret across weight vectors.

    For each weight vector w (nonnegative, sum=1), compute:
      r(w) = min_x w·Y_true(x) - min_x w·Y_pred(x)    (minimization)

    Accepts weights in shapes:
      - (k, m)  -> usual case
      - (m,)    -> one weight vector
      - (k,)    -> only valid when m == 1 (auto-expands to (k, 1))
      - scalar  -> only valid when m == 1 (treated as weight=1.0)

    Returns aggregated regret by `reduce` or per-weight array.
    """
    Yt = _as_minimization(Y_true, minimize)  # (n, m)
    Yp = _as_minimization(Y_pred, minimize)  # (n, m)
    m = Yt.shape[1]

    W = np.asarray(weights, float)

    # Normalize shapes:
    if W.ndim == 0:
        # scalar weight -> only valid for m == 1
        if m != 1:
            raise ValueError(f"Scalar weight given but number of objectives is {m}. "
                             f"Provide weights of shape (k,{m}) or ({m},).")
        W = np.array([[float(W)]], dtype=float)  # (1,1)
    elif W.ndim == 1:
        if m == 1:
            # (k,) -> (k,1)
            W = W.reshape(-1, 1)
        else:
            # (m,) -> (1,m)
            if W.size != m:
                raise ValueError(f"weights has length {W.size} but number of objectives is {m}.")
            W = W.reshape(1, m)
    elif W.ndim == 2:
        # (k,m) expected
        if W.shape[1] != m:
            # Helpful error with suggestion
            raise ValueError(
                f"weights must have second dimension == number of objectives. "
                f"Got weights.shape={W.shape}, objectives m={m}."
            )
    else:
        raise ValueError("weights must be scalar, 1-D, or 2-D array.")

    # Clip and simplex-normalize each weight vector
    W = np.clip(W, 0.0, np.inf)
    row_sums = W.sum(axis=1, keepdims=True)
    # If a row sums to 0 (all zeros), treat as uniform over m objectives
    zero_rows = (row_sums.squeeze() == 0)
    if np.any(zero_rows):
        W[zero_rows, :] = 1.0
        row_sums = W.sum(axis=1, keepdims=True)
    W = W / row_sums

    # Compute best scalarized values
    # (k, m) @ (m, n) -> (k, n), then min over points -> (k,)
    s_true = np.min(W @ Yt.T, axis=1)   # (k,)
    s_pred = np.min(W @ Yp.T, axis=1)   # (k,)
    r = s_true - s_pred                 # (k,)

    if reduce == "none":
        return r
    if reduce == "mean":
        return float(np.mean(r))
    if reduce == "median":
        return float(np.median(r))
    if reduce == "max":
        return float(np.max(r))
    if reduce == "dict":
        return {
            "mean": float(np.mean(r)),
            "median": float(np.median(r)),
            "max": float(np.max(r)),
            "p10": float(np.percentile(r, 10)),
            "p90": float(np.percentile(r, 90)),
        }
    raise ValueError("reduce must be one of {'none','mean','median','max','dict'}.")


def epsilon_regret(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    minimize: Union[bool, Iterable[bool]] = True,
) -> float:
    """
    Additive epsilon indicator I_ε(A, B) under minimization, with A=pred_front, B=true_front.
    Smaller is better; regret is max_{b∈B} min_{a∈A} max_i (a_i - b_i).

    Returns a nonnegative float (0.0 means predicted front weakly dominates B).
    """
    Yt = _as_minimization(Y_true, minimize)
    Yp = _as_minimization(Y_pred, minimize)
    Ft = Yt[pareto_mask(Yt, True)]
    Fp = Yp[pareto_mask(Yp, True)]

    m = Yt.shape[1]
    if m == 1:
        # In 1D: max_b min_a (a - b) = min_a a - min_b b
        eps = float(np.min(Fp[:, 0]) - np.min(Ft[:, 0])) if (Fp.size and Ft.size) else np.inf
        return float(max(0.0, eps))

    # For each b in true front, find best a in predicted front minimizing max_i(a_i - b_i)
    if Fp.size == 0 or Ft.size == 0:
        return float(np.inf)

    eps_list = []
    for b in Ft:
        val = np.max(Fp - b[None, :], axis=1)  # (na,)
        eps_list.append(np.min(val))
    eps = float(np.max(eps_list))
    return max(0.0, eps)


# --------------------------- iterative tracker ---------------------------

@dataclass
class RegretTracker:
    """
    Track regret over iterations of an optimization loop.

    Example
    -------
    tracker = RegretTracker(minimize=[True, True, True])
    for t in range(T):
        # Y_true_batch, Y_pred_batch are (n_t, m)
        tracker.step(Y_true_batch, Y_pred_batch)

    hist = tracker.history  # dict of lists
    """
    minimize: Union[bool, Iterable[bool]] = True
    reference_point: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None           # if None, will auto-sample on first step
    n_weight_samples: int = 16
    history: Dict[str, List[float]] = field(default_factory=lambda: {"hv": [], "scalar": [], "eps": []})

    def _maybe_init_weights(self, m: int):
        if self.weights is None:
            # Sample random weights on the simplex (Dirichlet)
            rng = np.random.default_rng(42)
            self.weights = rng.dirichlet(alpha=np.ones(m), size=self.n_weight_samples)

    def step(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> Dict[str, float]:
        Y_true = _ensure_2d(np.asarray(Y_true, float))
        Y_pred = _ensure_2d(np.asarray(Y_pred, float))
        if Y_true.shape[1] != Y_pred.shape[1]:
            raise ValueError("Y_true and Y_pred must have same number of objectives (columns).")
        self._maybe_init_weights(Y_true.shape[1])

        ref = self.reference_point
        if ref is None:
            ref = _default_reference_point([
                _as_minimization(Y_true, self.minimize),
                _as_minimization(Y_pred, self.minimize)
            ])

        hv = hypervolume_regret(Y_true, Y_pred, reference_point=ref, minimize=self.minimize)
        sc = scalarized_regret(Y_true, Y_pred, weights=self.weights, minimize=self.minimize, reduce="mean")
        ep = epsilon_regret(Y_true, Y_pred, minimize=self.minimize)

        self.history["hv"].append(hv)
        self.history["scalar"].append(float(sc))
        self.history["eps"].append(ep)
        return {"hv": hv, "scalar": float(sc), "eps": ep}
