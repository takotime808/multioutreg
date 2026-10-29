# Copyright (c) 2025 takotime808
"""
Multi-objective regret utilities for evaluation and optimization loops.

Provides:
- pareto_mask(Y, minimize=True)
- hypervolume(Y, reference_point=None, minimize=True): exact, any number of objectives
- hypervolume_regret(Y_true, Y_pred, reference_point=None, minimize=True)
- scalarized_regret(Y_true, Y_pred, weights, minimize=True, reduce="mean")
- epsilon_regret(Y_true, Y_pred, minimize=True)
- front_hypervolume_regret / front_scalarized_regret / front_epsilon_regret
- RegretTracker(...) for iterative workflows (e.g., BO).

Notes
-----
* Assumes objectives are columns in Y (shape: [n_samples, n_obj]).
* Accepts 1-D or 2-D arrays; 1-D is treated as (n, 1).
* By default, everything is oriented to **minimization**. If you have
  maximization objectives, flip their sign before calling, or pass
  minimize=[True/False,...] per objective (broadcasted inside).
* The default regret functions are **decision regret**: the model picks the
  candidates on the Pareto front of Y_pred, and the regret is how much worse
  the *true* objectives at those picks are than the true optimum. Y_true and
  Y_pred must hold the same candidates, row for row. An optimistic model
  cannot score 0 by over-promising.
* The front_* functions compare the predicted front with the true front
  (front fidelity), e.g. HV(true_front) - HV(pred_front), clipped at 0.
* Ref point must be "worse" than all points; for minimization,
  larger coordinate values are worse.
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


def _hv_exact_min(F: np.ndarray, ref: np.ndarray) -> float:
    """
    Exact hypervolume (minimization) in any number of objectives by slicing.

    Points are sorted by the last objective. Between consecutive values of it,
    the dominated region's cross-section is the (m-1)-objective hypervolume of
    the points at or below that value, so the volume is a sum of slab volumes
    and the recursion ends at the exact 2-D sweep. Cost grows as roughly
    n^(m-1), which is fine for the front sizes of surrogate evaluation
    (hundreds of points, a handful of objectives).
    """
    # Points not strictly better than the reference in every objective add nothing.
    F = F[np.all(F < ref, axis=1)]
    if F.shape[0] == 0:
        return 0.0
    m = F.shape[1]
    if m == 1:
        return float(ref[0] - F[:, 0].min())
    if m == 2:
        return _hv_2d_min(F, ref)  # the sweep ignores dominated points by itself

    F = F[pareto_mask(F, True)]
    F = F[np.argsort(F[:, -1], kind="stable")]
    levels = np.append(F[:, -1], ref[-1])
    volume = 0.0
    for k in range(F.shape[0]):
        depth = levels[k + 1] - levels[k]
        if depth > 0:
            volume += _hv_exact_min(F[: k + 1, :-1], ref[:-1]) * depth
    return float(volume)


def _hypervolume_min(F: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Compute hypervolume (minimization) of a front F given reference_point.
    Exact for any number of objectives: uses pymoo's implementation when it is
    installed and there are 3+ objectives, otherwise :func:`_hv_exact_min`.
    """
    F = _ensure_2d(F)
    reference_point = np.asarray(reference_point, dtype=float).reshape(-1)
    m = F.shape[1]

    if m >= 3:
        try:
            from pymoo.indicators.hv import HV  # type: ignore
            inside = F[np.all(F < reference_point, axis=1)]
            if inside.shape[0] == 0:
                return 0.0
            return float(HV(ref_point=reference_point).do(inside))
        except ImportError:
            pass
    return _hv_exact_min(F, reference_point)


def hypervolume(
    Y: np.ndarray,
    reference_point: Optional[np.ndarray] = None,
    minimize: Union[bool, Iterable[bool]] = True,
) -> float:
    """
    Exact hypervolume dominated by the Pareto front of ``Y``, in any number of objectives.

    Parameters
    ----------
    Y : array, shape (n, m) or (n,)
        Objective values; dominated rows are ignored.
    reference_point : array, shape (m,), optional
        Point worse than every point of interest, given in the minimization
        convention: for objectives with ``minimize=False`` pass the *negated*
        coordinate. Points not strictly better than it in every objective add
        nothing. Defaults to 10% of the range beyond the worst value of each
        objective (1.0 beyond it for a constant objective).
    minimize : bool or array-like of bool
        True for objectives to minimize, False for objectives to maximize.

    Returns
    -------
    float
        The hypervolume (length for 1 objective, area for 2, volume for 3+).
    """
    Y_ = _as_minimization(Y, minimize)
    if reference_point is None:
        reference_point = _default_reference_point([Y_])
    reference_point = _ensure_2d(reference_point).reshape(-1)
    if reference_point.size != Y_.shape[1]:
        raise ValueError("reference_point dimensionality must match number of objectives.")
    return _hypervolume_min(Y_[pareto_mask(Y_, True)], reference_point)


# --------------------------- shared regret helpers ---------------------------

def _paired_minimization(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    minimize: Union[bool, Iterable[bool]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Orient both arrays to minimization and check that they hold the same candidates.

    Decision regret evaluates the true objectives at the rows the predictions pick,
    so row ``i`` of ``Y_pred`` must be the prediction for row ``i`` of ``Y_true``.
    """
    Yt = _as_minimization(Y_true, minimize)
    Yp = _as_minimization(Y_pred, minimize)
    if Yt.shape != Yp.shape:
        raise ValueError(
            "Decision regret needs Y_pred to be the predictions for the same candidates as "
            f"Y_true (same shape); got {Yt.shape} and {Yp.shape}. To compare two fronts of "
            "different sizes, use the front_* regret functions."
        )
    if Yt.shape[0] == 0:
        raise ValueError("Decision regret needs at least one candidate.")
    return Yt, Yp


def _check_reference_point(reference_point: np.ndarray, m: int) -> np.ndarray:
    reference_point = _ensure_2d(reference_point).reshape(-1)
    if reference_point.size != m:
        raise ValueError("reference_point dimensionality must match number of objectives.")
    return reference_point


def _normalize_weights(weights: np.ndarray, m: int) -> np.ndarray:
    """
    Return weight vectors of shape (k, m), each clipped to >= 0 and normalized to sum to 1.

    Accepts weights in shapes:
      - (k, m)  -> usual case
      - (m,)    -> one weight vector
      - (k,)    -> only valid when m == 1 (auto-expands to (k, 1))
      - scalar  -> only valid when m == 1 (treated as weight=1.0)
    """
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
    return W / row_sums


def _reduce_regret(r: np.ndarray, reduce: str) -> Union[float, np.ndarray, Dict[str, float]]:
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


def _additive_epsilon(A: np.ndarray, B: np.ndarray) -> float:
    """
    Additive epsilon indicator max_{b∈B} min_{a∈A} max_i (a_i - b_i), clipped at 0.
    Returns inf when either set is empty.
    """
    if A.size == 0 or B.size == 0:
        return float(np.inf)
    eps_list = []
    for b in B:
        val = np.max(A - b[None, :], axis=1)  # (na,)
        eps_list.append(np.min(val))
    return float(max(0.0, np.max(eps_list)))


# --------------------------- decision regret (default) ---------------------------

def hypervolume_regret(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    reference_point: Optional[np.ndarray] = None,
    minimize: Union[bool, Iterable[bool]] = True,
) -> float:
    """
    Decision hypervolume regret: HV lost by acting on the model's picks.

    The model picks the candidates on the Pareto front of ``Y_pred``; the regret is
    ``HV(front(Y_true)) - HV(Y_true[picks])``, i.e. the true objectives are evaluated at
    the designs the model selects. It is always >= 0, and 0 when the picks contain the
    true front. An optimistic model is not rewarded: shifting every prediction by a
    constant leaves the picks, and therefore the regret, unchanged.

    Parameters
    ----------
    Y_true : array, shape (n, m) or (n,)
        True objective values of n candidates.
    Y_pred : array, same shape as ``Y_true``
        Predicted objective values of the same candidates, row for row.
    reference_point : array, shape (m,), optional
        In the minimization convention. Defaults to 10% of the range of ``Y_true``
        beyond its worst value (1.0 beyond it for a constant objective).
    minimize : bool or array-like of bool
        True for objectives to minimize, False for objectives to maximize.

    See Also
    --------
    front_hypervolume_regret : compares the predicted front with the true front.
    """
    Yt, Yp = _paired_minimization(Y_true, Y_pred, minimize)
    if reference_point is None:
        reference_point = _default_reference_point([Yt])
    reference_point = _check_reference_point(reference_point, Yt.shape[1])

    picks = Yt[pareto_mask(Yp, True)]
    hv_t = _hypervolume_min(Yt[pareto_mask(Yt, True)], reference_point)
    hv_p = _hypervolume_min(picks[pareto_mask(picks, True)], reference_point)
    return float(max(0.0, hv_t - hv_p))


def scalarized_regret(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    weights: np.ndarray,
    minimize: Union[bool, Iterable[bool]] = True,
    reduce: str = "mean",
) -> Union[float, np.ndarray, Dict[str, float]]:
    """
    Decision scalarized regret across weight vectors.

    For each weight vector w (nonnegative, sum=1) the model picks the candidate that
    minimizes the predicted ``w·Y_pred``; the regret is how much worse that pick is in
    truth than the true best (minimization):

      r(w) = w·Y_true(argmin_x w·Y_pred(x)) - min_x w·Y_true(x)    (>= 0)

    Ties in the predicted argmin go to the first candidate. ``Y_true`` and ``Y_pred``
    must hold the same candidates, row for row. Weight shapes are those accepted by
    :func:`front_scalarized_regret`.

    Returns aggregated regret by `reduce` or per-weight array.

    See Also
    --------
    front_scalarized_regret : compares the best predicted and true scalarized values.
    """
    Yt, Yp = _paired_minimization(Y_true, Y_pred, minimize)  # (n, m) each
    W = _normalize_weights(weights, Yt.shape[1])           # (k, m)

    # (k, m) @ (m, n) -> (k, n)
    S_true = W @ Yt.T
    picks = np.argmin(W @ Yp.T, axis=1)                    # (k,)
    r = S_true[np.arange(W.shape[0]), picks] - np.min(S_true, axis=1)
    return _reduce_regret(r, reduce)


def epsilon_regret(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    minimize: Union[bool, Iterable[bool]] = True,
) -> float:
    """
    Decision additive epsilon regret under minimization.

    With ``A = Y_true[picks]`` (the true values at the predicted Pareto front) and
    ``B`` the true front, returns ``max_{b∈B} min_{a∈A} max_i (a_i - b_i)``: the
    smallest amount every objective of the picks would have to improve for them to
    weakly dominate the true front. 0.0 means the picks contain the true front.
    ``Y_true`` and ``Y_pred`` must hold the same candidates, row for row.

    See Also
    --------
    front_epsilon_regret : epsilon indicator of the predicted front itself.
    """
    Yt, Yp = _paired_minimization(Y_true, Y_pred, minimize)
    picks = Yt[pareto_mask(Yp, True)]
    return _additive_epsilon(picks, Yt[pareto_mask(Yt, True)])


# --------------------------- front comparison ---------------------------

def front_hypervolume_regret(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    reference_point: Optional[np.ndarray] = None,
    minimize: Union[bool, Iterable[bool]] = True,
) -> float:
    """
    Front hypervolume regret: HV(true_front) - HV(pred_front), clipped at 0.

    Compares the *predicted* front with the true front, so it measures how well the
    predictions reproduce the front, not what acting on them costs. Clipping means an
    optimistic model (predicted front dominating the true one) scores 0; swap the
    arguments to measure optimism. ``Y_true`` and ``Y_pred`` may have different
    numbers of rows.

    Y_true, Y_pred: arrays of shape (n, m) or (n,) with objective values.
    The default reference point is computed from both arrays.
    """
    Yt = _as_minimization(Y_true, minimize)
    Yp = _as_minimization(Y_pred, minimize)

    mt = pareto_mask(Yt, True)
    mp = pareto_mask(Yp, True)
    Ft = Yt[mt]
    Fp = Yp[mp]

    if reference_point is None:
        reference_point = _default_reference_point([Yt, Yp])
    reference_point = _check_reference_point(reference_point, Yt.shape[1])

    hv_t = _hypervolume_min(Ft, reference_point)
    hv_p = _hypervolume_min(Fp, reference_point)
    return float(max(0.0, hv_t - hv_p))


def front_scalarized_regret(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    weights: np.ndarray,
    minimize: Union[bool, Iterable[bool]] = True,
    reduce: str = "mean",
) -> Union[float, np.ndarray, Dict[str, float]]:
    """
    Front scalarized regret across weight vectors.

    For each weight vector w (nonnegative, sum=1), compute:
      r(w) = min_x w·Y_true(x) - min_x w·Y_pred(x)    (minimization)

    Negative when the predictions are pessimistic, positive when optimistic.

    Accepts weights in shapes:
      - (k, m)  -> usual case
      - (m,)    -> one weight vector
      - (k,)    -> only valid when m == 1 (auto-expands to (k, 1))
      - scalar  -> only valid when m == 1 (treated as weight=1.0)

    Returns aggregated regret by `reduce` or per-weight array.
    """
    Yt = _as_minimization(Y_true, minimize)  # (n, m)
    Yp = _as_minimization(Y_pred, minimize)  # (n, m)
    W = _normalize_weights(weights, Yt.shape[1])

    # Compute best scalarized values
    # (k, m) @ (m, n) -> (k, n), then min over points -> (k,)
    s_true = np.min(W @ Yt.T, axis=1)   # (k,)
    s_pred = np.min(W @ Yp.T, axis=1)   # (k,)
    r = s_true - s_pred                 # (k,)
    return _reduce_regret(r, reduce)


def front_epsilon_regret(
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
    return _additive_epsilon(Fp, Ft)


# --------------------------- iterative tracker ---------------------------

@dataclass
class RegretTracker:
    """
    Track decision regret over iterations of an optimization loop.

    Each step records :func:`hypervolume_regret`, :func:`scalarized_regret` (mean over
    the weight vectors) and :func:`epsilon_regret`, so ``Y_true`` and ``Y_pred`` must
    hold the same candidates, row for row.

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
            ref = _default_reference_point([_as_minimization(Y_true, self.minimize)])

        hv = hypervolume_regret(Y_true, Y_pred, reference_point=ref, minimize=self.minimize)
        sc = scalarized_regret(Y_true, Y_pred, weights=self.weights, minimize=self.minimize, reduce="mean")
        ep = epsilon_regret(Y_true, Y_pred, minimize=self.minimize)

        self.history["hv"].append(hv)
        self.history["scalar"].append(float(sc))
        self.history["eps"].append(ep)
        return {"hv": hv, "scalar": float(sc), "eps": ep}
