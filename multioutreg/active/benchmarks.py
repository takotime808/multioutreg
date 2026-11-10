# Copyright (c) 2026 takotime808
"""Analytic benchmark functions for active learning, with fidelity variants.

Definitions follow the Virtual Library of Simulation Experiments
(Surjanovic & Bingham, https://www.sfu.ca/~ssurjano/) and, for Friedman 1,
Gramacy & Lee, Himmelblau, ZDT, Viennet 2, the heterogeneous and the
multi-fidelity Branin functions, the BenchmarkFcns library
(https://github.com/mazhar-ansari-ardeh/BenchmarkFcns, MIT), ported to NumPy.
Every function takes ``X`` of shape ``(n, d)`` in its own bounds and returns
``(n, n_outputs)``.

Fidelity levels are ordered lowest first: the first level (index 0) is the
cheapest and the last is the high fidelity, as in Kennedy & O'Hagan (2000),
SMT's ``MFK`` and Emukit. Mainini et al. (2022) and BenchmarkFcns number the
other way (``f_1`` / column 0 is the highest fidelity), so reverse their
levels when porting.

:data:`L1_BENCHMARKS` (``"mf1.1"`` ... ``"mf6"``) is the AVT-331 L1
multi-fidelity suite of Mainini et al. (2022, arXiv:2204.07867), written from
the paper and checked against the authors' reference optima
(https://gitlab.com/qudo046/avt-331-l1-benchmarks). Its raw functions keep
the paper's ``level`` numbering (1 is the highest fidelity); the
:class:`Benchmark` objects follow this module's lowest-first order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
from scipy.stats import qmc


def _col(y) -> np.ndarray:
    return np.asarray(y, dtype=np.float64).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Raw functions
# ---------------------------------------------------------------------------


def forrester(X):
    """Forrester et al. (2008): ``(6x - 2)^2 sin(12x - 4)`` on [0, 1]."""
    x = np.asarray(X, dtype=np.float64)[:, 0]
    return _col((6 * x - 2) ** 2 * np.sin(12 * x - 4))


def forrester_lo(X):
    """Low-fidelity Forrester: ``0.5 f(x) + 10 (x - 0.5) - 5``."""
    x = np.asarray(X, dtype=np.float64)[:, :1]
    return 0.5 * forrester(X) + 10 * (x - 0.5) - 5


def branin(X):
    """Branin-Hoo on [-5, 10] x [0, 15]; global minimum 0.397887."""
    X = np.asarray(X, dtype=np.float64)
    x1, x2 = X[:, 0], X[:, 1]
    b, c, t = 5.1 / (4 * np.pi ** 2), 5 / np.pi, 1 / (8 * np.pi)
    return _col((x2 - b * x1 ** 2 + c * x1 - 6) ** 2 + 10 * (1 - t) * np.cos(x1) + 10)


def currin(X):
    """Currin et al. (1988) exponential function on [0, 1]^2."""
    X = np.asarray(X, dtype=np.float64)
    x1, x2 = X[:, 0], X[:, 1]
    with np.errstate(divide="ignore"):
        factor = np.where(x2 > 0, 1 - np.exp(-1 / (2 * np.where(x2 > 0, x2, 1))), 1.0)
    num = 2300 * x1 ** 3 + 1900 * x1 ** 2 + 2092 * x1 + 60
    den = 100 * x1 ** 3 + 500 * x1 ** 2 + 4 * x1 + 20
    return _col(factor * num / den)


def currin_lo(X):
    """Low-fidelity Currin (Xiong et al. 2013): average of four shifted copies."""
    X = np.asarray(X, dtype=np.float64)
    x1, x2 = X[:, 0], X[:, 1]
    h = 0.05
    x2m = np.maximum(0.0, x2 - h)

    def f(a, b):
        return currin(np.column_stack([a, b]))

    return 0.25 * (f(x1 + h, x2 + h) + f(x1 + h, x2m) + f(x1 - h, x2 + h) + f(x1 - h, x2m))


def branin_currin(X):
    """Two-output Branin (rescaled to [0, 1]^2) and Currin, as in BoTorch."""
    X = np.asarray(X, dtype=np.float64)
    U = np.column_stack([15 * X[:, 0] - 5, 15 * X[:, 1]])
    return np.hstack([branin(U), currin(X)])


def ishigami(X, a: float = 7.0, b: float = 0.1):
    """Ishigami & Homma (1990) on [-pi, pi]^3."""
    X = np.asarray(X, dtype=np.float64)
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return _col(np.sin(x1) + a * np.sin(x2) ** 2 + b * x3 ** 4 * np.sin(x1))


_H3_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_H3_A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
_H3_P = 1e-4 * np.array(
    [[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]]
)


def hartmann3(X):
    """Hartmann 3-D on [0, 1]^3; global minimum -3.86278 at (0.1146, 0.5556, 0.8525)."""
    X = np.asarray(X, dtype=np.float64)
    inner = (_H3_A[None] * (X[:, None, :] - _H3_P[None]) ** 2).sum(axis=2)
    return _col(-(np.exp(-inner) * _H3_ALPHA).sum(axis=1))


def park(X):
    """Park (1991) first function on [0, 1]^4 (x1 is kept away from 0)."""
    X = np.asarray(X, dtype=np.float64)
    x1 = np.maximum(X[:, 0], 1e-6)
    x2, x3, x4 = X[:, 1], X[:, 2], X[:, 3]
    term1 = x1 / 2 * (np.sqrt(1 + (x2 + x3 ** 2) * x4 / x1 ** 2) - 1)
    term2 = (x1 + 3 * x4) * np.exp(1 + np.sin(x3))
    return _col(term1 + term2)


def park_lo(X):
    """Low-fidelity Park (Xiong et al. 2013)."""
    X = np.asarray(X, dtype=np.float64)
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return (1 + np.sin(x1)[:, None] / 10) * park(X) + _col(-2 * x1 + x2 ** 2 + x3 ** 2 + 0.5)


def _borehole(X, numerator: float, offset: float):
    X = np.asarray(X, dtype=np.float64)
    rw, r, Tu, Hu, Tl, Hl, L, Kw = X.T
    log_ratio = np.log(r / rw)
    den = log_ratio * (offset + 2 * L * Tu / (log_ratio * rw ** 2 * Kw) + Tu / Tl)
    return _col(numerator * Tu * (Hu - Hl) / den)


def borehole(X):
    """Borehole water flow (Morris et al. 1993), 8 inputs in physical units."""
    return _borehole(X, 2 * np.pi, 1.0)


def borehole_lo(X):
    """Low-fidelity borehole (Xiong et al. 2013)."""
    return _borehole(X, 5.0, 1.5)


def branin_mf_lo(X):
    """Low-fidelity Branin: ``10 f(1.2 x) + 5 x1 + 10 x2``.

    The low level sees stretched inputs, so no scaling of it matches the high
    level; a hard case for additive or scaled corrections.
    """
    X = np.asarray(X, dtype=np.float64)
    return 10 * branin(1.2 * X) + _col(5 * X[:, 0] + 10 * X[:, 1])


def heterogeneous(X):
    """Heterogeneous function (Mainini et al. 2022, eq. 16, D = 1) on [0, 1]: the high level ``f_1``."""
    x = np.asarray(X, dtype=np.float64)[:, 0]
    return _col(np.sin(30 * (x - 0.9) ** 4) * np.cos(2 * (x - 0.9)) + (x - 0.9) / 2)


def heterogeneous_lo(X):
    """Low level ``f_2 = (f_1 - 1 + x) / (1 + 0.25 x)``: the scale between levels varies with x.

    BenchmarkFcns labels these two levels the other way round; Mainini et al.
    obtain the low level from ``f_1`` through this bridge function.
    """
    x = np.asarray(X, dtype=np.float64)[:, :1]
    return (heterogeneous(X) - 1 + x) / (1 + 0.25 * x)


def friedman1(X):
    """Friedman (1991) on [0, 1]^10; only x1..x5 enter, x6..x10 are inert.

    Noise-free; the usual benchmark adds N(0, 1) noise.
    """
    X = np.asarray(X, dtype=np.float64)
    return _col(
        10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]
    )


def gramacy_lee(X):
    """Gramacy & Lee (2012) on [0.5, 2.5]: fast oscillation on the left, smooth on the right."""
    x = np.asarray(X, dtype=np.float64)[:, 0]
    return _col(np.sin(10 * np.pi * x) / (2 * x) + (x - 1) ** 4)


HIMMELBLAU_MINIMA = np.array(
    [[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]]
)


def himmelblau(X):
    """Himmelblau on [-5, 5]^2: four global minima of 0 (``HIMMELBLAU_MINIMA``)."""
    X = np.asarray(X, dtype=np.float64)
    x, y = X[:, 0], X[:, 1]
    return _col((x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2)


def _zdt(X, h):
    X = np.asarray(X, dtype=np.float64)
    if X.shape[1] < 2:
        raise ValueError("ZDT functions need at least 2 input dimensions.")
    f1 = X[:, 0]
    g = 1 + 9 * X[:, 1:].sum(axis=1) / (X.shape[1] - 1)
    return np.column_stack([f1, g * h(f1 / g, f1)])


def zdt1(X):
    """ZDT1 (Zitzler, Deb & Thiele 2000) on [0, 1]^d, two outputs; convex front."""
    return _zdt(X, lambda r, f1: 1 - np.sqrt(r))


def zdt2(X):
    """ZDT2 on [0, 1]^d, two outputs; concave front."""
    return _zdt(X, lambda r, f1: 1 - r ** 2)


def zdt3(X):
    """ZDT3 on [0, 1]^d, two outputs; front in five disconnected pieces."""
    return _zdt(X, lambda r, f1: 1 - np.sqrt(r) - r * np.sin(10 * np.pi * f1))


def zdt_front(name: str, n: int = 500) -> np.ndarray:
    """Points on the true Pareto front of ``"zdt1"``, ``"zdt2"`` or ``"zdt3"``.

    The front is reached at ``x2 = ... = xd = 0`` (so ``g = 1``); for ZDT3 only
    the non-dominated part of that curve is kept, so fewer than ``n`` rows
    come back.
    """
    fns = {"zdt1": zdt1, "zdt2": zdt2, "zdt3": zdt3}
    if name not in fns:
        raise ValueError(f"No analytic front for {name!r}. Available: {sorted(fns)}.")
    X = np.zeros((n, 2))
    X[:, 0] = np.linspace(0, 1, n)
    F = fns[name](X)
    # Sorted by f1, a point is non-dominated iff its f2 is below every earlier f2.
    keep = F[:, 1] < np.minimum.accumulate(np.r_[np.inf, F[:-1, 1]])
    return F[keep]


def dtlz2(X, n_objectives: int = 3):
    """DTLZ2 (Deb et al. 2005) on [0, 1]^d with ``n_objectives`` outputs.

    The first ``m - 1`` inputs place a point on the positive orthant of the
    unit sphere; the rest push it outwards by ``1 + g`` with
    ``g = sum((x_i - 0.5)^2)``, so the true front is the unit sphere
    (``g = 0``) and the outputs are strongly dependent. Needs ``d >= m``.
    """
    X = np.asarray(X, dtype=np.float64)
    m = int(n_objectives)
    if m < 2 or X.shape[1] < m:
        raise ValueError(f"DTLZ2 with {m} objectives needs n_objectives >= 2 and at least {m} inputs.")
    g = ((X[:, m - 1:] - 0.5) ** 2).sum(axis=1)
    theta = np.pi / 2 * X[:, : m - 1]
    cos, sin = np.cos(theta), np.sin(theta)
    F = np.empty((X.shape[0], m))
    for k in range(m):
        # f_k = prod_{i < m-1-k} cos(theta_i), times sin(theta_{m-1-k}) for k > 0.
        f = np.prod(cos[:, : m - 1 - k], axis=1)
        if k > 0:
            f = f * sin[:, m - 1 - k]
        F[:, k] = f
    return (1 + g)[:, None] * F


def dtlz_front(name: str = "dtlz2", n: int = 500, n_objectives: int = 3, seed=0) -> np.ndarray:
    """``n`` random points on the true Pareto front of ``"dtlz2"``: the unit sphere's positive orthant."""
    if name != "dtlz2":
        raise ValueError(f"No analytic front for {name!r}. Available: ['dtlz2'].")
    Z = np.abs(np.random.default_rng(seed).normal(size=(n, int(n_objectives))))
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def viennet2(X):
    """Viennet 2 on [-4, 4]^2: three quadratic outputs of different shape."""
    X = np.asarray(X, dtype=np.float64)
    x1, x2 = X[:, 0], X[:, 1]
    return np.column_stack([
        (x1 - 2) ** 2 / 2 + (x2 + 1) ** 2 / 13 + 3,
        (x1 + x2 - 3) ** 2 / 36 + (-x1 + x2 + 2) ** 2 / 8 - 17,
        (x1 + 2 * x2 - 1) ** 2 / 175 + (2 * x1 - x2) ** 2 / 17 - 13,
    ])


# ---------------------------------------------------------------------------
# AVT-331 L1 multi-fidelity suite (Mainini et al. 2022)
# ---------------------------------------------------------------------------
# ``level`` uses the paper's numbering: 1 is the highest fidelity. Where the
# paper's text and the authors' code (gitlab.com/qudo046/avt-331-l1-benchmarks)
# disagree, these follow the code, which produced the published optima.


def forrester_mf(X, level: int = 1):
    """Four-level Forrester on [0, 1] (MF1.1); ``f_3``, ``f_4`` are affine in ``f_1``."""
    x = np.asarray(X, dtype=np.float64)[:, 0]
    f1 = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
    if level == 1:
        return _col(f1)
    if level == 2:
        return _col((5.5 * x - 2.5) ** 2 * np.sin(12 * x - 4))
    if level == 3:
        return _col(0.75 * f1 + 5 * (x - 0.5) - 2)
    return _col(0.5 * f1 + 10 * (x - 0.5) - 5)


def jump_forrester(X, level: int = 1):
    """Forrester with a +10 jump at x = 0.5 (MF1.2), two levels.

    The low level's offset right of the jump is -7, as in the authors' code;
    the paper's text gives -2.
    """
    x = np.asarray(X, dtype=np.float64)[:, 0]
    right = x > 0.5
    f1 = (6 * x - 2) ** 2 * np.sin(12 * x - 4) + 10 * right
    if level == 1:
        return _col(f1)
    return _col(0.5 * f1 + 10 * (x - 0.5) - np.where(right, 7.0, 5.0))


def rosenbrock_mf(X, level: int = 1):
    """Three-level Rosenbrock on [-2, 2]^D (MF2). ``f_2`` moves the valley; ``f_3`` rescales ``f_1``."""
    X = np.asarray(X, dtype=np.float64)
    a, b = X[:, :-1], X[:, 1:]
    if level == 2:
        return _col((50 * (b - a ** 2) ** 2 + (-2 - a) ** 2).sum(1) - 0.5 * X.sum(1))
    f1 = (100 * (b - a ** 2) ** 2 + (1 - a) ** 2).sum(1)
    if level == 1:
        return _col(f1)
    # The paper's denominator reads 0.25 x_1; the code (used here) sums 0.25 x_i.
    return _col((f1 - 4 - 0.5 * X.sum(1)) / (10 + 0.25 * X.sum(1)))


_RASTRIGIN_PHI = {1: 10000.0, 2: 5000.0, 3: 2500.0}


def rastrigin_mf(X, level: int = 1, theta: float = 0.2):
    """Three-level shifted-rotated Rastrigin on [-0.1, 0.2]^D (MF3); minimum 0 at x = 0.1.

    Only the last two coordinates are rotated, as in the authors' code. Lower
    levels add a resolution error ``a cos^2(w z + b + pi)`` that grows as
    ``phi`` falls.
    """
    X = np.asarray(X, dtype=np.float64)
    D = X.shape[1]
    R = np.eye(D)
    c, s = np.cos(theta), np.sin(theta)
    R[D - 2:, D - 2:] = [[c, -s], [s, c]]
    z = (X - 0.1) @ R.T
    th = 1 - 1e-4 * _RASTRIGIN_PHI[level]
    f = (z ** 2 + 1 - np.cos(10 * np.pi * z)).sum(1)
    return _col(f + (th * np.cos(10 * np.pi * th * z + 0.5 * np.pi * th + np.pi) ** 2).sum(1))


def heterogeneous_mf(X, level: int = 1):
    """Heterogeneous function on [0, 1]^D (MF4), D = 1, 2 or 3, two levels.

    For D = 1 the high level equals :func:`heterogeneous`. For D >= 2 the
    minimum, -0.562712, is the whole face ``x_1 = 0``.
    """
    X = np.asarray(X, dtype=np.float64)
    D, x1 = X.shape[1], X[:, 0]
    if D == 1:
        f1 = np.sin(30 * (x1 - 0.9) ** 4) * np.cos(2 * (x1 - 0.9)) + (x1 - 0.9) / 2
        return _col(f1 if level == 1 else (f1 - 1 + x1) / (1 + 0.25 * x1))
    f1 = np.sin(21 * (x1 - 0.9) ** 4) * np.cos(2 * (x1 - 0.9)) + (x1 - 0.7) / 2
    prod = x1.copy()
    for i in range(2, D + 1):
        prod = prod * X[:, i - 1]
        f1 = f1 + i * X[:, i - 1] ** i * np.sin(prod)
    if level == 1:
        return _col(f1)
    w = np.array([0.25 * i * (1 if i <= 2 else -1) for i in range(1, D + 1)])
    return _col((f1 - 2 + X.sum(1)) / (5 + X @ w))


_SPRING_DT = {1: 0.01, 2: 0.6}


def _spring_x1(m1, m2, k1, k2, dt, tf: float = 6.0):
    """RK4 position of mass 1 at ``tf`` for two masses, three springs (k3 = k1), from x = (1, 0) at rest."""
    y = np.zeros((m1.shape[0], 4))
    y[:, 0] = 1.0

    def rhs(y):
        return np.column_stack([
            y[:, 2], y[:, 3],
            (-(k1 + k2) * y[:, 0] + k2 * y[:, 1]) / m1,
            (k2 * y[:, 0] - (k1 + k2) * y[:, 1]) / m2,
        ])

    for _ in range(int(tf / dt)):
        a = dt * rhs(y)
        b = dt * rhs(y + a / 2)
        c = dt * rhs(y + b / 2)
        d = dt * rhs(y + c)
        y = y + (a + 2 * b + 2 * c + d) / 6
    return _col(y[:, 0])


def springs(X, level: int = 1):
    """Spring stiffnesses ``(k1, k2)`` in [1, 4]^2 with unit masses (MF5.1); RK4 step 0.01 / 0.6."""
    X = np.asarray(X, dtype=np.float64)
    one = np.ones(len(X))
    return _spring_x1(one, one, X[:, 0], X[:, 1], _SPRING_DT[level])


def spring_masses(X, level: int = 1):
    """``(k1, k2, m1, m2)`` in [1, 4]^4 (MF5.2). The minimum -1 is reached at more than one design."""
    X = np.asarray(X, dtype=np.float64)
    return _spring_x1(X[:, 2], X[:, 3], X[:, 0], X[:, 1], _SPRING_DT[level])


def paciorek(X, level: int = 1, rng=None):
    """Paciorek ``sin(1 / (x1 x2))`` on [0.3, 1]^2 (MF6) with Gaussian noise.

    ``level=0`` is the noise-free truth; level 1 adds N(0, 0.0125^2) and level
    2 (the low fidelity, ``- 9 A^2 cos(1 / (x1 x2))`` with A = 0.5) adds
    N(0, 0.075^2).
    """
    X = np.asarray(X, dtype=np.float64)
    p = 1.0 / (X[:, 0] * X[:, 1])
    if level == 0:
        return _col(np.sin(p))
    rng = np.random.default_rng() if rng is None else rng
    if level == 1:
        return _col(np.sin(p) + rng.normal(0, 0.0125, len(p)))
    return _col(np.sin(p) - 2.25 * np.cos(p) + rng.normal(0, 0.075, len(p)))


def _paciorek_minima(n: int = 4001) -> np.ndarray:
    """Points on the curves ``x1 x2 = 2 / ((3 + 4j) pi)``, j = 0, 1, inside [0.3, 1]^2."""
    x1 = np.linspace(0.3, 1.0, n)
    pts = []
    for p in (2 / (3 * np.pi), 2 / (7 * np.pi)):
        x2 = p / x1
        ok = (x2 >= 0.3) & (x2 <= 1.0)
        pts.append(np.column_stack([x1[ok], x2[ok]]))
    return np.vstack(pts)


def _x1_face_distance(U):
    return np.abs(np.atleast_2d(U)[:, 0])


# ---------------------------------------------------------------------------
# Benchmark container
# ---------------------------------------------------------------------------


@dataclass
class Benchmark:
    """An analytic test problem with bounds and, optionally, fidelity levels.

    ``functions`` and ``costs`` are keyed by level, ordered cheapest first
    (index 0 is the lowest fidelity); the last level is the high fidelity.

    The optional fields describe the answer, for scoring an optimizer with
    :meth:`optimum_errors`: ``maximum`` (the largest value of the high level),
    ``argmin`` (known minimizers, one per row, in the problem's coordinates)
    or ``argmin_distance`` (unit-cube distance to an optimal set that is not
    a finite list of points), ``truth`` (the noise-free high level, for noisy
    problems) and ``budget`` (the evaluation budget, in high-fidelity
    equivalents, that the problem's source prescribes).
    """

    name: str
    bounds: np.ndarray
    functions: Dict[str, Callable[[np.ndarray], np.ndarray]]
    costs: Dict[str, float] = field(default_factory=dict)
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    argmin: Optional[np.ndarray] = None
    argmin_distance: Optional[Callable[[np.ndarray], np.ndarray]] = None
    truth: Optional[Callable[[np.ndarray], np.ndarray]] = None
    budget: Optional[float] = None

    def __post_init__(self):
        self.bounds = np.asarray(self.bounds, dtype=np.float64)
        if not self.costs:
            self.costs = {lvl: 1.0 for lvl in self.functions}
        if self.argmin is not None:
            self.argmin = np.atleast_2d(np.asarray(self.argmin, dtype=np.float64))

    @property
    def levels(self):
        return list(self.functions)

    @property
    def high(self) -> str:
        return self.levels[-1]

    @property
    def n_features(self) -> int:
        return self.bounds.shape[0]

    def __call__(self, X, level: Optional[str] = None) -> np.ndarray:
        return self.functions[level or self.high](np.asarray(X, dtype=np.float64))

    def sample(self, n: int, method: str = "sobol", seed=None) -> np.ndarray:
        """Space-filling (``"sobol"``, ``"lhs"``) or ``"random"`` points in bounds."""
        d = self.n_features
        if method == "sobol":
            U = qmc.Sobol(d, scramble=True, seed=seed).random(n)
        elif method == "lhs":
            U = qmc.LatinHypercube(d, seed=seed).random(n)
        elif method == "random":
            U = np.random.default_rng(seed).random((n, d))
        else:
            raise ValueError(f"Unknown sampling method {method!r}.")
        return qmc.scale(U, self.bounds[:, 0], self.bounds[:, 1])

    def true_value(self, X) -> np.ndarray:
        """Noise-free high-fidelity value: ``truth`` if set, else the high level."""
        fn = self.truth or self.functions[self.high]
        return fn(np.asarray(X, dtype=np.float64))

    def optimum_errors(self, x_hat) -> Dict[str, float]:
        """Mainini et al. (2022) errors ``{"Ex", "Ef", "Et"}`` of an estimated minimizer ``x_hat``.

        Needs ``minimum``, ``maximum`` and ``argmin`` or ``argmin_distance``.
        ``Ef`` uses the noise-free high level at ``x_hat``.
        """
        from multioutreg.metrics.optimization_errors import optimum_errors

        if self.minimum is None or self.maximum is None or (self.argmin is None and self.argmin_distance is None):
            raise ValueError(f"Benchmark {self.name!r} does not define its optimum and range.")
        x_hat = np.atleast_2d(np.asarray(x_hat, dtype=np.float64))
        return optimum_errors(
            x_hat, float(self.true_value(x_hat)[0, 0]), self.minimum, self.maximum,
            x_star=self.argmin, bounds=self.bounds, distance=self.argmin_distance,
        )


_LEVEL_NAMES = {2: ["lo", "hi"], 3: ["lo", "mid", "hi"], 4: ["lo", "mid_lo", "mid_hi", "hi"]}

# id: (function, bounds per input, D, budget, costs from f_1 down, f_min, f_max, argmin)
# Values from Tables 2 and 4 of Mainini et al. (2022), at the precision of the
# authors' AVT331_L1Problem_MinMaxValues.csv where it gives one.
_L1_TABLE = {
    "mf1.1": (forrester_mf, (0, 1), 1, 100, [1, 0.5, 0.1, 0.05], -6.020740056, 15.82973195, [[0.757248751]]),
    "mf1.2": (jump_forrester, (0, 1), 1, 100, [1, 0.2], -0.986325406, 25.82973195, [[0.142589189]]),
    "mf2.1": (rosenbrock_mf, (-2, 2), 2, 200, [1, 0.5, 0.1], 0.0, 3609.0, [[1, 1]]),
    "mf2.2": (rosenbrock_mf, (-2, 2), 5, 500, [1, 0.5, 0.1], 0.0, 14436.0, [[1] * 5]),
    "mf2.3": (rosenbrock_mf, (-2, 2), 10, 1000, [1, 0.5, 0.1], 0.0, 32481.0, [[1] * 10]),
    "mf3.1": (rastrigin_mf, (-0.1, 0.2), 2, 200, [1, 6.25e-2, 3.91e-3], 0.0, 4.020040611, [[0.1] * 2]),
    "mf3.2": (rastrigin_mf, (-0.1, 0.2), 5, 500, [1, 6.25e-2, 3.91e-3], 0.0, 10.05010153, [[0.1] * 5]),
    "mf3.3": (rastrigin_mf, (-0.1, 0.2), 10, 1000, [1, 6.25e-2, 3.91e-3], 0.0, 20.10020305, [[0.1] * 10]),
    "mf4.1": (heterogeneous_mf, (0, 1), 1, 100, [1, 0.2], -0.624998901, 0.361513623, [[0.275501812]]),
    "mf4.2": (heterogeneous_mf, (0, 1), 2, 200, [1, 0.2], -0.562712306, 1.835000108, None),
    "mf4.3": (heterogeneous_mf, (0, 1), 3, 300, [1, 0.2], -0.562712306, 4.359413062, None),
    "mf5.1": (springs, (1, 4), 2, 200, [1, 1.67e-2], -1.0, 1.0, [[2.46740108, 2.19324544]]),
    "mf5.2": (spring_masses, (1, 4), 4, 400, [1, 1.67e-2], -1.0, 1.0, [[1, 3.946018148, 4, 3.28627657]]),
    "mf6": (paciorek, (0.3, 1.0), 2, 200, [1, 0.2], -1.0, 1.0, None),
}

L1_BENCHMARKS = tuple(_L1_TABLE)


def _l1_benchmark(name: str, seed=None) -> Benchmark:
    fn, (lo, hi), D, budget, costs, f_min, f_max, argmin = _L1_TABLE[name]
    names = _LEVEL_NAMES[len(costs)]
    # Paper level l (1 = highest) is names[-l]; costs are fractions of one f_1 run.
    if fn is paciorek:
        rng = np.random.default_rng(seed)
        functions = {names[-l]: (lambda X, l=l: paciorek(X, l, rng=rng)) for l in range(len(costs), 0, -1)}
        truth = lambda X: paciorek(X, 0)
        argmin = _paciorek_minima()
    else:
        functions = {names[-l]: (lambda X, l=l: fn(X, l)) for l in range(len(costs), 0, -1)}
        truth = None
    return Benchmark(
        name, [[lo, hi]] * D, functions,
        costs={names[-l]: float(costs[l - 1]) for l in range(len(costs), 0, -1)},
        minimum=f_min, maximum=f_max, argmin=argmin,
        argmin_distance=_x1_face_distance if name in ("mf4.2", "mf4.3") else None,
        truth=truth, budget=float(budget),
    )


_BOREHOLE_BOUNDS = [
    [0.05, 0.15], [100, 50000], [63070, 115600], [990, 1110],
    [63.1, 116], [700, 820], [1120, 1680], [9855, 12045],
]


def get_benchmark(
    name: str, cost_ratio: Optional[float] = None, n_features: Optional[int] = None, seed=None
) -> Benchmark:
    """Return a :class:`Benchmark` by name.

    Multi-fidelity benchmarks have levels ``"lo"`` and ``"hi"`` with costs
    ``1`` and ``cost_ratio`` (default 10). ``n_features`` sets the input
    dimension of the scalable ZDT problems (default 6; the literature uses
    30, which needs far more training points) and of DTLZ2 (default 7, three
    objectives); the other benchmarks have a fixed dimension.

    The names in :data:`L1_BENCHMARKS` (``"mf1.1"`` ... ``"mf6"``) are the
    AVT-331 L1 suite of Mainini et al. (2022), with the paper's own costs (a
    fraction of one high-fidelity run), budget, range and optimum, so
    ``cost_ratio`` does not apply. They have two to four levels, named
    ``"lo"``, (``"mid"`` or ``"mid_lo"``, ``"mid_hi"``) and ``"hi"``; the
    paper's ``f_1`` is ``"hi"``. ``seed`` seeds the noise of ``"mf6"``.
    """
    if name in _L1_TABLE:
        if cost_ratio is not None or n_features is not None:
            raise ValueError(f"Benchmark {name!r} fixes its costs and dimension; cost_ratio and n_features do not apply.")
        return _l1_benchmark(name, seed)
    mf = {"lo": 1.0, "hi": float(10.0 if cost_ratio is None else cost_ratio)}
    scalable = name.startswith(("zdt", "dtlz"))
    if n_features is not None and not scalable:
        raise ValueError(f"Benchmark {name!r} has a fixed input dimension; n_features applies to ZDT and DTLZ only.")
    if n_features is not None:
        d = int(n_features)
    else:
        d = 7 if name.startswith("dtlz") else 6
    if name.startswith("zdt") and d < 2:
        raise ValueError("ZDT benchmarks need n_features >= 2.")
    if name.startswith("dtlz") and d < 3:
        raise ValueError("DTLZ2 (3 objectives) needs n_features >= 3.")
    table = {
        "forrester": lambda: Benchmark("forrester", [[0, 1]], {"lo": forrester_lo, "hi": forrester}, dict(mf), -6.02074),
        "branin": lambda: Benchmark("branin", [[-5, 10], [0, 15]], {"hi": branin}, minimum=0.397887),
        "currin": lambda: Benchmark("currin", [[0, 1], [0, 1]], {"lo": currin_lo, "hi": currin}, dict(mf)),
        "branin_currin": lambda: Benchmark("branin_currin", [[0, 1], [0, 1]], {"hi": branin_currin}),
        "ishigami": lambda: Benchmark("ishigami", [[-np.pi, np.pi]] * 3, {"hi": ishigami}),
        "hartmann3": lambda: Benchmark("hartmann3", [[0, 1]] * 3, {"hi": hartmann3}, minimum=-3.86278),
        "park": lambda: Benchmark("park", [[0, 1]] * 4, {"lo": park_lo, "hi": park}, dict(mf)),
        "borehole": lambda: Benchmark("borehole", _BOREHOLE_BOUNDS, {"lo": borehole_lo, "hi": borehole}, dict(mf)),
        "branin_mf": lambda: Benchmark("branin_mf", [[-5, 10], [0, 15]], {"lo": branin_mf_lo, "hi": branin}, dict(mf), 0.397887),
        "heterogeneous": lambda: Benchmark("heterogeneous", [[0, 1]], {"lo": heterogeneous_lo, "hi": heterogeneous}, dict(mf)),
        "friedman1": lambda: Benchmark("friedman1", [[0, 1]] * 10, {"hi": friedman1}),
        "gramacy_lee": lambda: Benchmark("gramacy_lee", [[0.5, 2.5]], {"hi": gramacy_lee}, minimum=-0.869011),
        "himmelblau": lambda: Benchmark("himmelblau", [[-5, 5]] * 2, {"hi": himmelblau}, minimum=0.0),
        "zdt1": lambda: Benchmark("zdt1", [[0, 1]] * d, {"hi": zdt1}),
        "zdt2": lambda: Benchmark("zdt2", [[0, 1]] * d, {"hi": zdt2}),
        "zdt3": lambda: Benchmark("zdt3", [[0, 1]] * d, {"hi": zdt3}),
        "viennet2": lambda: Benchmark("viennet2", [[-4, 4]] * 2, {"hi": viennet2}),
        "dtlz2": lambda: Benchmark("dtlz2", [[0, 1]] * d, {"hi": dtlz2}),
    }
    if name not in table:
        raise ValueError(f"Unknown benchmark {name!r}. Available: {sorted(table) + list(L1_BENCHMARKS)}.")
    return table[name]()


BENCHMARKS = (
    "forrester", "branin", "currin", "branin_currin",
    "ishigami", "hartmann3", "park", "borehole",
    "branin_mf", "heterogeneous", "friedman1", "gramacy_lee",
    "himmelblau", "zdt1", "zdt2", "zdt3", "viennet2", "dtlz2",
)
