# Copyright (c) 2026 takotime808

"""Multi-fidelity co-kriging surrogate — wraps SMT's ``MFK`` / ``MFKPLS``.

Recursive co-kriging (Le Gratiet & Garnier 2014), the standard autoregressive
(AR1, Kennedy & O'Hagan 2000) multi-fidelity GP. Each level is modelled as::

    f_k(x) = rho_{k-1}(x) * f_{k-1}(x) + delta_k(x)

with ``delta_k`` a GP. Unlike ``AdditiveCorrectionVFM``, which fits ``rho`` by
least squares on the low-fidelity predictions (biased whenever ``delta`` is
correlated with ``f_lo``), ``rho`` and ``delta`` are estimated jointly by
maximum likelihood, for any number of levels. On the Forrester pair it
recovers the true ``rho = 2`` from five high-fidelity points.

The designs of the levels do not need to be nested. With ``n_comp`` set, the
PLS variant (``MFKPLS``) reduces the input dimension before kriging, for
high-dimensional inputs.

Installation
------------
    pip install smt          # or: pip install multioutreg[smt]

If the optional dependency is not installed this module still imports, and
``MFKSurrogate`` raises ``ImportError`` at instantiation.
"""

import numpy as np

try:
    from smt.applications import MFK as _SMTMFK, MFKPLS as _SMTMFKPLS
    _MFK_AVAILABLE = True
except ImportError:
    _SMTMFK = _SMTMFKPLS = None
    _MFK_AVAILABLE = False

from multioutreg.surrogates.conformal_mixin import ConformalMixin


def _check_available():
    if not _MFK_AVAILABLE:
        raise ImportError(
            "MFKSurrogate requires the smt package. "
            "Install it with: pip install smt"
        )


class MFKSurrogate(ConformalMixin):
    """Multi-fidelity co-kriging (AR1) surrogate via SMT, one model per output.

    Requires ``pip install smt``.

    Parameters
    ----------
    fidelity_levels : sequence of str
        Level names ordered from lowest to highest fidelity (at least 2).
    rho_regr : {"constant", "linear", "quadratic"}
        Form of the scale factor ``rho(x)`` between consecutive levels.
    corr : {"squar_exp", "abs_exp", "pow_exp", "matern52", "matern32"}
        Correlation function of each level's GP.
    poly : {"constant", "linear", "quadratic"}
        Trend of each level's GP.
    n_comp : int | None
        Use ``MFKPLS`` with this many PLS components (clamped to the number
        of inputs). ``None`` (default) uses plain ``MFK``.
    eval_noise : bool
        Estimate a noise variance per level (for noisy simulators).
    propagate_uncertainty : bool
        Include the lower levels' variance in the predicted std (default
        ``True``). Without it the std counts the lower levels as exact.
    n_start : int
        Optimizer restarts for the hyperparameters.

    Attributes
    ----------
    rho_ : np.ndarray, shape (n_outputs, n_levels - 1) | None
        Fitted scale factor between each pair of consecutive levels, per
        output. Only set when ``rho_regr="constant"``.
    """

    def __init__(
        self,
        fidelity_levels=("lo", "hi"),
        rho_regr="constant",
        corr="squar_exp",
        poly="constant",
        n_comp=None,
        eval_noise=False,
        propagate_uncertainty=True,
        n_start=10,
    ):
        _check_available()
        self.fidelity_levels = list(fidelity_levels)
        if len(self.fidelity_levels) < 2:
            raise ValueError("MFKSurrogate needs at least two fidelity levels.")
        self.rho_regr = rho_regr
        self.corr = corr
        self.poly = poly
        self.n_comp = n_comp
        self.eval_noise = eval_noise
        self.propagate_uncertainty = propagate_uncertainty
        self.n_start = n_start

    def _make_model(self, n_features):
        kwargs = dict(
            rho_regr=self.rho_regr,
            corr=self.corr,
            poly=self.poly,
            eval_noise=self.eval_noise,
            propagate_uncertainty=self.propagate_uncertainty,
            n_start=self.n_start,
            print_global=False,
        )
        if self.n_comp is None:
            return _SMTMFK(**kwargs)
        return _SMTMFKPLS(n_comp=min(int(self.n_comp), n_features), **kwargs)

    def fit(self, data):
        """Fit on ``{level: (X, Y)}`` with one entry per fidelity level.

        The designs of different levels may differ in size and need not be
        nested. ``Y`` may be 1-D for a single output.
        """
        missing = [lvl for lvl in self.fidelity_levels if lvl not in data]
        if missing:
            raise ValueError(f"no data for fidelity levels {missing}")
        Xs, Ys = [], []
        for lvl in self.fidelity_levels:
            X, Y = data[lvl]
            X = np.asarray(X, dtype=float)
            Y = np.asarray(Y, dtype=float)
            Xs.append(X)
            Ys.append(Y.reshape(len(Y), -1))
        n_outputs = {Y.shape[1] for Y in Ys}
        if len(n_outputs) != 1:
            raise ValueError("every fidelity level must have the same number of outputs")
        self.n_outputs_ = n_outputs.pop()
        n_features = Xs[0].shape[1]

        self.estimators_ = []
        for j in range(self.n_outputs_):
            model = self._make_model(n_features)
            for k, (X, Y) in enumerate(zip(Xs[:-1], Ys[:-1])):
                model.set_training_values(X, Y[:, j:j + 1], name=k)
            model.set_training_values(Xs[-1], Ys[-1][:, j:j + 1])
            model.train()
            self.estimators_.append(model)

        if self.rho_regr == "constant":
            # beta of level k >= 1 is [rho, trend coefficients...].
            self.rho_ = np.array([
                [float(m.optimal_par[k]["beta"][0, 0]) for k in range(1, len(self.fidelity_levels))]
                for m in self.estimators_
            ])
        else:
            self.rho_ = None
        return self

    def _level_index(self, level):
        if level is None:
            return len(self.fidelity_levels) - 1
        if level not in self.fidelity_levels:
            raise ValueError(f"unknown fidelity level {level!r}; expected one of {self.fidelity_levels}")
        return self.fidelity_levels.index(level)

    def predict(self, X, return_std=False, level=None):
        """Predict at ``level`` (default: the highest fidelity)."""
        if not hasattr(self, "estimators_"):
            raise AttributeError("MFKSurrogate is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=float)
        idx = self._level_index(level)
        top = idx == len(self.fidelity_levels) - 1
        means, stds = [], []
        for model in self.estimators_:
            if top:
                means.append(model.predict_values(X).ravel())
            else:
                means.append(model._predict_intermediate_values(X, idx + 1).ravel())
            if return_std:
                if top:
                    var = model.predict_variances(X).ravel()
                else:
                    var = model.predict_variances_all_levels(X)[0][:, idx]
                stds.append(np.sqrt(np.maximum(var, 0.0)))
        mean = np.column_stack(means)
        if return_std:
            return mean, np.column_stack(stds)
        return mean

    def _conformal_point_predict(self, X):
        return self.predict(X)
