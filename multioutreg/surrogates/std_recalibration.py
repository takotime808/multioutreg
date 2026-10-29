# Copyright (c) 2026 takotime808

"""Std recalibration wrapper: rescale a surrogate's own predicted std per output.

Conformal prediction (``ConformalMixin``) corrects *intervals*, but the
acquisition functions, the joint scores and the report's calibration plots
read a model's ``std`` directly. When a surrogate is over- or under-dispersed
those all inherit the error. This wrapper learns one factor ``s_j`` per output
so that ``s_j * std_j`` is calibrated (Kuleshov et al. 2018, variance
scaling), leaving the mean unchanged.

The factors are learned from out-of-fold predictions (``cv`` folds) so the
in-sample overconfidence of interpolating models does not leak in, or from a
held-out set via :meth:`StdRecalibratedSurrogate.recalibrate`.
"""

import numpy as np
from sklearn.model_selection import KFold

from multioutreg.surrogates.conformal_mixin import ConformalMixin

_CRITERIA = ("ma_cal", "rms_cal", "miscal", "nll")


def _fit_ratio(mean, std, y, criterion):
    """Scale factor for one output's std."""
    std = np.maximum(std, 1e-12)
    if criterion == "nll":
        # Gaussian NLL is minimized in closed form by s^2 = mean(z^2).
        return float(np.sqrt(np.mean(((y - mean) / std) ** 2)))
    import uncertainty_toolbox as uct

    return float(
        uct.recalibration.optimize_recalibration_ratio(mean, std, y, criterion=criterion)
    )


class StdRecalibratedSurrogate(ConformalMixin):
    """Scale a surrogate's predicted std per output to make it calibrated.

    Parameters
    ----------
    surrogate_cls : type | None
        Surrogate to recalibrate; must accept ``fit(X, Y)`` and
        ``predict(X, return_std=True)``. Defaults to
        ``GaussianProcessSurrogate``.
    criterion : {"ma_cal", "rms_cal", "miscal", "nll"}
        What the factor optimizes. The first three are uncertainty-toolbox's
        calibration errors (mean absolute, root mean square, max), searched
        numerically; ``"nll"`` minimizes the Gaussian negative log-likelihood
        in closed form. Default ``"ma_cal"``.
    cv : int
        Folds for the out-of-fold predictions the factors are learned from.
    random_state : int | None
        Seed for the fold split.
    **surrogate_kwargs
        Passed to ``surrogate_cls``.

    Attributes
    ----------
    std_scale_ : np.ndarray, shape (n_outputs,)
        Factor applied to each output's std. Above 1 means the wrapped
        surrogate was overconfident.
    """

    def __init__(self, surrogate_cls=None, criterion="ma_cal", cv=5, random_state=None, **surrogate_kwargs):
        if criterion not in _CRITERIA:
            raise ValueError(f"criterion must be one of {_CRITERIA}, got {criterion!r}")
        self.surrogate_cls = surrogate_cls
        self.criterion = criterion
        self.cv = cv
        self.random_state = random_state
        self.surrogate_kwargs = surrogate_kwargs

    def _make_surrogate(self):
        cls = self.surrogate_cls
        if cls is None:
            from multioutreg.surrogates.gp_sklearn import GaussianProcessSurrogate

            cls = GaussianProcessSurrogate
        return cls(**self.surrogate_kwargs)

    @staticmethod
    def _mean_std(model, X):
        mean, std = model.predict(X, return_std=True)
        return (
            np.asarray(mean, dtype=float).reshape(len(X), -1),
            np.asarray(std, dtype=float).reshape(len(X), -1),
        )

    def _set_scales(self, mean, std, Y):
        self.std_scale_ = np.array([
            _fit_ratio(mean[:, j], std[:, j], Y[:, j], self.criterion)
            for j in range(Y.shape[1])
        ])

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.n_outputs_ = Y.shape[1]
        n_splits = min(int(self.cv), len(X))
        if n_splits < 2:
            raise ValueError(f"cv must be >= 2 (and at most the number of rows); got {self.cv}")

        mean = np.empty_like(Y)
        std = np.empty_like(Y)
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        for train, test in folds.split(X):
            model = self._make_surrogate().fit(X[train], Y[train])
            mean[test], std[test] = self._mean_std(model, X[test])
        self._set_scales(mean, std, Y)

        self.surrogate_ = self._make_surrogate().fit(X, Y)
        return self

    def recalibrate(self, X_cal, Y_cal):
        """Relearn the factors from held-out data, keeping the fitted surrogate."""
        X_cal = np.asarray(X_cal, dtype=float)
        Y_cal = np.asarray(Y_cal, dtype=float).reshape(len(X_cal), -1)
        mean, std = self._mean_std(self.surrogate_, X_cal)
        self._set_scales(mean, std, Y_cal)
        return self

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=float)
        if not return_std:
            return np.asarray(self.surrogate_.predict(X), dtype=float).reshape(len(X), -1)
        mean, std = self._mean_std(self.surrogate_, X)
        return mean, std * self.std_scale_

    def _conformal_point_predict(self, X):
        return self.predict(X)
