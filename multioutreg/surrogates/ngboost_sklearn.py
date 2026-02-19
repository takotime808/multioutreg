# Copyright (c) 2026 takotime808

"""NGBoost surrogate with native probabilistic uncertainty."""

import numpy as np

try:
    from ngboost import NGBRegressor as _NGBRegressor
    _NGBOOST_AVAILABLE = True
except ImportError:
    _NGBOOST_AVAILABLE = False

from multioutreg.surrogates.base_sklearn import BaseSurrogate


def _require_ngboost():
    if not _NGBOOST_AVAILABLE:
        raise ImportError(
            "ngboost is required for NGBoostSurrogate. "
            "Install it with: pip install ngboost"
        )


class NGBoostSurrogate(BaseSurrogate):
    """NGBoost surrogate providing native probabilistic uncertainty.

    Wraps ngboost.NGBRegressor in a MultiOutputRegressor and exposes
    per-output predictive standard deviations via NGBoost's natural-gradient
    distributional output.

    Parameters
    ----------
    n_estimators : int, default 500
        Number of boosting iterations.
    learning_rate : float, default 0.01
    verbose : bool, default False
    **kwargs
        Additional keyword arguments passed to NGBRegressor.
    """

    def __init__(self, n_estimators=500, learning_rate=0.01, verbose=False, **kwargs):
        _require_ngboost()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.verbose = verbose
        self._extra_kwargs = kwargs
        super().__init__(
            _NGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                verbose=verbose,
                **kwargs,
            )
        )

    def get_params(self, deep=True):
        params = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "verbose": self.verbose,
        }
        params.update(self._extra_kwargs)
        return params

    def set_params(self, **params):
        _require_ngboost()
        from sklearn.multioutput import MultiOutputRegressor
        for key, value in params.items():
            setattr(self, key, value)
            if key in self._extra_kwargs:
                self._extra_kwargs[key] = value
        self.model = MultiOutputRegressor(
            _NGBRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                verbose=self.verbose,
                **self._extra_kwargs,
            )
        )
        return self

    def predict(self, X, return_std=False):
        preds = np.asarray(self.model.predict(X))
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        if not return_std:
            return preds

        stds = []
        for estimator in self.model.estimators_:
            dist = estimator.pred_dist(X)
            stds.append(dist.std())
        std = np.column_stack(stds)
        return preds, std
