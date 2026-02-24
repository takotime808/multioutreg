# Copyright (c) 2026 takotime808

"""XGBoost surrogate — optional dependency ``pip install xgboost``."""

from sklearn.multioutput import MultiOutputRegressor

from multioutreg.surrogates.base_sklearn import BaseSurrogate

try:
    from xgboost import XGBRegressor as _XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBRegressor = None  # type: ignore[assignment,misc]
    _XGBOOST_AVAILABLE = False


def _require_xgboost() -> None:
    if not _XGBOOST_AVAILABLE:
        raise ImportError(
            "xgboost is required for XGBoostSurrogate. "
            "Install it with: pip install xgboost"
        )


class XGBoostSurrogate(BaseSurrogate):
    """XGBoost regression surrogate.

    Wraps :class:`xgboost.XGBRegressor` in a ``MultiOutputRegressor``.
    Handles missing values natively; GPU support available via
    ``device="cuda"``; widely used engineering competition standard.

    Requires the optional ``xgboost`` package::

        pip install xgboost

    Uncertainty is not natively returned; use :meth:`wrap_conformal` /
    :meth:`conformal_predict` for distribution-free prediction intervals.

    Parameters
    ----------
    n_estimators : int, default 200
        Number of boosting rounds.
    learning_rate : float, default 0.05
        Shrinkage applied to each tree.
    max_depth : int, default 6
        Maximum tree depth.
    verbosity : int, default 0
        XGBoost verbosity.  ``0`` is silent.
    **kwargs
        Additional keyword arguments forwarded to ``XGBRegressor``.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        verbosity: int = 0,
        **kwargs,
    ):
        _require_xgboost()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.verbosity = verbosity
        self._extra_kwargs = kwargs
        super().__init__(
            _XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                verbosity=verbosity,
                **kwargs,
            )
        )

    def get_params(self, deep: bool = True) -> dict:
        params = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "verbosity": self.verbosity,
        }
        params.update(self._extra_kwargs)
        return params

    def set_params(self, **params) -> "XGBoostSurrogate":
        _require_xgboost()
        for key, value in params.items():
            setattr(self, key, value)
            if key in self._extra_kwargs:
                self._extra_kwargs[key] = value
        self.model = MultiOutputRegressor(
            _XGBRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                verbosity=self.verbosity,
                **self._extra_kwargs,
            )
        )
        return self
