# Copyright (c) 2026 takotime808

"""LightGBM surrogate — optional dependency ``pip install lightgbm``."""

from sklearn.multioutput import MultiOutputRegressor

from multioutreg.surrogates.base_sklearn import BaseSurrogate

try:
    from lightgbm import LGBMRegressor as _LGBMRegressor
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LGBMRegressor = None  # type: ignore[assignment,misc]
    _LIGHTGBM_AVAILABLE = False


def _require_lightgbm() -> None:
    if not _LIGHTGBM_AVAILABLE:
        raise ImportError(
            "lightgbm is required for LightGBMSurrogate. "
            "Install it with: pip install lightgbm"
        )


class LightGBMSurrogate(BaseSurrogate):
    """LightGBM regression surrogate.

    Wraps :class:`lightgbm.LGBMRegressor` in a ``MultiOutputRegressor``.
    Typically 10–20× faster than sklearn ``GradientBoostingRegressor`` at
    comparable accuracy; supports GPU acceleration via ``device="gpu"``.

    Requires the optional ``lightgbm`` package::

        pip install lightgbm

    Uncertainty is not natively returned; use :meth:`wrap_conformal` /
    :meth:`conformal_predict` for distribution-free prediction intervals.

    Parameters
    ----------
    n_estimators : int, default 200
        Number of boosting rounds.
    learning_rate : float, default 0.05
        Shrinkage applied to each tree.
    num_leaves : int, default 31
        Maximum number of leaves per tree (main complexity control in LightGBM).
    verbose : int, default -1
        Verbosity level.  ``-1`` suppresses all output.
    **kwargs
        Additional keyword arguments forwarded to ``LGBMRegressor``.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        verbose: int = -1,
        **kwargs,
    ):
        _require_lightgbm()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.verbose = verbose
        self._extra_kwargs = kwargs
        super().__init__(
            _LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                verbose=verbose,
                **kwargs,
            )
        )

    def get_params(self, deep: bool = True) -> dict:
        params = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "verbose": self.verbose,
        }
        params.update(self._extra_kwargs)
        return params

    def set_params(self, **params) -> "LightGBMSurrogate":
        _require_lightgbm()
        for key, value in params.items():
            setattr(self, key, value)
            if key in self._extra_kwargs:
                self._extra_kwargs[key] = value
        self.model = MultiOutputRegressor(
            _LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                verbose=self.verbose,
                **self._extra_kwargs,
            )
        )
        return self
