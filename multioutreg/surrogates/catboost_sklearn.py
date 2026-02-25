# Copyright (c) 2026 takotime808

"""CatBoost surrogate — optional dependency ``pip install catboost``."""

import numpy as np
from sklearn.multioutput import MultiOutputRegressor

from multioutreg.surrogates.base_sklearn import BaseSurrogate

try:
    from catboost import CatBoostRegressor as _CatBoostRegressor
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CatBoostRegressor = None  # type: ignore[assignment,misc]
    _CATBOOST_AVAILABLE = False


def _require_catboost() -> None:
    if not _CATBOOST_AVAILABLE:
        raise ImportError(
            "catboost is required for CatBoostSurrogate. "
            "Install it with: pip install catboost"
        )


class CatBoostSurrogate(BaseSurrogate):
    """CatBoost regression surrogate with optional native uncertainty.

    Wraps :class:`catboost.CatBoostRegressor` in a ``MultiOutputRegressor``.
    When ``use_uncertainty=True`` (default), trains with
    ``loss_function="RMSEWithUncertainty"`` and returns aleatoric + epistemic
    variance via CatBoost's virtual ensemble mechanism — the only tree-based
    surrogate in this library with native (non-conformal) uncertainty output.

    Requires the optional ``catboost`` package::

        pip install catboost

    Parameters
    ----------
    n_estimators : int, default 200
        Number of boosting iterations.
    learning_rate : float, default 0.05
        Shrinkage applied to each tree.
    depth : int, default 6
        Maximum tree depth.
    use_uncertainty : bool, default True
        If True, trains with ``loss_function="RMSEWithUncertainty"`` and
        returns native uncertainty from ``virtual_ensembles_predict``.
        If False, uses standard RMSE loss and returns zeros for std.
    virtual_ensembles_count : int, default 10
        Number of virtual ensembles used when estimating uncertainty.
        Higher values give smoother variance estimates at extra compute cost.
    verbose : bool | int, default False
        CatBoost verbosity. ``False`` / ``0`` suppresses all output.
    **kwargs
        Additional keyword arguments forwarded to ``CatBoostRegressor``.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        depth: int = 6,
        use_uncertainty: bool = True,
        virtual_ensembles_count: int = 10,
        verbose: bool | int = False,
        **kwargs,
    ):
        _require_catboost()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.depth = depth
        self.use_uncertainty = use_uncertainty
        self.virtual_ensembles_count = virtual_ensembles_count
        self.verbose = verbose
        self._extra_kwargs = kwargs
        loss_function = "RMSEWithUncertainty" if use_uncertainty else "RMSE"
        super().__init__(
            _CatBoostRegressor(
                iterations=n_estimators,
                learning_rate=learning_rate,
                depth=depth,
                loss_function=loss_function,
                verbose=verbose,
                **kwargs,
            )
        )

    def predict(
        self, X: np.ndarray, return_std: bool = False
    ) -> "np.ndarray | tuple[np.ndarray, np.ndarray]":
        """Predict outputs for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        return_std : bool, default False
            If True and ``use_uncertainty=True``, returns predictive std from
            CatBoost virtual ensembles (total uncertainty = aleatoric + epistemic).

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples, n_outputs)
        y_std  : np.ndarray, shape (n_samples, n_outputs)  [only if return_std]
        """
        if not return_std or not self.use_uncertainty:
            return super().predict(X, return_std=return_std)

        # Virtual-ensemble uncertainty: call virtual_ensembles_predict per output.
        # "TotalUncertainty" returns (n_samples, 3): [mean, total_var, knowledge_var]
        preds, stds = [], []
        for est in self.model.estimators_:
            unc = est.virtual_ensembles_predict(
                X,
                virtual_ensembles_count=self.virtual_ensembles_count,
                prediction_type="TotalUncertainty",
            )
            preds.append(unc[:, 0])
            stds.append(np.sqrt(np.maximum(unc[:, 1], 0.0)))

        return np.column_stack(preds), np.column_stack(stds)

    def get_params(self, deep: bool = True) -> dict:
        params = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "use_uncertainty": self.use_uncertainty,
            "virtual_ensembles_count": self.virtual_ensembles_count,
            "verbose": self.verbose,
        }
        params.update(self._extra_kwargs)
        return params

    def set_params(self, **params) -> "CatBoostSurrogate":
        _require_catboost()
        for key, value in params.items():
            setattr(self, key, value)
            if key in self._extra_kwargs:
                self._extra_kwargs[key] = value
        loss_function = "RMSEWithUncertainty" if self.use_uncertainty else "RMSE"
        self.model = MultiOutputRegressor(
            _CatBoostRegressor(
                iterations=self.n_estimators,
                learning_rate=self.learning_rate,
                depth=self.depth,
                loss_function=loss_function,
                verbose=self.verbose,
                **self._extra_kwargs,
            )
        )
        return self
