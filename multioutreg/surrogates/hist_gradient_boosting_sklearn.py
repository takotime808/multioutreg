# Copyright (c) 2026 takotime808

"""HistGradientBoosting surrogate — zero extra dependencies."""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from multioutreg.surrogates.base_sklearn import BaseSurrogate


class HistGradientBoostingSurrogate(BaseSurrogate):
    """Histogram-based Gradient Boosting regression surrogate.

    Wraps :class:`sklearn.ensemble.HistGradientBoostingRegressor` in a
    ``MultiOutputRegressor``.  Significantly faster than
    ``GradientBoostingSurrogate`` for large datasets (n > 5 000) due to the
    histogram-bin approximation.  Requires no additional dependencies beyond
    scikit-learn.

    Uncertainty is not natively available from this estimator; use
    :meth:`wrap_conformal` / :meth:`conformal_predict` for
    distribution-free prediction intervals.

    Parameters
    ----------
    max_iter : int, default 100
        Number of boosting rounds.
    learning_rate : float, default 0.1
        Shrinkage applied to each tree.
    max_leaf_nodes : int or None, default 31
        Maximum number of leaves per tree.  ``None`` means no limit.
    max_depth : int or None, default None
        Maximum depth of each tree.
    min_samples_leaf : int, default 20
        Minimum number of samples per leaf.
    **kwargs
        Additional keyword arguments forwarded to
        ``HistGradientBoostingRegressor``.
    """

    def __init__(
        self,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        max_leaf_nodes: int = 31,
        max_depth=None,
        min_samples_leaf: int = 20,
        **kwargs,
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self._extra_kwargs = kwargs
        super().__init__(
            HistGradientBoostingRegressor(
                max_iter=max_iter,
                learning_rate=learning_rate,
                max_leaf_nodes=max_leaf_nodes,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                **kwargs,
            )
        )

    def get_params(self, deep: bool = True) -> dict:
        params = {
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "max_leaf_nodes": self.max_leaf_nodes,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
        }
        params.update(self._extra_kwargs)
        return params

    def set_params(self, **params) -> "HistGradientBoostingSurrogate":
        for key, value in params.items():
            setattr(self, key, value)
            if key in self._extra_kwargs:
                self._extra_kwargs[key] = value
        self.model = MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=self.max_iter,
                learning_rate=self.learning_rate,
                max_leaf_nodes=self.max_leaf_nodes,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                **self._extra_kwargs,
            )
        )
        return self
