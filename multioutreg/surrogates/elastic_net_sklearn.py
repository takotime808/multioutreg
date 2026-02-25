# Copyright (c) 2026 takotime808

"""ElasticNet and Lasso regression surrogates (sparse linear models)."""

from sklearn.linear_model import ElasticNet, Lasso
from sklearn.multioutput import MultiOutputRegressor

from multioutreg.surrogates.base_sklearn import BaseSurrogate


class ElasticNetSurrogate(BaseSurrogate):
    """Elastic Net regression surrogate.

    Wraps :class:`sklearn.linear_model.ElasticNet` in a ``MultiOutputRegressor``.
    Combines L1 (Lasso) and L2 (Ridge) penalties to zero out irrelevant features
    while remaining stable when inputs are correlated.  Particularly useful in the
    high-dimensional regime ``p >> n`` where purely linear baselines help isolate
    which features drive each output.

    Uncertainty is not natively returned; use :meth:`wrap_conformal` /
    :meth:`conformal_predict` for distribution-free prediction intervals.

    Parameters
    ----------
    alpha : float, default 1.0
        Overall regularization strength (larger → stronger shrinkage).
    l1_ratio : float, default 0.5
        Mix between L1 (``l1_ratio=1``) and L2 (``l1_ratio=0``) penalty.
        ``l1_ratio=0.5`` balances sparsity and stability.
    max_iter : int, default 1000
        Maximum number of coordinate descent iterations.
    **kwargs
        Additional keyword arguments forwarded to ``ElasticNet``.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        **kwargs,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self._extra_kwargs = kwargs
        super().__init__(
            ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, **kwargs)
        )

    def get_params(self, deep: bool = True) -> dict:
        params = {
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "max_iter": self.max_iter,
        }
        params.update(self._extra_kwargs)
        return params

    def set_params(self, **params) -> "ElasticNetSurrogate":
        for key, value in params.items():
            setattr(self, key, value)
            if key in self._extra_kwargs:
                self._extra_kwargs[key] = value
        self.model = MultiOutputRegressor(
            ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                max_iter=self.max_iter,
                **self._extra_kwargs,
            )
        )
        return self


class LassoSurrogate(BaseSurrogate):
    """Lasso (L1) regression surrogate.

    Wraps :class:`sklearn.linear_model.Lasso` in a ``MultiOutputRegressor``.
    Pure L1 penalty drives feature coefficients to exactly zero, producing a
    sparse model where only the most predictive inputs retain non-zero weights.
    Useful when interpretability and explicit feature selection matter more than
    prediction at moderate sample sizes.

    Uncertainty is not natively returned; use :meth:`wrap_conformal` /
    :meth:`conformal_predict` for distribution-free prediction intervals.

    Parameters
    ----------
    alpha : float, default 1.0
        L1 regularization strength.
    max_iter : int, default 1000
        Maximum number of coordinate descent iterations.
    **kwargs
        Additional keyword arguments forwarded to ``Lasso``.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        max_iter: int = 1000,
        **kwargs,
    ):
        self.alpha = alpha
        self.max_iter = max_iter
        self._extra_kwargs = kwargs
        super().__init__(Lasso(alpha=alpha, max_iter=max_iter, **kwargs))

    def get_params(self, deep: bool = True) -> dict:
        params = {"alpha": self.alpha, "max_iter": self.max_iter}
        params.update(self._extra_kwargs)
        return params

    def set_params(self, **params) -> "LassoSurrogate":
        for key, value in params.items():
            setattr(self, key, value)
            if key in self._extra_kwargs:
                self._extra_kwargs[key] = value
        self.model = MultiOutputRegressor(
            Lasso(alpha=self.alpha, max_iter=self.max_iter, **self._extra_kwargs)
        )
        return self
