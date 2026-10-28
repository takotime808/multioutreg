# Copyright (c) 2026 takotime808

"""``sklearn.base.clone`` support for surrogates that are not ``BaseEstimator``s."""

from sklearn.base import clone


class CloneMixin:
    """Make ``sklearn.base.clone`` return a fresh, unfitted copy of a surrogate.

    Surrogates with ``get_params`` are rebuilt from those parameters, so
    ``set_params`` changes carry over. The others (e.g. the ``BaseSurrogate``
    wrappers, whose hyperparameters live inside the wrapped estimator) are
    rebuilt from the arguments they were constructed with. Either way nothing
    fitted is copied, so this also works for models whose fitted state cannot
    be deep-copied (e.g. the Rust-backed GPX model).
    """

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj._clone_init_args = (args, kwargs)
        return obj

    def __sklearn_clone__(self):
        if hasattr(self, "get_params"):
            args, kwargs = (), self.get_params(deep=False)
        else:
            args, kwargs = self._clone_init_args
        return type(self)(
            *(clone(a, safe=False) for a in args),
            **{k: clone(v, safe=False) for k, v in kwargs.items()},
        )
