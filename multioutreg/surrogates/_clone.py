# Copyright (c) 2026 takotime808

"""``sklearn.base.clone`` support for surrogates that are not ``BaseEstimator``s."""

import inspect

from sklearn.base import clone


class CloneMixin:
    """Make ``sklearn.base.clone`` return a fresh, unfitted copy of a surrogate.

    Surrogates are rebuilt from ``get_params``, so ``set_params`` changes carry
    over. The generic ``get_params`` / ``set_params`` here work from the
    arguments the surrogate was constructed with (the ``BaseSurrogate``
    wrappers keep their hyperparameters inside the wrapped estimator, so there
    are no attributes to read them from); surrogates with their own versions
    override them. Either way nothing fitted is copied, so this also works for
    models whose fitted state cannot be deep-copied (e.g. the Rust-backed GPX
    model).
    """

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj._clone_init_args = (args, kwargs)
        return obj

    def get_params(self, deep: bool = True) -> dict:
        """Constructor arguments this surrogate was built with, by name.

        Arguments taken through ``**kwargs`` (passed on to the wrapped
        estimator) are listed individually; defaults filled in inside the
        constructor are not. ``deep`` is accepted for scikit-learn
        compatibility; nested estimator parameters are not expanded.
        """
        args, kwargs = self._clone_init_args
        sig = inspect.signature(type(self).__init__)
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        params = {}
        for name, value in list(bound.arguments.items())[1:]:  # skip self
            kind = sig.parameters[name].kind
            if kind is inspect.Parameter.VAR_KEYWORD:
                params.update(value)
            elif kind is inspect.Parameter.VAR_POSITIONAL:
                if value:
                    raise TypeError(f"{type(self).__name__}: *args cannot be named in get_params")
            else:
                params[name] = value
        return params

    def set_params(self, **params):
        """Set constructor arguments and rebuild the surrogate.

        The surrogate is reconstructed from the merged arguments, so a fitted
        surrogate must be refitted afterwards.
        """
        if not params:
            return self
        merged = {**self.get_params(deep=False), **params}
        self.__init__(**merged)
        self._clone_init_args = ((), merged)
        return self

    def __sklearn_clone__(self):
        kwargs = self.get_params(deep=False)
        return type(self)(**{k: clone(v, safe=False) for k, v in kwargs.items()})
