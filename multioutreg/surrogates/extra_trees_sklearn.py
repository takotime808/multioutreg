# Copyright (c) 2026 takotime808

from sklearn.ensemble import ExtraTreesRegressor
from multioutreg.surrogates.forest_uncertainty import ForestSurrogate


class ExtraTreesRegressorSurrogate(ForestSurrogate):
    """Extra-Trees surrogate; std = tree spread + held-out noise estimate.

    Extra-Trees does not bootstrap by default, so the noise is estimated by
    ``noise_cv``-fold refits (out-of-bag when ``bootstrap=True``).
    ``include_noise=False`` returns the tree spread alone (epistemic only).
    See :mod:`multioutreg.surrogates.forest_uncertainty`.
    """

    def __init__(self, include_noise=True, noise_cv=5, **kwargs):
        super().__init__(ExtraTreesRegressor(**kwargs), include_noise, noise_cv)
