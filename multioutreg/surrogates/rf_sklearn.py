# Copyright (c) 2025 takotime808

from sklearn.ensemble import RandomForestRegressor
from multioutreg.surrogates.forest_uncertainty import ForestSurrogate

class RandomForestSurrogate(ForestSurrogate):
    """Random Forest surrogate; std = tree spread + out-of-bag noise estimate.

    ``include_noise=False`` returns the tree spread alone (epistemic only).
    See :mod:`multioutreg.surrogates.forest_uncertainty`.
    """
    def __init__(self, include_noise=True, noise_cv=5, **kwargs):
        super().__init__(RandomForestRegressor(**kwargs), include_noise, noise_cv)
