# Copyright (c) 2025 takotime808

from sklearn.ensemble import GradientBoostingRegressor
from multioutreg.surrogates.base_sklearn import BaseSurrogate

class GradientBoostingSurrogate(BaseSurrogate):
    """Gradient Boosting regression surrogate."""
    def __init__(self, **kwargs):
        super().__init__(GradientBoostingRegressor(**kwargs))