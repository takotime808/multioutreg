# Copyright (c) 2025 takotime808

from sklearn.tree import DecisionTreeRegressor
from multioutreg.surrogates.base_sklearn import BaseSurrogate

class DecisionTreeRegressorSurrogate(BaseSurrogate):
    """Multi-output Decision-tree surrogate."""
    def __init__(self, **kwargs):
        super().__init__(DecisionTreeRegressor(**kwargs))