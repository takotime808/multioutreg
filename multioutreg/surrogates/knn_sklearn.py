# Copyright (c) 2025 takotime808

from sklearn.neighbors import KNeighborsRegressor
from multioutreg.surrogates.base_sklearn import BaseSurrogate

class KNeighborsSurrogate(BaseSurrogate):
    """K-Nearest Neighbors regression surrogate."""
    def __init__(self, **kwargs):
        super().__init__(KNeighborsRegressor(**kwargs))