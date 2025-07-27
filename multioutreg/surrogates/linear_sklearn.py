# Copyright (c) 2025 takotime808

from sklearn.linear_model import LinearRegression
from multioutreg.surrogates.base_sklearn import BaseSurrogate

class LinearRegressionSurrogate(BaseSurrogate):
    """Multi-output linear regression surrogate."""
    def __init__(self, **kwargs):
        super().__init__(LinearRegression(**kwargs))