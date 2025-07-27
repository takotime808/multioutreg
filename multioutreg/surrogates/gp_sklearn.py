# Copyright (c) 2025 takotime808

from sklearn.gaussian_process import GaussianProcessRegressor
from multioutreg.surrogates.base_sklearn import BaseSurrogate

class GaussianProcessSurrogate(BaseSurrogate):
    """Gaussian Process surrogate with uncertainty."""
    def __init__(self, **kwargs):
        super().__init__(GaussianProcessRegressor(**kwargs))