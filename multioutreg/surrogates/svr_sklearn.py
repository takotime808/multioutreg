# Copyright (c) 2025 takotime808

from sklearn.svm import SVR
from multioutreg.surrogates.base_sklearn import BaseSurrogate

class SVRSurrogate(BaseSurrogate):
    """Support Vector Regression surrogate."""
    def __init__(self, **kwargs):
        super().__init__(SVR(**kwargs))