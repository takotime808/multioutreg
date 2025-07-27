# Copyright (c) 2025 takotime808

"""Collection of vendorized surrogate models."""

from multioutreg.surrogates.base_sklearn import BaseSurrogate
from multioutreg.surrogates.linear_sklearn import LinearRegressionSurrogate
from multioutreg.surrogates.gp_sklearn import GaussianProcessSurrogate
from multioutreg.surrogates.rf_sklearn import RandomForestSurrogate
from multioutreg.surrogates.gradient_boosting_sklearn import GradientBoostingSurrogate
from multioutreg.surrogates.svr_sklearn import SVRSurrogate
from multioutreg.surrogates.knn_sklearn import KNeighborsSurrogate
from multioutreg.surrogates.decision_tree_sklearn import DecisionTreeRegressorSurrogate

__all__ = [
    "BaseSurrogate",
    "LinearRegressionSurrogate",
    "GaussianProcessSurrogate",
    "RandomForestSurrogate",
    "GradientBoostingSurrogate",
    "SVRSurrogate",
    "KNeighborsSurrogate",
    "DecisionTreeRegressorSurrogate",
]

