# Copyright (c) 2025 takotime808

"""Collection of vendorized surrogate models."""

from multioutreg.surrogates.conformal_mixin import ConformalMixin
from multioutreg.surrogates.base_sklearn import BaseSurrogate
from multioutreg.surrogates.linear_sklearn import LinearRegressionSurrogate
from multioutreg.surrogates.gp_sklearn import GaussianProcessSurrogate
from multioutreg.surrogates.rf_sklearn import RandomForestSurrogate
from multioutreg.surrogates.gradient_boosting_sklearn import GradientBoostingSurrogate
from multioutreg.surrogates.svr_sklearn import SVRSurrogate
from multioutreg.surrogates.knn_sklearn import KNeighborsSurrogate
from multioutreg.surrogates.decision_tree_sklearn import DecisionTreeRegressorSurrogate
from multioutreg.surrogates.conformal_network_sklearn import ConformalPredictionNetworkSurrogate
from multioutreg.surrogates.extra_trees_sklearn import ExtraTreesRegressorSurrogate
from multioutreg.surrogates.ngboost_sklearn import NGBoostSurrogate
from multioutreg.surrogates.bnn_pytorch import BNNSurrogate
from multioutreg.surrogates.multi_fidelity import MultiFidelitySurrogate
from multioutreg.surrogates.moe_surrogate import MixtureOfExpertsSurrogate

__all__ = [
    "ConformalMixin",
    "BaseSurrogate",
    "LinearRegressionSurrogate",
    "GaussianProcessSurrogate",
    "RandomForestSurrogate",
    "GradientBoostingSurrogate",
    "SVRSurrogate",
    "KNeighborsSurrogate",
    "DecisionTreeRegressorSurrogate",
    "ConformalPredictionNetworkSurrogate",
    "ExtraTreesRegressorSurrogate",
    "NGBoostSurrogate",
    "BNNSurrogate",
    "MultiFidelitySurrogate",
    "MixtureOfExpertsSurrogate",
]

