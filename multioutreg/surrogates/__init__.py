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
from multioutreg.surrogates.stacked_vfm import StackedVFMSurrogate, AdditiveCorrectionVFM
from multioutreg.surrogates.moe_surrogate import MixtureOfExpertsSurrogate
from multioutreg.surrogates.bayesian_ridge_sklearn import BayesianRidgeSurrogate
from multioutreg.surrogates.rfgp_sklearn import RFFGPSurrogate
from multioutreg.surrogates.polynomial_bayesian_ridge_sklearn import PolynomialBayesianRidgeSurrogate
from multioutreg.surrogates.nystroem_gp_sklearn import NystroemGPSurrogate
from multioutreg.surrogates.gpx_smt import GPXSurrogate
from multioutreg.surrogates.kpls_smt import KPLSSurrogate
from multioutreg.surrogates.ard_gp_sklearn import ARDGPSurrogate
from multioutreg.surrogates.hist_gradient_boosting_sklearn import HistGradientBoostingSurrogate
from multioutreg.surrogates.lightgbm_sklearn import LightGBMSurrogate
from multioutreg.surrogates.xgboost_sklearn import XGBoostSurrogate
from multioutreg.surrogates.catboost_sklearn import CatBoostSurrogate
from multioutreg.surrogates.elastic_net_sklearn import ElasticNetSurrogate, LassoSurrogate
from multioutreg.surrogates.deep_ensemble_pytorch import DeepEnsembleSurrogate
from multioutreg.surrogates.sparse_gp_gpytorch import SparseGPSurrogate
from multioutreg.surrogates.quantile_sklearn import QuantileRegressionSurrogate

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
    "StackedVFMSurrogate",
    "AdditiveCorrectionVFM",
    "MixtureOfExpertsSurrogate",
    "BayesianRidgeSurrogate",
    "RFFGPSurrogate",
    "PolynomialBayesianRidgeSurrogate",
    "NystroemGPSurrogate",
    "GPXSurrogate",
    "KPLSSurrogate",
    "ARDGPSurrogate",
    "HistGradientBoostingSurrogate",
    "LightGBMSurrogate",
    "XGBoostSurrogate",
    "CatBoostSurrogate",
    "ElasticNetSurrogate",
    "LassoSurrogate",
    "DeepEnsembleSurrogate",
    "SparseGPSurrogate",
    "QuantileRegressionSurrogate",
]

