# Copyright (c) 2025 takotime808

"""Uncertainty-aware model wrappers and HTML report used by the grid-search app and CLI.

Kept free of Streamlit calls so it can be imported without running the app.
"""

import io
import json
import os

import joblib
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from scipy.stats import norm
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from jinja2 import Template


# from multioutreg.gui.report_plotting_utils import (
#     # plot_to_b64,
#     generate_prediction_plot,
#     generate_shap_plot,
#     generate_pdp_plot,
#     generate_uncertainty_plots,
#     generate_umap_plot,
#     generate_error_histogram
#     generate_pca_variance_plot,
# )

from multioutreg.figures.error_histograms import generate_error_histogram
from multioutreg.figures.shap_multioutput import generate_shap_plot
from multioutreg.figures.pca_plots import generate_pca_variance_plot
from multioutreg.figures.pdp_plots import generate_pdp_plot
from multioutreg.utils.figure_utils import safe_plot_b64
from multioutreg.figures.umap_plot_classify import generate_umap_plot
from multioutreg.report.build import build_report_data
from multioutreg.report.render import render_report, resolve_style
from multioutreg.figures.prediction_plots import plot_predictions_with_error_bars
# from multioutreg.figures.shap_multioutput import plot_multioutput_shap_bar_subplots
from multioutreg.figures.coverage_plots import plot_coverage
from multioutreg.figures.residuals import plot_residuals_multioutput_with_regplot
# from multioutreg.figures.prediction_plots import plot_predictions
from multioutreg.figures.confidence_intervals import plot_intervals_ordered_multi
from multioutreg.figures.conformal_plots import (
    plot_conformal_intervals_ordered,
    plot_conformal_vs_gaussian,
)
from multioutreg.surrogates import MultiFidelitySurrogate, LinearRegressionSurrogate
from multioutreg.surrogates.conformal_network_sklearn import ConformalPredictionNetworkSurrogate
from multioutreg.surrogates.rfgp_sklearn import _RFFEstimator
from multioutreg.surrogates.polynomial_bayesian_ridge_sklearn import _PBREstimator
from multioutreg.surrogates.nystroem_gp_sklearn import _NystroemEstimator
from multioutreg.surrogates.ard_gp_sklearn import _ARDGPEstimator
from multioutreg.surrogates.gp_sklearn import default_gp_kernel, matern_gp_kernel
from multioutreg.model_selection.screening import ModelScreener

try:
    from ngboost import NGBRegressor as _NGBRegressor
    _NGBOOST_AVAILABLE = True
except ImportError:
    _NGBOOST_AVAILABLE = False

from multioutreg.surrogates.gpx_smt import _GPXEstimator, _GPX_AVAILABLE
from multioutreg.surrogates.kpls_smt import _KPLSEstimator, _KPLS_AVAILABLE
from multioutreg.surrogates.lightgbm_sklearn import _LIGHTGBM_AVAILABLE, _LGBMRegressor
from multioutreg.surrogates.xgboost_sklearn import _XGBOOST_AVAILABLE, _XGBRegressor
from multioutreg.surrogates.catboost_sklearn import _CATBOOST_AVAILABLE, _CatBoostRegressor
from multioutreg.utils.thread_limits import catboost_thread_kwargs
from multioutreg.surrogates.sparse_gp_gpytorch import _GPYTORCH_AVAILABLE, _SGPREstimator
from sklearn.linear_model import ElasticNet as _ElasticNet, Lasso as _Lasso, QuantileRegressor as _QuantileRegressor
from multioutreg.surrogates.forest_uncertainty import estimate_noise_variance
from multioutreg.time_series.ts_suitability import check_ts_suitability

# NOTE: NOT used...yet.
from multioutreg.figures.doe_plots import make_doe_plot
from multioutreg.figures.model_comparison import plot_surrogate_model_summary


# ----- Surrogate Models with Uncertainty -----
class RandomForestWithUncertainty(RandomForestRegressor):
    """Random forest whose std is tree spread + held-out noise (audit item 27)."""

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight=sample_weight)
        self.noise_var_ = estimate_noise_variance(self, X, y)
        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict using the random forest, optionally returning standard deviation.

        Parameters
        ----------
        X : np.ndarray
            Input feature array.
        return_std : bool, optional
            Whether to return the standard deviation of predictions.

        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            Mean predictions or (mean, std) tuple.
        """
        mean = super().predict(X)
        if not return_std:
            return mean
        all_preds = np.stack([tree.predict(X) for tree in self.estimators_], axis=0)
        std = np.sqrt(all_preds.var(axis=0) + getattr(self, "noise_var_", 0.0))
        return mean, std


class ExtraTreesWithUncertainty(ExtraTreesRegressor):
    """Extra-Trees whose std is tree spread + held-out noise (audit item 27)."""

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight=sample_weight)
        self.noise_var_ = estimate_noise_variance(self, X, y)
        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        mean = super().predict(X)
        if not return_std:
            return mean
        all_preds = np.stack([tree.predict(X) for tree in self.estimators_], axis=0)
        std = np.sqrt(all_preds.var(axis=0) + getattr(self, "noise_var_", 0.0))
        return mean, std


class GradientBoostingWithUncertainty(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.95, n_estimators=100):
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.lower = GradientBoostingRegressor(loss="quantile", alpha=(1 - alpha) / 2, n_estimators=n_estimators)
        self.upper = GradientBoostingRegressor(loss="quantile", alpha=1 - (1 - alpha) / 2, n_estimators=n_estimators)
        self.mid = GradientBoostingRegressor(loss="squared_error", n_estimators=n_estimators)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingWithUncertainty":
        """
        Fit the quantile and mid regressors.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target array.

        Returns
        -------
        GradientBoostingWithUncertainty
            The fitted model.
        """
        self.lower.fit(X, y)
        self.upper.fit(X, y)
        self.mid.fit(X, y)
        return self

    def predict(
            self,
            X: np.ndarray,
            return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict using the model.

        Parameters
        ----------
        X : np.ndarray
            Input feature array.
        return_std : bool, optional
            Whether to return standard deviation.

        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            Mean predictions or (mean, std) tuple.
        """
        y_pred = self.mid.predict(X)
        if not return_std:
            return y_pred
        lower = self.lower.predict(X)
        upper = self.upper.predict(X)
        # The (lower, upper) quantiles span a central alpha-interval, whose
        # half-width is z * sigma for Gaussian noise; divide by z to get sigma.
        z = norm.ppf(0.5 + self.alpha / 2)
        std = np.clip((upper - lower) / (2 * z), 0.0, None)  # quantile models can cross
        return y_pred, std


class KNeighborsRegressorWithUncertainty(KNeighborsRegressor):
    def predict(self, X: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict using KNN, optionally returning standard deviation.

        Parameters
        ----------
        X : np.ndarray
            Input feature array.
        return_std : bool, optional
            Whether to return standard deviation.

        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            Mean predictions or (mean, std) tuple.
        """
        mean = super().predict(X)
        if not return_std:
            return mean
        neigh_ind = self.kneighbors(X, return_distance=False)
        y_neigh = self._y[neigh_ind]
        std = y_neigh.std(axis=1)
        return mean, std


class BootstrapLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, n_bootstraps=20):
        self.n_bootstraps = n_bootstraps

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BootstrapLinearRegression":
        """
        Fit bootstrapped linear regression models.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target array.

        Returns
        -------
        BootstrapLinearRegression
            The fitted model.
        """
        self.models_ = []
        n = X.shape[0]
        for _ in range(self.n_bootstraps):
            idx = np.random.choice(n, n, replace=True)
            model = LinearRegression().fit(X[idx], y[idx])
            self.models_.append(model)
        return self

    def predict(
            self,
            X: np.ndarray,
            return_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict using ensemble of linear models.

        Parameters
        ----------
        X : np.ndarray
            Input feature array.
        return_std : bool, optional
            Whether to return standard deviation.

        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            Mean predictions or (mean, std) tuple.
        """
        all_preds = [m.predict(X).reshape(X.shape[0], -1) for m in self.models_]
        preds = np.stack(all_preds, axis=2)
        mean = preds.mean(axis=2)
        if not return_std:
            return mean
        std = preds.std(axis=2)
        return mean, std


class PerTargetRegressorWithStd(BaseEstimator, RegressorMixin):
    def __init__(self, estimators):
        self.estimators = list(estimators)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PerTargetRegressorWithStd":
        """
        Fit separate estimators for each output column.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Multi-output target matrix.

        Returns
        -------
        PerTargetRegressorWithStd
            The fitted model.
        """
        self.estimators_ = []
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        for est, col in zip(self.estimators, y.T):
            est_fitted = clone(est)
            est_fitted.fit(X, col)
            self.estimators_.append(est_fitted)
        return self

    def predict(
            self,
            X: np.ndarray,
            return_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict using individual estimators for each target dimension.

        Parameters
        ----------
        X : np.ndarray
            Input feature array.
        return_std : bool, optional
            Whether to return standard deviation.

        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            Mean predictions or (mean, std) tuple.
        """
        preds, stds = [], []
        for est in self.estimators_:
            if return_std:
                try:
                    pred, std = est.predict(X, return_std=True)
                except TypeError:
                    pred = est.predict(X)
                    std = np.full(pred.shape, np.nan)
                preds.append(pred.reshape(-1, 1))
                stds.append(std.reshape(-1, 1))
            else:
                pred = est.predict(X)
                preds.append(np.asarray(pred).reshape(-1, 1))
        if return_std:
            return np.hstack(preds), np.hstack(stds)
        return np.hstack(preds)


# ----- Candidate models and per-output selection -----
def _scaled(est):
    """StandardScaler + est; ``predict(..., return_std=True)`` passes through."""
    return make_pipeline(StandardScaler(), est)


def _scaled_factory(cls, **fixed):
    return lambda **params: _scaled(cls(**fixed, **params))


def _threaded_catboost(**params):
    """CatBoost with ``thread_count`` capped to the container's CPU quota."""
    return _CatBoostRegressor(**catboost_thread_kwargs(), **params)


#: Slow candidates skipped by default on the hosted app and in the CLI,
#: with display names.
EXPENSIVE_SURROGATES = {
    "gpr": "Gaussian Process",
    "ngb": "NGBoost",
    "cpn": "Conformal Prediction Network",
    "ard_gp": "ARD Gaussian Process",
    "gpx": "GPX (Rust GP)",
    "kpls": "KPLS (Kriging+PLS)",
    "sgpr": "Sparse GP (GPyTorch)",
}


def default_surrogate_defs() -> List[Tuple[str, Any, dict]]:
    """``(name, factory, param_grid)`` for every candidate the grid search tries.

    Distance/kernel/penalty-based models (GP, KNN, SVR, ElasticNet, Lasso) are
    standardised pipelines; the GP has a learned amplitude and noise term.
    """
    defs = [
        ("gpr", _scaled_factory(GaussianProcessRegressor, normalize_y=True),
         {"kernel": [default_gp_kernel(), matern_gp_kernel()]}),
        ("rf", RandomForestWithUncertainty, {"n_estimators": [50], "max_depth": [3, None]}),
        ("et", ExtraTreesWithUncertainty, {"n_estimators": [50], "max_depth": [3, None]}),
        ("gb", GradientBoostingWithUncertainty, {"alpha": [0.95], "n_estimators": [50]}),
        ("hgb", HistGradientBoostingRegressor, {"max_iter": [100, 200], "learning_rate": [0.05, 0.1]}),
        ("knn", _scaled_factory(KNeighborsRegressorWithUncertainty), {"n_neighbors": [3, 5, 7]}),
        ("blr", BootstrapLinearRegression, {"n_bootstraps": [20]}),
        ("svr", _scaled_factory(SVR), {"C": [1.0, 10.0], "gamma": ["scale", "auto"]}),
        ("dt", DecisionTreeRegressor, {"max_depth": [1, None]}),
        ("cpn", ConformalPredictionNetworkSurrogate,
         {"hidden_layer_sizes": [(64,), (128,)], "max_iter": [500], "random_state": [0]}),
        # Plain linear baseline. (Was "mfs_lr", a one-level MultiFidelitySurrogate
        # whose fit((X, y)) signature made it fail on every output.)
        ("linear", LinearRegression, {}),
        ("bayesian_ridge", BayesianRidge, {}),
        ("rfgp", _RFFEstimator, {"n_components": [100, 500], "length_scale": [0.1, 1.0, 10.0], "kernel": ["rbf", "matern52"]}),
        ("pbr", _PBREstimator, {"degree": [2, 3], "interaction_only": [False, True]}),
        ("sgp", _NystroemEstimator, {"n_components": [50, 200], "gamma": [None, 0.1, 1.0]}),
        ("ard_gp", _ARDGPEstimator, {"alpha": [1e-6, 1e-2]}),
        ("elastic_net", _scaled_factory(_ElasticNet), {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.25, 0.5, 0.75]}),
        ("lasso", _scaled_factory(_Lasso), {"alpha": [0.01, 0.1, 1.0]}),
        ("quantile", _QuantileRegressor, {"alpha": [0.1, 1.0]}),
    ]
    if _NGBOOST_AVAILABLE:
        defs.append(("ngb", _NGBRegressor, {"n_estimators": [100, 200], "verbose": [False]}))
    if _GPX_AVAILABLE:
        defs.append(("gpx", _GPXEstimator, {"corr": ["squar_exp", "matern52"], "poly": ["constant", "linear"]}))
    if _KPLS_AVAILABLE:
        defs.append(("kpls", _KPLSEstimator, {"n_comp": [2, 4], "corr": ["squar_exp", "matern52"]}))
    if _LIGHTGBM_AVAILABLE:
        defs.append(("lgbm", _LGBMRegressor, {"n_estimators": [100, 200], "learning_rate": [0.01, 0.05], "num_leaves": [31, 63]}))
    if _XGBOOST_AVAILABLE:
        defs.append(("xgb", _XGBRegressor, {"n_estimators": [100, 200], "learning_rate": [0.01, 0.05], "max_depth": [4, 6]}))
    if _CATBOOST_AVAILABLE:
        defs.append(("catboost", _threaded_catboost, {"n_estimators": [100, 200], "learning_rate": [0.01, 0.05], "depth": [4, 6], "verbose": [False]}))
    if _GPYTORCH_AVAILABLE:
        defs.append(("sgpr", _SGPREstimator, {"n_inducing": [30, 50], "learning_rate": [0.05, 0.1]}))
    return defs


@dataclass
class GridSearchResult:
    """Outcome of :func:`select_per_target`."""
    model: Any                   # fitted on all of X (PerTargetRegressorWithStd or MoE)
    combo: List[dict]            # winning config per output (or the MoE config)
    val_mse: List[float]         # mean CV (or validation) MSE of the winner, per output
    moe_selected: bool = False
    n_failed: int = 0            # (config, output) fits that raised
    cv_description: str = ""     # how the configs were scored, for reports
    # One row per config for reports: CV MSE / RMSE averaged over outputs, the
    # fold spread of the RMSE, and the outputs it was selected for.
    cv_table: List[dict] = field(default_factory=list)


def select_per_target(
    surrogate_defs: List[Tuple[str, Any, dict]],
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: Optional[float] = None,
    random_state: int = 0,
    moe: Any = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    cv: Any = "auto",
) -> GridSearchResult:
    """Pick the best config per output by cross-validated MSE on the training data.

    Every ``(name, factory, params)`` config is scored by its mean MSE over the
    folds of ``cv`` (resolved by sample size, as in
    :func:`~multioutreg.model_selection.auto_detect.resolve_cv`), independently
    per output (outputs do not interact in a ``PerTargetRegressorWithStd``, so
    this finds the same winner as trying every combination). The winners are
    then refit on all of (X, y). The test set is never seen, so metrics on it
    are unbiased. Passing ``val_fraction`` instead scores on a single validation
    split of that size (the old behaviour; high-variance at small n).

    ``moe``, an unfitted joint multi-output model, replaces the per-output
    winners if its mean CV MSE (averaged over outputs) is lower.

    ``on_progress(done, total, label)`` is called before each step (every
    config, the MoE candidate, and the final refit) so a UI can show progress.
    """
    from multioutreg.model_selection.auto_detect import describe_cv, resolve_cv

    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if val_fraction is not None:
        idx_fit, idx_val = train_test_split(
            np.arange(len(X)), test_size=val_fraction, random_state=random_state)
        splits = [(idx_fit, idx_val)]
        cv_description = f"a single {val_fraction:.0%} validation split"
    else:
        splitter = resolve_cv(cv, len(X), random_state)
        splits = list(splitter.split(X, y))
        cv_description = describe_cv(splitter)
    configs = [(name, factory, params) for name, factory, grid in surrogate_defs
               for params in ParameterGrid(grid)]

    n_out = y.shape[1]
    best: List[Any] = [None] * n_out
    best_mse = [np.inf] * n_out
    best_cfg = [-1] * n_out
    n_failed = 0
    # fold_scores[i][j] = per-fold MSE of config i on output j (None if it failed)
    fold_scores: List[List[Any]] = []
    total = len(configs) + (moe is not None) + 1

    def _report(done: int, label: str) -> None:
        if on_progress is not None:
            on_progress(done, total, label)

    for i, (name, factory, params) in enumerate(configs):
        _report(i, f"Fitting {name}" + (f" {params}" if params else ""))
        fold_scores.append([None] * n_out)
        for j in range(n_out):
            try:
                fold_mse = []
                for tr, te in splits:
                    est = factory(**params)
                    est.fit(X[tr], y[tr, j])
                    pred = np.asarray(est.predict(X[te])).reshape(-1)
                    fold_mse.append(mean_squared_error(y[te, j], pred))
                mse = float(np.mean(fold_mse))
            except Exception:
                n_failed += 1
                continue
            fold_scores[i][j] = fold_mse
            if mse < best_mse[j]:
                best_mse[j], best[j], best_cfg[j] = mse, (name, factory, params), i
    missing = [j for j in range(n_out) if best[j] is None]
    if missing:
        raise RuntimeError(f"Every candidate failed for output(s) {missing}")

    cv_table = _cv_table(configs, fold_scores, best_cfg)

    if moe is not None:
        _report(len(configs), f"Fitting {type(moe).__name__}")
        try:
            moe_folds = [
                mean_squared_error(
                    y[te], np.asarray(clone(moe).fit(X[tr], y[tr]).predict(X[te])))
                for tr, te in splits]
            moe_val = float(np.mean(moe_folds))
        except Exception as exc:
            moe_folds, moe_val = None, np.inf
            cv_table.append(_cv_row(type(moe).__name__, None, f"Error: {exc}"))
        if moe_folds is not None:
            cv_table.append(_cv_row(type(moe).__name__, moe_folds))
        if moe_val < np.mean(best_mse):
            for row in cv_table:
                row["Selected for"] = []
            cv_table[-1]["Selected for"] = list(range(n_out))
            _report(total - 1, "Refitting the selected model on all training data")
            return GridSearchResult(
                model=moe.fit(X, y), combo=[{"model": type(moe).__name__}],
                val_mse=[float(moe_val)] * n_out, moe_selected=True, n_failed=n_failed,
                cv_description=cv_description, cv_table=_sort_cv_table(cv_table))

    _report(total - 1, "Refitting the selected models on all training data")
    model = PerTargetRegressorWithStd(
        [factory(**params) for _, factory, params in best]).fit(X, y)
    return GridSearchResult(
        model=model,
        combo=[dict(model=name, **params) for name, _, params in best],
        val_mse=[float(m) for m in best_mse],
        n_failed=n_failed,
        cv_description=cv_description,
        cv_table=_sort_cv_table(cv_table),
    )


def _cv_row(label: str, fold_mse: Optional[List[float]], status: str = "OK") -> dict:
    """A report row from per-fold MSEs (already averaged over outputs)."""
    if fold_mse is None:
        return {"Model": label, "CV MSE": None, "CV RMSE": None, "CV RMSE std": None,
                "CV R²": None, "Status": status, "Selected for": []}
    fold_mse = np.asarray(fold_mse, dtype=float)
    return {
        "Model": label,
        "CV MSE": round(float(fold_mse.mean()), 6),
        "CV RMSE": round(float(np.sqrt(fold_mse.mean())), 6),
        "CV RMSE std": round(float(np.std(np.sqrt(fold_mse))), 6),
        "CV R²": None,
        "Status": status,
        "Selected for": [],
    }


def _cv_table(configs, fold_scores, best_cfg) -> List[dict]:
    """One row per config: fold MSEs averaged over the outputs it fitted."""
    rows = []
    for i, (name, _, params) in enumerate(configs):
        label = name + (f" {params}" if params else "")
        ok = [f for f in fold_scores[i] if f is not None]
        if not ok:
            rows.append(_cv_row(label, None, "Error: failed for every output"))
            continue
        status = "OK" if len(ok) == len(fold_scores[i]) else (
            f"Failed for {len(fold_scores[i]) - len(ok)} output(s)")
        row = _cv_row(label, np.mean(np.asarray(ok, dtype=float), axis=0), status)
        row["Selected for"] = [j for j, b in enumerate(best_cfg) if b == i]
        rows.append(row)
    return rows


def _sort_cv_table(rows: List[dict]) -> List[dict]:
    return sorted(rows, key=lambda r: (r["CV MSE"] is None, r["CV MSE"] or 0.0))


class PerTargetGridSearch:
    """Estimator form of :func:`select_per_target`: ``fit`` runs the selection.

    Lets the whole selection be refitted per fold by nested CV and CV+
    (:mod:`multioutreg.model_selection.evaluation`). ``moe_factory`` builds a
    fresh joint candidate for each fit. ``pre_screen=True`` runs
    :class:`~multioutreg.model_selection.screening.ModelScreener` on the rows
    being fitted and drops candidates that fail it for every output. After
    ``fit``, ``result_`` holds the :class:`GridSearchResult` and ``skipped_``
    the screened-out candidate names.
    """

    def __init__(self, surrogate_defs=None, moe_factory=None, cv="auto",
                 random_state=0, on_progress=None, pre_screen=False):
        self.surrogate_defs = surrogate_defs
        self.moe_factory = moe_factory
        self.cv = cv
        self.random_state = random_state
        self.on_progress = on_progress
        self.pre_screen = pre_screen

    def fit(self, X, y) -> "PerTargetGridSearch":
        defs = self.surrogate_defs if self.surrogate_defs is not None else default_surrogate_defs()
        self.skipped_ = []
        if self.pre_screen:
            from multioutreg.model_selection.screening import ModelScreener
            eligible = ModelScreener().fit(X, y).screen_per_output()
            self.skipped_ = [d[0] for d in defs
                             if d[0] in eligible.index and not eligible.loc[d[0]].any()]
            defs = [d for d in defs if d[0] not in self.skipped_]
        moe = self.moe_factory() if self.moe_factory is not None else None
        self.result_ = select_per_target(
            defs, X, y, random_state=self.random_state, moe=moe,
            on_progress=self.on_progress, cv=self.cv)
        return self

    def predict(self, X, return_std: bool = False):
        return self.result_.model.predict(X, return_std=return_std)


# ----- Exportable model -----
class InputColumnSelector(BaseEstimator, TransformerMixin):
    """First pipeline step: pick the training input columns, in training order.

    Accepts a DataFrame (extra columns are ignored, missing ones raise) or an
    array whose columns are already in ``columns`` order.
    """

    def __init__(self, columns):
        self.columns = list(columns)

    def fit(self, X=None, y=None) -> "InputColumnSelector":
        self.columns_ = list(self.columns)
        return self

    def transform(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.columns_ if c not in X.columns]
            if missing:
                raise KeyError(f"Input is missing column(s) {missing}")
            return X[self.columns_].to_numpy(dtype=float)
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.columns_):
            raise ValueError(
                f"Expected {len(self.columns_)} input columns {self.columns_}, "
                f"got array of shape {X.shape}")
        return X


def build_model_pipeline(
    model: Any,
    input_cols: List[str],
    imputer: Any = None,
    input_transformer: Any = None,
) -> Pipeline:
    """Chain the already-fitted preprocessing steps and ``model`` into one Pipeline.

    Steps, in the order the app applies them: ``columns``
    (:class:`InputColumnSelector`), ``imputer`` (KNN, if given), ``pca``
    (standardise + PCA, if given) and ``model``. Nothing is refitted.
    ``pipeline.predict(X, return_std=True)`` is passed through to the model.
    """
    steps = [("columns", InputColumnSelector(input_cols).fit())]
    if imputer is not None:
        steps.append(("imputer", imputer))
    if input_transformer is not None:
        steps.append(("pca", input_transformer))
    steps.append(("model", model))
    return Pipeline(steps)


def _package_versions() -> Dict[str, str]:
    import sys
    from importlib.metadata import PackageNotFoundError, version

    versions = {"python": sys.version.split()[0]}
    for pkg in ("multioutreg", "scikit-learn", "numpy", "joblib"):
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "unknown"
    return versions


def build_model_bundle(
    pipeline: Pipeline,
    input_cols: List[str],
    output_cols: List[str],
    combo: List[dict],
    pca_settings: Optional[dict] = None,
    conformal_alpha: Optional[float] = None,
    conformal_quantiles: Optional[np.ndarray] = None,
    data_split: Optional[dict] = None,
) -> Dict[str, Any]:
    """Everything needed to reuse (``pipeline``) or rebuild (``config``) the model.

    ``config`` is JSON-safe (non-JSON values such as GP kernels become their
    ``repr``) and is what the app offers as the config download. Conformal
    intervals are ``pipeline.predict(X) -/+ conformal["quantiles"][output]``.
    """
    conformal = None
    if conformal_alpha is not None and conformal_quantiles is not None:
        conformal = {
            "alpha": float(conformal_alpha),
            "quantiles": {name: float(q) for name, q in zip(output_cols, conformal_quantiles)},
        }
    imputer = pipeline.named_steps.get("imputer")
    config = {
        "input_columns": list(input_cols),
        "output_columns": list(output_cols),
        "preprocessing": {
            "imputer": None if imputer is None else {
                "type": type(imputer).__name__, "n_neighbors": imputer.n_neighbors},
            "pca": pca_settings,
        },
        "models": combo,
        "conformal": conformal,
        "data_split": data_split,
        "versions": _package_versions(),
    }
    config = json.loads(json.dumps(config, default=repr))
    return {
        "pipeline": pipeline,
        "input_columns": list(input_cols),
        "output_columns": list(output_cols),
        "conformal": conformal,
        "config": config,
        "versions": config["versions"],
    }


def serialize_model_bundle(bundle: Dict[str, Any]) -> bytes:
    """``joblib.dump`` the bundle into bytes, e.g. for a download button."""
    buf = io.BytesIO()
    joblib.dump(bundle, buf, compress=3)
    return buf.getvalue()


# ----- HTML Report Generation Wrapper -----
def generate_html_report(
    model_type: str,
    fidelity_levels: List[str],
    output_names: List[str],
    description: str,
    metrics: Dict[str, Dict[str, float]],
    uncertainty_metrics: Dict[str, float],
    y_test: np.ndarray,
    best_pred: np.ndarray,
    best_std: np.ndarray,
    best_model: Any,
    X_train: np.ndarray,
    n_train: int,
    n_test: int,
    cross_validation: str,
    seed: int,
    notes: str,
    feature_names: List[str] | None = None,
    feature_names_pca: List[str] | None = None,
    pca_explained_variance: List[float] | None = None,
    pca_variance_plot: str | None = None,
    pca_method: str | None = None,
    pca_threshold: float | None = None,
    pca_n_components: int | None = None,
    kaiser_rule_suggestion: str | None = None,
    conformal_intervals: tuple | None = None,
    conformal_alpha: float | None = None,
    calibration_residuals: np.ndarray | None = None,
    imputation_summary: dict | None = None,
    grid_search_results: List[dict] | None = None,
    report_style: str | None = None,
) -> str:
    """
    Generate an HTML report from model training and evaluation results.

    Parameters
    ----------
    model_type : str
        Type of model used.
    fidelity_levels : List[str]
        Fidelity levels used (if any).
    output_names : List[str]
        Names of the output dimensions.
    description : str
        Description of the modeling project.
    metrics : Dict[str, Dict[str, float]]
        Dictionary of evaluation metrics for each output.
    uncertainty_metrics : Dict[str, float]
        Dictionary of uncertainty metrics.
    y_test : np.ndarray
        Ground truth values for test set.
    best_pred : np.ndarray
        Predictions from the best model.
    best_std : np.ndarray
        Uncertainty estimates from the best model.
    best_model : Any
        The best performing model.
    X_train : np.ndarray
        Training input features.
    n_train : int
        Number of training samples.
    n_test : int
        Number of test samples.
    cross_validation : str
        Description of the CV strategy.
    seed : int
        Random seed used.
    notes : str
        Additional notes.
    feature_names : List[str] | None, optional
        Names of the columns of ``X_train`` (used to label the PDPs). Defaults
        to ``feature_names_pca`` when given, else ``feature_0``, ``feature_1``, ...
    feature_names_pca : List[str] | None, optional
        Names to use for PCA.
    pca_explained_variance : List[float] | None, optional
        Explained variance ratios from PCA if applied.
    pca_variance_plot : str | None, optional
        Base64-encoded plot showing PCA variance ratios.
    pca_method : str | None, optional
        Method used to choose the number of PCA components.
    pca_threshold : float | None, optional
        Explained variance threshold if that method was used.
    pca_n_components : int | None, optional
        Final number of PCA components retained.
    kaiser_rule_suggestion: str | None, optional
        PCA explanation statement.
    conformal_intervals : tuple | None, optional
        ``(y_lower, y_upper)`` conformal intervals on the test set.
    conformal_alpha : float | None, optional
        Miscoverage level of ``conformal_intervals``.
    calibration_residuals : np.ndarray | None, optional
        ``|y_cal - y_cal_pred|`` from the held-out conformal calibration set.
        The Gaussian-vs-conformal coverage plot is drawn only when given.
    imputation_summary : dict | None, optional
        Rows before/after cleaning and the action taken per column.
    grid_search_results : List[dict] | None, optional
        ``GridSearchResult.cv_table``: one row per candidate config, shown in
        the v2 report's model-selection section.
    report_style : str | None, optional
        ``"v2"`` (the redesigned report, default) or ``"classic"``. Defaults to
        the ``MOR_REPORT_STYLE`` environment variable, then ``"v2"``.

    Returns
    -------
    str
        Rendered HTML report.
    """
    if resolve_style(report_style) == "v2":
        if feature_names is None or len(feature_names) != X_train.shape[1]:
            feature_names = feature_names_pca
        if feature_names is None or len(feature_names) != X_train.shape[1]:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        sampling_plot, sampling_explanation = generate_umap_plot(X_train)
        data = build_report_data(
            title="Surrogate Modeling Report",
            model_type=model_type, output_names=list(output_names), description=description,
            notes=notes, metrics=metrics, uncertainty_metrics=uncertainty_metrics,
            y_test=y_test, best_pred=best_pred, best_std=best_std, best_model=best_model,
            X_train=X_train, input_names=list(feature_names), n_train=n_train, n_test=n_test,
            cross_validation=cross_validation, seed=seed, fidelity_levels=fidelity_levels,
            conformal_intervals=conformal_intervals, conformal_alpha=conformal_alpha,
            calibration_residuals=calibration_residuals,
            imputation_summary=imputation_summary, grid_search_results=grid_search_results,
            pca_explained_variance=pca_explained_variance, pca_variance_plot=pca_variance_plot,
            pca_method=pca_method, pca_threshold=pca_threshold,
            pca_n_components=pca_n_components, kaiser_rule_suggestion=kaiser_rule_suggestion,
            sampling_plot=sampling_plot, sampling_explanation=sampling_explanation,
        )
        return render_report(data)

    # prediction_plots = generate_prediction_plot(y_test, best_pred, best_std, output_names)
    # # pdp_plots = generate_pdp_plot(output_names)
    # pdp_plots = generate_pdp_plot(best_model, X_train, output_names, feature_names=input_cols)
    # uncertainty_plots = generate_uncertainty_plots()

    prediction_plots = {}
    prediction_plots["all_in_one"] = safe_plot_b64(
        plot_intervals_ordered_multi,
        best_pred,
        best_std,
        y_test,
        # max_cols=3,
        target_list=output_names,
    )
    # # PREDICTION PLOTS OPTION 2
    # prediction_plots = {}
    for i, name in enumerate(output_names):
        prediction_plots[name] = safe_plot_b64(
            plot_predictions_with_error_bars,
            y_test[:, [i]],
            best_pred[:, [i]],
            best_std[:, [i]],
            output_names=[name],
            n_cols=3
        )

    shap_plots = generate_shap_plot(best_model, X_train, output_names)
    # shap_img = safe_plot_b64(
    #     plot_multioutput_shap_bar_subplots,
    #     best_model, X_train,
    #     feature_names=input_cols, output_names=output_names
    # )
    # shap_plots = {name: shap_img for name in output_names}

    unc_img = safe_plot_b64(
        plot_coverage, y_test, best_pred, best_std, output_names=output_names
    )
    # uncertainty_plots = [{"img_b64": unc_img, "title": "Coverage Plot", "caption": "Nominal vs empirical coverage."}]
    uncertainty_plots = [
        {
            "img_b64": unc_img,
            "title": "Coverage Plot",
            "caption": "Nominal vs empirical coverage.",
        }
    ]

    # PDP labels are the *input* columns; sklearn rejects a list whose length
    # differs from X_train.shape[1], which made every PDP "not supported".
    if feature_names is None or len(feature_names) != X_train.shape[1]:
        feature_names = feature_names_pca
    if feature_names is None or len(feature_names) != X_train.shape[1]:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    pdp_plots = generate_pdp_plot(best_model, X_train, output_names, feature_names=list(feature_names))
    
    sampling_umap_plot, sampling_method_explanation = generate_umap_plot(X_train)
    # other_plots = generate_error_histogram(y_test, best_pred, output_names)
    other_img = safe_plot_b64(
        plot_residuals_multioutput_with_regplot,
        best_pred,
        y_test,
        target_list=output_names,
    )
    sampling_other_plots = [
        {
            "img_b64": other_img,
            "title": "Residuals",
            "caption": "Residual vs predicted values.",
        }
    ]

    other_plots = generate_error_histogram(y_test, best_pred, output_names)

    # Conformal prediction plots
    conformal_plots_list = []
    if conformal_intervals is not None and conformal_alpha is not None:
        y_lower, y_upper = conformal_intervals
        alpha = conformal_alpha

        conformal_ordered_img = safe_plot_b64(
            plot_conformal_intervals_ordered,
            y_test, y_lower, y_upper,
            y_pred=best_pred,
            output_names=output_names,
            alpha=alpha,
        )
        conformal_plots_list.append({
            "img_b64": conformal_ordered_img,
            "title": f"Conformal Prediction Intervals ({int((1-alpha)*100)}%)",
            "caption": "Distribution-free prediction intervals ordered by observed value. Guaranteed marginal coverage.",
        })

        # Conformal curves must use calibration residuals; test residuals
        # would match the diagonal by construction.
        if calibration_residuals is not None:
            conformal_vs_gauss_img = safe_plot_b64(
                plot_conformal_vs_gaussian,
                y_test, best_pred, best_std,
                calibration_residuals,
                output_names=output_names,
            )
            conformal_plots_list.append({
                "img_b64": conformal_vs_gauss_img,
                "title": "Gaussian CI vs Conformal PI Coverage",
                "caption": "Comparison of nominal vs empirical coverage for Gaussian confidence intervals and conformal prediction intervals.",
            })

    template_path = os.path.join(os.path.dirname(__file__), "../report/template.html")
    with open(template_path, "r", encoding="utf-8") as f:
        template_text = f.read()
    template = Template(template_text)
    rendered = template.render(
        project_title="Multi-Fidelity Surrogate Modeling Report",
        model_type=model_type,
        fidelity_levels=fidelity_levels,
        output_names=output_names,
        description=description,
        metrics=metrics,
        uncertainty_metrics=uncertainty_metrics,
        uncertainty_plots=uncertainty_plots,
        prediction_plots=prediction_plots,
        shap_plots=shap_plots,
        pdp_plots=pdp_plots,
        sampling_umap_plot=sampling_umap_plot,
        sampling_method_explanation=sampling_method_explanation,
        sampling_other_plots=sampling_other_plots,
        other_plots=other_plots,
        n_train=n_train,
        n_test=n_test,
        cross_validation=cross_validation,
        seed=seed,
        notes=notes,
        feature_names_pca=feature_names_pca,
        pca_explained_variance=pca_explained_variance,
        pca_variance_plot=pca_variance_plot,
        pca_method=pca_method,
        pca_threshold=pca_threshold,
        pca_n_components=pca_n_components,
        kaiser_rule_suggestion=kaiser_rule_suggestion,
        conformal_plots=conformal_plots_list,
        imputation_summary=imputation_summary,
    )
    return rendered
