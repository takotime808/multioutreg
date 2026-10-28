# Copyright (c) 2025 takotime808

"""Utilities for automatically selecting the best regressor per output."""

from __future__ import annotations

import functools
import numbers
from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    LeaveOneOut,
    RepeatedKFold,
    check_cv,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

from multioutreg.surrogates import (
    LinearRegressionSurrogate,
    GaussianProcessSurrogate,
    RandomForestSurrogate,
    GradientBoostingSurrogate,
    SVRSurrogate,
    KNeighborsSurrogate,
    DecisionTreeRegressorSurrogate,
    ConformalPredictionNetworkSurrogate,
    ExtraTreesRegressorSurrogate,
    NGBoostSurrogate,
    MultiFidelitySurrogate,
    BayesianRidgeSurrogate,
    RFFGPSurrogate,
    PolynomialBayesianRidgeSurrogate,
    NystroemGPSurrogate,
    GPXSurrogate,
    KPLSSurrogate,
    ARDGPSurrogate,
    HistGradientBoostingSurrogate,
    LightGBMSurrogate,
    XGBoostSurrogate,
    CatBoostSurrogate,
    ElasticNetSurrogate,
    LassoSurrogate,
    SparseGPSurrogate,
    QuantileRegressionSurrogate,
)
from multioutreg.surrogates.rfgp_sklearn import _RFFEstimator
from multioutreg.surrogates.polynomial_bayesian_ridge_sklearn import _PBREstimator
from multioutreg.surrogates.nystroem_gp_sklearn import _NystroemEstimator
from multioutreg.surrogates.ard_gp_sklearn import _ARDGPEstimator
from multioutreg.surrogates.conformal_network_sklearn import _CPNEstimator
from multioutreg.surrogates.gp_sklearn import default_gp_kernel, matern_gp_kernel

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
from multioutreg.surrogates.sparse_gp_gpytorch import _GPYTORCH_AVAILABLE

from sklearn.linear_model import (
    ElasticNet as _ElasticNet,
    Lasso as _Lasso,
    QuantileRegressor as _QuantileRegressor,
)

# Scorers that are undefined on a single held-out sample (R² needs a variance),
# so LeaveOneOut cannot be used with them.
_NEEDS_MULTI_SAMPLE_FOLDS = {"r2", "explained_variance"}


def select_cv_strategy(n_samples: int, random_state: int | None = 0):
    """Return a CV splitter appropriate for a dataset of *n_samples* rows.

    Small datasets need low-variance error estimates and cannot spare data for
    large validation folds; large datasets make many folds expensive and do
    not need them.

    - ``RepeatedKFold(n_splits=max(2, min(5, n // 4)), n_repeats=3)``
      when ``n_samples <= 100`` (at least ~4 rows per fold)
    - shuffled ``KFold(n_splits=10)`` when ``100 < n_samples <= 1000``
    - shuffled ``KFold(n_splits=5)``  when ``n_samples > 1000``

    Leave-one-out is not used by default: it is inconsistent for model
    selection (Shao 1993), 10-fold is preferred to more folds even when compute
    allows (Kohavi 1995), and selection should rest on repeated rather than
    single CV runs (Krstajic et al. 2014). Pass ``cv="loocv"`` to force it.
    The fold counts and the 3 repeats are conventions; see
    ``docs/AUDIT_CHECKLIST.md`` (S3).
    """
    if n_samples <= 100:
        return RepeatedKFold(n_splits=max(2, min(5, n_samples // 4)), n_repeats=3,
                             random_state=random_state)
    if n_samples <= 1000:
        return KFold(n_splits=10, shuffle=True, random_state=random_state)
    return KFold(n_splits=5, shuffle=True, random_state=random_state)


def resolve_cv(cv, n_samples: int, random_state: int | None = 0, scoring=None):
    """Resolve a ``cv`` setting to a concrete sklearn splitter.

    Parameters
    ----------
    cv : {"auto", "loocv"} | int | CV splitter
        ``"auto"`` picks by dataset size via :func:`select_cv_strategy`
        (never leave-one-out, so it suits every scorer);
        ``"loocv"`` always uses ``LeaveOneOut``; an int means a *shuffled*
        ``KFold`` with that many splits (sklearn's int default is unshuffled,
        which gives unrepresentative folds on ordered data); anything else is
        passed through ``check_cv``.
    n_samples : int
        Number of rows the splitter will be applied to.
    random_state : int | None
        Seed for shuffled / repeated splitters.
    scoring : str | None
        Scorer name. R²-type scorers are undefined on single-sample folds, so
        ``"loocv"`` raises for them.
    """
    single_ok = scoring not in _NEEDS_MULTI_SAMPLE_FOLDS
    if isinstance(cv, str):
        if cv == "auto":
            return select_cv_strategy(n_samples, random_state)
        if cv == "loocv":
            if not single_ok:
                raise ValueError(f"cv='loocv' cannot be used with scoring={scoring!r}: "
                                 "the score is undefined on a single held-out sample")
            return LeaveOneOut()
        raise ValueError(f"cv must be 'auto', 'loocv', an int or a CV splitter, got {cv!r}")
    if isinstance(cv, numbers.Integral):
        return KFold(cv, shuffle=True, random_state=random_state)
    return check_cv(cv)


def describe_cv(cv) -> str:
    """Human-readable description of a resolved CV splitter (for reports)."""
    if isinstance(cv, LeaveOneOut):
        return "Leave-One-Out CV"
    if isinstance(cv, RepeatedKFold):
        return f"Repeated {cv.cvargs['n_splits']}-Fold CV (x{cv.n_repeats} repeats)"
    if isinstance(cv, KFold):
        return f"{cv.n_splits}-Fold CV" + (" (shuffled)" if cv.shuffle else "")
    if isinstance(cv, numbers.Integral):
        return f"{cv}-Fold CV"
    return type(cv).__name__


def _clone_surrogate(surrogate):
    """Return an unfitted copy of a surrogate (sklearn clone, else deepcopy)."""
    try:
        return clone(surrogate)
    except Exception:
        import copy
        return copy.deepcopy(surrogate)


# Estimator class name -> screening rule key, for pre_screen on a custom
# estimator list (with_vendored_surrogates sets ``_model_names`` explicitly).
_SCREENING_NAMES = {
    "LinearRegression": "linear",
    "GaussianProcessRegressor": "gp",
    "RandomForestRegressor": "rf",
    "ExtraTreesRegressor": "et",
    "GradientBoostingRegressor": "gb",
    "HistGradientBoostingRegressor": "hgb",
    "SVR": "svr",
    "KNeighborsRegressor": "knn",
    "DecisionTreeRegressor": "dt",
    "MLPRegressor": "mlp",
    "_CPNEstimator": "mlp",
    "BayesianRidge": "bayesian_ridge",
    "ElasticNet": "elastic_net",
    "Lasso": "lasso",
    "QuantileRegressor": "quantile",
    "NGBRegressor": "ngboost",
    "LGBMRegressor": "lgbm",
    "XGBRegressor": "xgb",
    "CatBoostRegressor": "catboost",
    "_RFFEstimator": "rfgp",
    "_PBREstimator": "pbr",
    "_NystroemEstimator": "nystroem_gp",
    "_ARDGPEstimator": "ard_gp",
    "_GPXEstimator": "gpx",
    "_KPLSEstimator": "kpls",
    "_SGPREstimator": "sgpr",
}


def _screening_name(est) -> str:
    """Screening rule key for *est* (the final step of a Pipeline)."""
    if hasattr(est, "steps"):
        est = est.steps[-1][1]
    name = type(est).__name__
    return _SCREENING_NAMES.get(name, name)


# Complexity tier per screening name for selection="one_se" (0 = simplest).
# Names not listed count as 1; joint multi-output candidates count as 2.
_COMPLEXITY = {
    **dict.fromkeys(["linear", "bayesian_ridge", "elastic_net", "lasso",
                     "quantile", "dt", "knn", "pbr"], 0),
    **dict.fromkeys(["rf", "et", "gb", "hgb", "svr", "lgbm", "xgb", "catboost",
                     "rfgp", "nystroem_gp"], 1),
    **dict.fromkeys(["gp", "ard_gp", "gpx", "kpls", "sgpr", "mlp", "ngboost"], 2),
}


def _candidate(name: str, fold_scores, complexity: int,
               test_train_ratio: float | None = None, **payload) -> dict:
    """CV summary of one candidate with a Nadeau–Bengio corrected standard error.

    The naive SE, std/sqrt(k), assumes independent folds; K-fold training sets
    overlap, so the variance is inflated by (1/k + n_test/n_train) instead of 1/k.
    *test_train_ratio* defaults to 1/(k-1), which is exact for plain K-fold and
    leave-one-out; repeated K-fold has k = n_splits * n_repeats and must pass it.
    """
    s = np.asarray(fold_scores, dtype=float)
    k = len(s)
    if test_train_ratio is None and k > 1:
        test_train_ratio = 1 / (k - 1)
    se = float(np.sqrt((1 / k + test_train_ratio) * s.var(ddof=1))) if k > 1 else 0.0
    return {"name": name, "mean": float(s.mean()), "se": se,
            "complexity": complexity, **payload}


class _OutputColumn(RegressorMixin, BaseEstimator):
    """Expose column ``j`` of a multi-output model so sklearn scorers can use it."""

    def __init__(self, model, j: int) -> None:
        self.model = model
        self.j = j

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.asarray(self.model.predict(X))
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return preds[:, self.j]


class AutoDetectMultiOutputRegressor(BaseEstimator, RegressorMixin):
    """Fit a separate estimator per output choosing the best via grid search.

    Parameters
    ----------
    estimators : sequence of BaseEstimator
        Candidate sklearn estimators to evaluate.
    param_spaces : sequence of dict
        Parameter grids corresponding to each estimator.
    cv : {"auto", "loocv"} | int | CV splitter, default "auto"
        Cross-validation used to score every candidate. ``"auto"`` picks the
        scheme from the number of training rows at fit time (see
        :func:`select_cv_strategy`): repeated k-fold for small datasets,
        10-fold and then 5-fold as data grows.
        ``"loocv"`` forces leave-one-out, an int means a shuffled ``KFold``
        with that many splits, and any sklearn splitter is used as given.
    scoring : str
        Sklearn scoring metric for GridSearchCV.
    pre_screen : bool, default False
        When True, run statistical pre-screening tests before fitting each
        output and skip models that are either computationally expensive
        without justification or unlikely to improve on a linear baseline.
        See :class:`~multioutreg.model_selection.screening.ModelScreener`.
    selection : {"best", "one_se"}, default "best"
        ``"best"`` picks the highest mean CV score. ``"one_se"`` picks the
        least complex candidate whose mean score is within one standard error
        of the best (Nadeau–Bengio corrected, since CV folds overlap), so a
        costly model must beat a simple one by more than CV noise.
    random_state : int | None, default 0
        Seed for the shuffled / repeated splitters built from ``cv``.

    Attributes
    ----------
    cv_strategy_ : CV splitter
        The splitter ``cv`` resolved to for the training data.
    cv_description_ : str
        Human-readable form of ``cv_strategy_``, for reports.
    selection_scores_ : list of dict
        Per output, ``{candidate: {"mean": ..., "se": ...}}`` of the CV scores
        that drove the choice. The winner's mean is optimistically biased (it
        is the max over many candidates); evaluate on a held-out test set for
        an unbiased estimate.
    """

    def __init__(
        self,
        estimators: Sequence[BaseEstimator],
        param_spaces: Sequence[dict],
        cv: int | str = "auto",
        scoring: str = "neg_mean_squared_error",
        pre_screen: bool = False,
        selection: str = "best",
        random_state: int | None = 0,
    ) -> None:
        if len(estimators) != len(param_spaces):
            raise ValueError("Each estimator must have a corresponding param space")
        if selection not in ("best", "one_se"):
            raise ValueError(f"selection must be 'best' or 'one_se', got {selection!r}")
        # Stored unmodified: sklearn.clone checks constructor params by identity.
        self.estimators = estimators
        self.param_spaces = param_spaces
        self.cv = cv
        self.scoring = scoring
        self.pre_screen = pre_screen
        self.selection = selection
        self.random_state = random_state

    def __sklearn_clone__(self) -> "AutoDetectMultiOutputRegressor":
        # Configuration set outside __init__ (with_vendored_surrogates,
        # register_multi_output_candidates) is not a constructor param, so the
        # default clone would drop it.
        new = super().__sklearn_clone__()
        for attr in ("_model_names", "_surrogate_constructors", "fidelity_levels"):
            if hasattr(self, attr):
                setattr(new, attr, list(getattr(self, attr)))
        if hasattr(self, "_multi_output_candidates"):
            # fit() refits these in place, so clones must not share them.
            new._multi_output_candidates = [
                _clone_surrogate(s) for s in self._multi_output_candidates]
        return new

    def _splitter(self, n_samples: int):
        return resolve_cv(self.cv, n_samples, self.random_state, self.scoring)

    def _choose(self, candidates: list[dict]) -> dict:
        best = max(candidates, key=lambda c: c["mean"])
        if self.selection == "best":
            return best
        within = [c for c in candidates if c["mean"] >= best["mean"] - best["se"]]
        return min(within, key=lambda c: (c["complexity"], -c["mean"]))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AutoDetectMultiOutputRegressor":
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Run statistical pre-screening once across all outputs
        _screener = None
        if self.pre_screen:
            from multioutreg.model_selection.screening import ModelScreener
            _screener = ModelScreener().fit(X, y)
        names = getattr(self, "_model_names", None) or [
            _screening_name(est) for est in self.estimators]

        # One splitter for every candidate, so all scores use the same folds.
        splitter = self._splitter(len(X))
        self.cv_strategy_ = splitter
        self.cv_description_ = describe_cv(splitter)
        splits = list(splitter.split(X, y))
        ratio = float(np.mean([len(te) / len(tr) for tr, te in splits]))

        # Pre-evaluate joint multi-output surrogate candidates (models with
        # _multi_output = True that cannot be wrapped in MultiOutputRegressor).
        # These are scored per output so they can compete in the per-output loop.
        # They are scored with the same CV splitter and scorer as GridSearchCV so
        # that the comparison against best_score_ is out-of-sample on both sides.
        _mo_scores: list[tuple] = []  # (fold_scores (n_folds, n_outputs), fitted_surrogate)
        _mo_candidates = getattr(self, "_multi_output_candidates", [])
        if _mo_candidates:
            scorer = get_scorer(self.scoring)
        for mo_surrogate in _mo_candidates:
            fold_scores = np.zeros((len(splits), y.shape[1]))
            for f, (tr, te) in enumerate(splits):
                fold_model = _clone_surrogate(mo_surrogate).fit(X[tr], y[tr])
                for j in range(y.shape[1]):
                    fold_scores[f, j] = scorer(
                        _OutputColumn(fold_model, j), X[te], y[te, j]
                    )
            mo_surrogate.fit(X, y)
            _mo_scores.append((fold_scores, mo_surrogate))

        self.models_ = []
        self.selection_scores_: list[dict] = []
        # _model_output_col[i] = which column of models_[i].predict(X) gives output i.
        # Multi-output models store the full (n_samples, n_outputs) matrix and we
        # slice a single column per slot; single-output models always return column 0.
        self._model_output_col: list[int] = []
        for i in range(y.shape[1]):
            # Determine which estimator indices to evaluate for this output
            if _screener is not None:
                eligible = set(_screener.eligible_indices_for_output(i, names))
                if not eligible:  # screening ruled out every candidate: try them all
                    eligible = set(range(len(self.estimators)))
            else:
                eligible = set(range(len(self.estimators)))

            candidates = []
            for idx, (est, params) in enumerate(zip(self.estimators, self.param_spaces)):
                if idx not in eligible:
                    continue
                gs = GridSearchCV(est, params, cv=splits, scoring=self.scoring)
                gs.fit(X, y[:, i])
                folds = [gs.cv_results_[f"split{f}_test_score"][gs.best_index_]
                         for f in range(len(splits))]
                candidates.append(_candidate(
                    names[idx], folds, _COMPLEXITY.get(names[idx], 1), ratio,
                    idx=idx, est=gs.best_estimator_, params=gs.best_params_))
            for k, (fold_scores, mo_surrogate) in enumerate(_mo_scores):
                candidates.append(_candidate(
                    f"{type(mo_surrogate).__name__}[{k}]", fold_scores[:, i], 2,
                    ratio, mo=mo_surrogate))
            if not candidates:
                raise RuntimeError("No valid estimator found")

            self.selection_scores_.append(
                {c["name"]: {"mean": c["mean"], "se": c["se"]} for c in candidates})
            chosen = self._choose(candidates)

            if "mo" in chosen:
                self.models_.append(chosen["mo"])
                self._model_output_col.append(i)  # column i of the joint prediction
            elif hasattr(self, "_surrogate_constructors"):
                params = chosen["params"]
                if isinstance(self.estimators[chosen["idx"]], Pipeline):
                    # Grid keys are "<step>__<param>"; surrogates take the bare name.
                    params = {k.split("__", 1)[1]: v for k, v in params.items()}
                surrogate = self._surrogate_constructors[chosen["idx"]](**params)
                if isinstance(surrogate, MultiFidelitySurrogate):
                    surrogate.fit((X, y[:, [i]]))
                else:
                    surrogate.fit(X, y[:, [i]])
                self.models_.append(surrogate)
                self._model_output_col.append(0)  # single-output, always column 0
            else:
                best_est = clone(chosen["est"]).fit(X, y[:, i])
                self.models_.append(best_est)
                self._model_output_col.append(0)  # single-output, always column 0

        # expose base estimators for compatibility with plotting utilities
        self.estimators_ = []
        for model in self.models_:
            if hasattr(model, "model") and hasattr(model.model, "estimators_"):
                # vendored surrogate using MultiOutputRegressor internally
                self.estimators_.append(model.model.estimators_[0])
            elif hasattr(model, "estimators_"):
                self.estimators_.append(model.estimators_[0])
            else:
                self.estimators_.append(model)

        return self

    def register_multi_output_candidates(
        self, surrogates: list
    ) -> "AutoDetectMultiOutputRegressor":
        """Register joint multi-output surrogate candidates.

        These surrogates predict all outputs simultaneously (i.e. they set
        ``_multi_output = True``). They are evaluated once on the full Y matrix
        during ``fit()`` and each output's score is compared against the
        per-output GridSearchCV winners.

        Parameters
        ----------
        surrogates : list
            Fitted or unfitted surrogate instances with ``fit(X, Y)`` and
            ``predict(X)`` that return an ``(n_samples, n_outputs)`` array.

        Returns
        -------
        self
        """
        self._multi_output_candidates = list(surrogates)
        return self

    def predict(self, X: np.ndarray, return_std: bool = False) -> np.ndarray:
        if not hasattr(self, "models_"):
            raise AttributeError("Estimator not fitted")
        preds = []
        stds = []
        # Cache full prediction matrices keyed by model id so that multi-output
        # models shared across several output slots are only called once.
        # _model_output_col[i] stores which column of models_[i]'s prediction
        # corresponds to output i (set during fit()).
        _cache: dict[int, tuple] = {}  # id(model) -> (pred_2d, std_2d_or_None)
        _model_cols = getattr(self, "_model_output_col", [0] * len(self.models_))
        for model, col in zip(self.models_, _model_cols):
            mid = id(model)
            if mid not in _cache:
                if return_std:
                    try:
                        pred, std = model.predict(X, return_std=True)
                    except TypeError:
                        pred = model.predict(X)
                        std = np.zeros_like(pred)
                else:
                    pred = model.predict(X)
                    std = None

                pred = np.asarray(pred)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                if return_std and std is not None:
                    std = np.asarray(std)
                    if std.ndim == 1:
                        std = std.reshape(-1, 1)
                _cache[mid] = (pred, std)

            full_pred, full_std = _cache[mid]
            preds.append(full_pred[:, col : col + 1])
            if return_std:
                stds.append(
                    full_std[:, col : col + 1]
                    if full_std is not None
                    else np.zeros((X.shape[0], 1))
                )

        pred_mat = np.column_stack(preds)
        if return_std:
            std_mat = np.column_stack(stds)
            return pred_mat, std_mat
        return pred_mat

    @classmethod
    def with_vendored_surrogates(
        cls,
        cv: int | str = "auto",
        scoring: str = "neg_mean_squared_error",
        fidelity_levels: Sequence[str] | None = None,
        pre_screen: bool = False,
        selection: str = "best",
    ) -> "AutoDetectMultiOutputRegressor":
        """Return instance configured to search all vendored surrogates.

        Parameters
        ----------
        pre_screen : bool, default False
            When True, statistical tests are run at the start of ``fit()``
            to skip expensive models that are unlikely to help on this data.
            Computationally heavy models (GP, SVR, MLP, heteroscedastic
            ensembles) are only run when the data characteristics justify them.
        selection : {"best", "one_se"}, default "best"
            See the class docstring.

        Scale-sensitive candidates (GP, SVR, KNN, MLP, ElasticNet, Lasso) are
        grid-searched as ``StandardScaler`` pipelines, matching the scaling
        their surrogates apply when deployed.
        """

        def scaled(est):
            return make_pipeline(StandardScaler(), est)

        estimators = [
            LinearRegression(),
            scaled(GaussianProcessRegressor(kernel=default_gp_kernel(), normalize_y=True)),
            RandomForestRegressor(),
            ExtraTreesRegressor(),
            GradientBoostingRegressor(),
            scaled(SVR()),
            scaled(KNeighborsRegressor()),
            DecisionTreeRegressor(),
            _CPNEstimator(max_iter=500, random_state=0),
            BayesianRidge(),
            _RFFEstimator(n_components=500, length_scale=1.0),
            _PBREstimator(degree=2),
            _NystroemEstimator(n_components=100),
            _ARDGPEstimator(),
            HistGradientBoostingRegressor(),
            scaled(_ElasticNet()),
            scaled(_Lasso()),
            _QuantileRegressor(quantile=0.5),
        ]

        param_spaces = [
            {},
            {"gaussianprocessregressor__kernel": [default_gp_kernel(), matern_gp_kernel()]},
            {"n_estimators": [50, 100], "max_depth": [3, 5, None]},
            {"n_estimators": [50, 100], "max_depth": [3, 5, None]},
            {"n_estimators": [50, 100], "max_depth": [3, 5]},
            {"svr__C": [1.0, 10.0], "svr__gamma": ["scale", "auto"]},
            {"kneighborsregressor__n_neighbors": [3, 5, 7]},
            {"max_depth": [1, None]},
            {"hidden_layer_sizes": [(64,), (128,)], "alpha": [1e-4, 1e-3]},
            {},
            {"n_components": [100, 500], "length_scale": [0.1, 1.0, 10.0], "kernel": ["rbf", "matern52"]},
            {"degree": [2, 3], "interaction_only": [False, True]},
            {"n_components": [50, 200], "gamma": [None, 0.1, 1.0]},
            {"alpha": [1e-6, 1e-2]},
            {"max_iter": [100, 200], "learning_rate": [0.05, 0.1]},
            {"elasticnet__alpha": [0.01, 0.1, 1.0], "elasticnet__l1_ratio": [0.25, 0.5, 0.75]},
            {"lasso__alpha": [0.01, 0.1, 1.0]},
            {"alpha": [0.1, 1.0]},
        ]

        model_names = ["linear", "gp", "rf", "et", "gb", "svr", "knn", "dt", "mlp",
                       "bayesian_ridge", "rfgp", "pbr", "nystroem_gp", "ard_gp", "hgb",
                       "elastic_net", "lasso", "quantile"]
        surrogate_constructors = [
            LinearRegressionSurrogate,
            GaussianProcessSurrogate,
            RandomForestSurrogate,
            ExtraTreesRegressorSurrogate,
            GradientBoostingSurrogate,
            SVRSurrogate,
            KNeighborsSurrogate,
            DecisionTreeRegressorSurrogate,
            # same random_state as the _CPNEstimator that was grid-searched
            functools.partial(ConformalPredictionNetworkSurrogate, random_state=0),
            BayesianRidgeSurrogate,
            RFFGPSurrogate,
            PolynomialBayesianRidgeSurrogate,
            NystroemGPSurrogate,
            ARDGPSurrogate,
            HistGradientBoostingSurrogate,
            ElasticNetSurrogate,
            LassoSurrogate,
            QuantileRegressionSurrogate,
        ]

        if _NGBOOST_AVAILABLE:
            estimators.append(_NGBRegressor(n_estimators=200, verbose=False))
            param_spaces.append({"n_estimators": [100, 200], "learning_rate": [0.01, 0.05]})
            model_names.append("ngboost")
            surrogate_constructors.append(NGBoostSurrogate)

        if _GPX_AVAILABLE:
            estimators.append(_GPXEstimator())
            param_spaces.append({"corr": ["squar_exp", "matern52"], "poly": ["constant", "linear"]})
            model_names.append("gpx")
            surrogate_constructors.append(GPXSurrogate)

        if _KPLS_AVAILABLE:
            estimators.append(_KPLSEstimator())
            param_spaces.append({"n_comp": [2, 4], "corr": ["squar_exp", "matern52"]})
            model_names.append("kpls")
            surrogate_constructors.append(KPLSSurrogate)

        if _LIGHTGBM_AVAILABLE:
            estimators.append(_LGBMRegressor(n_estimators=200, verbose=-1))
            param_spaces.append({"n_estimators": [100, 200], "learning_rate": [0.01, 0.05], "num_leaves": [31, 63]})
            model_names.append("lgbm")
            surrogate_constructors.append(LightGBMSurrogate)

        if _XGBOOST_AVAILABLE:
            estimators.append(_XGBRegressor(n_estimators=200, verbosity=0))
            param_spaces.append({"n_estimators": [100, 200], "learning_rate": [0.01, 0.05], "max_depth": [4, 6]})
            model_names.append("xgb")
            surrogate_constructors.append(XGBoostSurrogate)

        if _CATBOOST_AVAILABLE:
            estimators.append(_CatBoostRegressor(n_estimators=200, verbose=False, **catboost_thread_kwargs()))
            param_spaces.append({"n_estimators": [100, 200], "learning_rate": [0.01, 0.05], "depth": [4, 6]})
            model_names.append("catboost")
            surrogate_constructors.append(CatBoostSurrogate)

        if _GPYTORCH_AVAILABLE:
            from multioutreg.surrogates.sparse_gp_gpytorch import _SGPREstimator
            estimators.append(_SGPREstimator(n_inducing=50, max_iter=100))
            param_spaces.append({"n_inducing": [30, 50, 100], "learning_rate": [0.05, 0.1]})
            model_names.append("sgpr")
            surrogate_constructors.append(SparseGPSurrogate)

        instance = cls(estimators, param_spaces, cv=cv, scoring=scoring,
                       pre_screen=pre_screen, selection=selection)
        # Short names used by ModelScreener.eligible_indices_for_output()
        instance._model_names = model_names

        if fidelity_levels is None:
            instance._surrogate_constructors = surrogate_constructors
        else:
            def wrap(cls_sur):
                return lambda **p: MultiFidelitySurrogate(
                    lambda: cls_sur(**p), fidelity_levels
                )

            instance._surrogate_constructors = [wrap(c) for c in surrogate_constructors]
            instance.fidelity_levels = list(fidelity_levels)

        return instance

    def calibrate_conformal(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        **kwargs,
    ) -> "AutoDetectMultiOutputRegressor":
        """Calibrate conformal prediction intervals using held-out data.

        Computes absolute residuals on the calibration set using the
        already-fitted model. These residuals are used at prediction time
        to construct distribution-free intervals with coverage guarantees.

        Parameters
        ----------
        X_cal : np.ndarray
            Calibration features (must NOT overlap with training data).
        y_cal : np.ndarray
            Calibration targets.

        Returns
        -------
        self
        """
        from multioutreg.conformal.base import BaseConformalPredictor

        if not hasattr(self, "models_"):
            raise AttributeError("Estimator not fitted. Call fit() first.")

        y_cal = np.asarray(y_cal)
        if y_cal.ndim == 1:
            y_cal = y_cal.reshape(-1, 1)

        y_cal_pred = self.predict(X_cal)
        if y_cal_pred.ndim == 1:
            y_cal_pred = y_cal_pred.reshape(-1, 1)

        self._conformal_residuals = np.abs(y_cal - y_cal_pred)
        self._conformal_n_outputs = y_cal.shape[1]
        return self

    def predict_interval(
        self, X: np.ndarray, alpha: float = 0.1
    ) -> tuple:
        """Return conformal prediction intervals.

        Parameters
        ----------
        X : np.ndarray
        alpha : float
            Miscoverage level. Intervals target 1-alpha coverage.

        Returns
        -------
        y_lower, y_upper : np.ndarray
        """
        if not hasattr(self, "_conformal_residuals"):
            raise AttributeError(
                "No conformal predictor calibrated. Call calibrate_conformal() first."
            )
        from multioutreg.conformal.base import BaseConformalPredictor

        y_pred = self.predict(X)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        q = np.array([
            BaseConformalPredictor._conformal_quantile(
                self._conformal_residuals[:, j], alpha
            )
            for j in range(self._conformal_n_outputs)
        ])

        y_lower = y_pred - q[np.newaxis, :]
        y_upper = y_pred + q[np.newaxis, :]
        return y_lower, y_upper