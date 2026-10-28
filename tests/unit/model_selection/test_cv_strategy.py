# Copyright (c) 2025 takotime808

"""Tests for dataset-size-dependent cross-validation in model selection."""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, LeaveOneOut, RepeatedKFold
from sklearn.tree import DecisionTreeRegressor

from multioutreg.model_selection.auto_detect import (
    AutoDetectMultiOutputRegressor,
    _candidate,
    describe_cv,
    resolve_cv,
    select_cv_strategy,
)


@pytest.mark.parametrize("n, cls, n_splits", [
    (3, RepeatedKFold, 2),
    (8, RepeatedKFold, 2),
    (12, RepeatedKFold, 3),
    (20, RepeatedKFold, 5),
    (21, RepeatedKFold, 5),
    (100, RepeatedKFold, 5),
    (101, KFold, 10),
    (1000, KFold, 10),
    (1001, KFold, 5),
    (50_000, KFold, 5),
])
def test_select_cv_strategy_thresholds(n, cls, n_splits):
    cv = select_cv_strategy(n)
    assert type(cv) is cls
    if cls is RepeatedKFold:
        assert cv.cvargs["n_splits"] == n_splits and cv.n_repeats == 3
    elif cls is KFold:
        assert cv.n_splits == n_splits and cv.shuffle


def test_select_cv_strategy_is_seeded():
    X = np.arange(60).reshape(-1, 1)
    a = [te.tolist() for _, te in select_cv_strategy(60, random_state=1).split(X)]
    b = [te.tolist() for _, te in select_cv_strategy(60, random_state=1).split(X)]
    c = [te.tolist() for _, te in select_cv_strategy(60, random_state=2).split(X)]
    assert a == b and a != c


def test_resolve_cv_forms():
    assert isinstance(resolve_cv("auto", 10), RepeatedKFold)
    assert isinstance(resolve_cv("auto", 50), RepeatedKFold)
    assert isinstance(resolve_cv("loocv", 5000), LeaveOneOut)
    int_cv = resolve_cv(4, 100)
    assert isinstance(int_cv, KFold) and int_cv.n_splits == 4 and int_cv.shuffle
    custom = KFold(n_splits=3)
    assert resolve_cv(custom, 100) is custom


def test_resolve_cv_rejects_unknown_string():
    with pytest.raises(ValueError, match="cv must be"):
        resolve_cv("kfold", 100)


def test_auto_never_uses_loo():
    # LOO is inconsistent for model selection (Shao 1993); "auto" uses repeated
    # k-fold with >= 4 rows per fold, so R² is defined on every fold too.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cv = resolve_cv("auto", 12, scoring="r2")
    assert isinstance(cv, RepeatedKFold)
    X = np.zeros((12, 1))
    assert min(len(te) for _, te in cv.split(X)) >= 4


def test_loocv_is_still_available_but_not_for_r2():
    assert isinstance(resolve_cv("loocv", 12), LeaveOneOut)
    with pytest.raises(ValueError, match="loocv"):
        resolve_cv("loocv", 12, scoring="r2")


@pytest.mark.parametrize("cv, text", [
    (LeaveOneOut(), "Leave-One-Out CV"),
    (RepeatedKFold(n_splits=5, n_repeats=3), "Repeated 5-Fold CV (x3 repeats)"),
    (KFold(10, shuffle=True, random_state=0), "10-Fold CV (shuffled)"),
    (KFold(3), "3-Fold CV"),
    (5, "5-Fold CV"),
])
def test_describe_cv(cv, text):
    assert describe_cv(cv) == text


def test_candidate_se_uses_actual_test_train_ratio():
    scores = [1.0, 2.0, 3.0, 1.5, 2.5] * 3     # 15 repeated-K-fold scores
    var = np.var(scores, ddof=1)
    default = _candidate("m", scores, 0)["se"]
    repeated = _candidate("m", scores, 0, test_train_ratio=0.25)["se"]
    assert default == pytest.approx(np.sqrt((1 / 15 + 1 / 14) * var))
    assert repeated == pytest.approx(np.sqrt((1 / 15 + 0.25) * var))


def _model(cv="auto", scoring="neg_mean_squared_error"):
    return AutoDetectMultiOutputRegressor(
        [LinearRegression(), DecisionTreeRegressor(random_state=0)],
        [{}, {"max_depth": [1, None]}],
        cv=cv, scoring=scoring,
    )


@pytest.mark.parametrize("n, description, n_folds", [
    (12, "Repeated 3-Fold CV (x3 repeats)", 9),
    (60, "Repeated 5-Fold CV (x3 repeats)", 15),
    (300, "10-Fold CV (shuffled)", 10),
])
def test_fit_resolves_cv_from_training_size(n, description, n_folds):
    X, y = make_regression(n_samples=n, n_features=3, n_targets=2, noise=0.1,
                           random_state=0)
    model = _model().fit(X, y)
    assert model.cv_description_ == description
    assert model.cv_strategy_.get_n_splits(X) == n_folds
    assert model.predict(X).shape == y.shape
    assert all(np.isfinite(v["mean"]) and np.isfinite(v["se"])
               for scores in model.selection_scores_ for v in scores.values())


def test_fit_with_r2_scoring_on_tiny_data_is_finite():
    X, y = make_regression(n_samples=12, n_features=3, n_targets=1, noise=0.1,
                           random_state=0)
    model = _model(scoring="r2").fit(X, y)
    assert isinstance(model.cv_strategy_, RepeatedKFold)
    assert all(np.isfinite(v["mean"]) for v in model.selection_scores_[0].values())


def test_explicit_cv_settings_still_work():
    X, y = make_regression(n_samples=40, n_features=3, n_targets=2, noise=0.1,
                           random_state=0)
    assert _model(cv="loocv").fit(X, y).cv_description_ == "Leave-One-Out CV"
    assert _model(cv=2).fit(X, y).cv_description_ == "2-Fold CV (shuffled)"


def test_vendored_default_is_auto():
    assert AutoDetectMultiOutputRegressor.with_vendored_surrogates().cv == "auto"


def test_cv_attributes_not_set_before_fit():
    # sklearn convention: trailing-underscore attributes only exist after fit.
    assert not hasattr(_model(), "cv_description_")


def test_clone_round_trip_keeps_auto_cv():
    from sklearn.base import clone

    X, y = make_regression(n_samples=30, n_features=3, n_targets=1, noise=0.1,
                           random_state=0)
    vendored = AutoDetectMultiOutputRegressor.with_vendored_surrogates()
    copy = clone(vendored)
    assert copy.cv == "auto" and copy._model_names == vendored._model_names
    copy = clone(_model().fit(X, y))
    assert copy.cv == "auto" and not hasattr(copy, "cv_strategy_")
