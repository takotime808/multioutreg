# Copyright (c) 2025 takotime808

"""Integration smoke tests for dataset-size-dependent CV selection."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, LeaveOneOut, RepeatedKFold
from sklearn.tree import DecisionTreeRegressor

from multioutreg.model_selection.auto_detect import (
    AutoDetectMultiOutputRegressor,
    _resolve_cv,
    select_cv_strategy,
)


# ── select_cv_strategy boundary tests ──────────────────────────────────────────

def test_loocv_tiny():
    assert isinstance(select_cv_strategy(1), LeaveOneOut)
    assert isinstance(select_cv_strategy(20), LeaveOneOut)


def test_repeated_kfold_small():
    cv = select_cv_strategy(21)
    assert isinstance(cv, RepeatedKFold)
    cv = select_cv_strategy(100)
    assert isinstance(cv, RepeatedKFold)


def test_kfold_10_medium():
    cv = select_cv_strategy(101)
    assert isinstance(cv, KFold)
    assert cv.n_splits == 10
    cv = select_cv_strategy(1000)
    assert isinstance(cv, KFold)
    assert cv.n_splits == 10


def test_kfold_5_large():
    cv = select_cv_strategy(1001)
    assert isinstance(cv, KFold)
    assert cv.n_splits == 5
    cv = select_cv_strategy(50_000)
    assert isinstance(cv, KFold)
    assert cv.n_splits == 5


# ── _resolve_cv passthrough tests ──────────────────────────────────────────────

def test_resolve_auto_delegates_to_select():
    assert isinstance(_resolve_cv("auto", 10), LeaveOneOut)
    assert isinstance(_resolve_cv("auto", 50), RepeatedKFold)
    assert isinstance(_resolve_cv("auto", 200), KFold)


def test_resolve_loocv_string():
    assert isinstance(_resolve_cv("loocv", 5000), LeaveOneOut)


def test_resolve_int_passthrough():
    assert _resolve_cv(5, 100) == 5


def test_resolve_splitter_passthrough():
    custom = KFold(n_splits=3)
    assert _resolve_cv(custom, 100) is custom


# ── End-to-end fit/predict with auto CV ────────────────────────────────────────

@pytest.fixture
def tiny_data():
    rng = np.random.RandomState(42)
    X = rng.rand(12, 3)
    Y = np.column_stack([
        X[:, 0] + rng.randn(12) * 0.01,
        X[:, 1] + rng.randn(12) * 0.01,
    ])
    return X, Y


@pytest.fixture
def medium_data():
    rng = np.random.RandomState(42)
    X = rng.rand(300, 3)
    Y = np.column_stack([
        X[:, 0] + rng.randn(300) * 0.01,
        X[:, 1] + rng.randn(300) * 0.01,
    ])
    return X, Y


def _simple_model(cv="auto"):
    est = [LinearRegression(), DecisionTreeRegressor(random_state=0)]
    params = [{}, {"max_depth": [1, None]}]
    return AutoDetectMultiOutputRegressor(est, params, cv=cv)


def test_auto_cv_tiny_dataset_runs(tiny_data):
    """n=12 → LeaveOneOut is selected; fit+predict must complete."""
    X, Y = tiny_data
    model = _simple_model(cv="auto")
    model.fit(X, Y)
    preds = model.predict(X)
    assert preds.shape == Y.shape


def test_auto_cv_medium_dataset_runs(medium_data):
    """n=300 → KFold(10) is selected; fit+predict must complete."""
    X, Y = medium_data
    model = _simple_model(cv="auto")
    model.fit(X, Y)
    preds = model.predict(X)
    assert preds.shape == Y.shape


def test_explicit_loocv_string(tiny_data):
    """cv='loocv' forces LeaveOneOut regardless of dataset size."""
    X, Y = tiny_data
    model = _simple_model(cv="loocv")
    model.fit(X, Y)
    preds = model.predict(X)
    assert preds.shape == Y.shape


def test_int_cv_backward_compat(tiny_data):
    """Passing an integer still works (backward compatibility)."""
    X, Y = tiny_data
    model = _simple_model(cv=2)
    model.fit(X, Y)
    preds = model.predict(X)
    assert preds.shape == Y.shape


def test_default_is_auto():
    model = _simple_model()
    assert model.cv == "auto"
