# Copyright (c) 2026 takotime808
"""Every exported surrogate can be copied for refitting (CV folds, conformal, AutoDetect).

``sklearn.base.clone`` is tried first. The vendored surrogates do not all expose
``get_params``, so when clone raises ``TypeError`` a ``copy.deepcopy`` fallback is
accepted instead, mirroring ``multioutreg.conformal.cv_plus._fresh_copy`` and
``multioutreg.model_selection.auto_detect._clone_surrogate``.
"""

import copy

import numpy as np
import pytest
from sklearn.base import clone

import multioutreg.surrogates as surrogates_pkg
from multioutreg.conformal.cv_plus import _fresh_copy
from multioutreg.model_selection.auto_detect import _clone_surrogate
from tests.unit.surrogates.test_all_surrogates_contract import (
    SPECS,
    SURROGATE_NAMES,
    _skip_if_unavailable,
)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(60, 2))
    Y = np.column_stack([np.sin(3 * X[:, 0]) + X[:, 1], X[:, 0] * X[:, 1]])
    return X, Y + 0.05 * rng.normal(size=Y.shape)


def _build(name):
    """Construct an unfitted ``name`` surrogate from its contract-test spec."""
    cls = getattr(surrogates_pkg, name)
    spec = SPECS[name]
    kind = spec["kind"]
    linear = surrogates_pkg.LinearRegressionSurrogate
    if kind == "plain":
        return cls(**spec["kwargs"])
    if kind == "multi_fidelity":
        return cls(linear, ["hi"])
    if kind == "stacked_vfm":
        return cls(["lo", "hi"], surrogate_cls=linear, oof_cv=3)
    if kind == "additive_vfm":
        return cls(lo_surrogate_cls=linear, hi_surrogate_cls=linear, oof_cv=3)
    raise AssertionError(f"unknown spec kind {kind!r}")


def _fit(name, model, X, Y):
    kind = SPECS[name]["kind"]
    if kind == "plain":
        return model.fit(X, Y)
    if kind == "multi_fidelity":
        return model.fit((X, Y))
    # A cheap, deterministic low-fidelity version of Y for the VFM models.
    return model.fit({"lo": (X, 0.8 * Y + 0.1), "hi": (X, Y)})


def _clone_or_fallback(model):
    """Return ``(copy, method)``: sklearn clone when supported, else a deep copy."""
    try:
        return clone(model), "clone"
    except TypeError:
        return copy.deepcopy(model), "deepcopy"


@pytest.mark.parametrize("name", SURROGATE_NAMES)
def test_unfitted_copy_fits_like_the_original(name, data):
    _skip_if_unavailable(name)
    X, Y = data
    model = _build(name)
    dup, method = _clone_or_fallback(model)

    assert dup is not model, method
    assert type(dup) is type(model)
    if method == "clone":
        assert dup.get_params(deep=False).keys() == model.get_params(deep=False).keys()

    pred_orig = np.asarray(_fit(name, model, X, Y).predict(X))
    pred_dup = np.asarray(_fit(name, dup, X, Y).predict(X))
    assert pred_dup.shape == pred_orig.shape == Y.shape
    assert np.all(np.isfinite(pred_dup))


@pytest.mark.parametrize("name", SURROGATE_NAMES)
def test_refitting_a_copy_leaves_the_fitted_original_untouched(name, data):
    # Copying a fitted model and refitting it (as CV+ and AutoDetect folds do)
    # must not share state with the original.
    _skip_if_unavailable(name)
    X, Y = data
    model = _fit(name, _build(name), X, Y)
    before = np.asarray(model.predict(X)).copy()

    dup, _ = _clone_or_fallback(model)
    _fit(name, dup, X, 3.0 * Y + 10.0)

    np.testing.assert_allclose(np.asarray(model.predict(X)), before)
    assert not np.allclose(np.asarray(dup.predict(X)), before)


@pytest.mark.parametrize("name", SURROGATE_NAMES)
def test_sklearn_clone_of_a_fitted_surrogate_refits(name, data):
    # Every surrogate supports clone itself (no fallback), including ones whose
    # fitted state cannot be deep-copied.
    _skip_if_unavailable(name)
    X, Y = data
    model = _fit(name, _build(name), X, Y)
    dup = clone(model)
    assert dup is not model
    assert type(dup) is type(model)
    pred = np.asarray(_fit(name, dup, X, Y).predict(X))
    assert pred.shape == Y.shape
    assert np.all(np.isfinite(pred))


@pytest.mark.parametrize("copier", [_fresh_copy, _clone_surrogate], ids=["cv_plus", "auto_detect"])
@pytest.mark.parametrize("name", SURROGATE_NAMES)
def test_library_copy_helpers_handle_every_surrogate(name, copier, data):
    _skip_if_unavailable(name)
    X, Y = data
    model = _build(name)
    dup = copier(model)
    assert dup is not model
    pred = np.asarray(_fit(name, dup, X, Y).predict(X))
    assert pred.shape == Y.shape
    assert np.all(np.isfinite(pred))
