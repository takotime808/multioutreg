# Copyright (c) 2026 takotime808
"""Fit/predict contract checked against every surrogate exported by ``multioutreg.surrogates``.

Surrogates whose optional dependency is missing are skipped, unless
``MULTIOUTREG_REQUIRE_ALL_SURROGATES=1`` is set (as in CI), in which case
they fail so a dropped dependency cannot silently remove coverage.
"""

import importlib
import os

import numpy as np
import pytest

import multioutreg.surrogates as surrogates_pkg
from multioutreg.surrogates import ConformalMixin

# Exported names that are building blocks, not concrete surrogates.
NON_SURROGATES = {"ConformalMixin", "BaseSurrogate"}

REQUIRE_ALL = os.environ.get("MULTIOUTREG_REQUIRE_ALL_SURROGATES") == "1"

N_OUTPUTS = 2


def _plain(**kwargs):
    """Spec for a surrogate with the standard ``fit(X, Y)`` / ``predict(X, return_std)`` API."""
    return {"kwargs": kwargs, "kind": "plain"}


# name -> spec.  ``requires`` is (module, availability flag) for optional backends.
# Hyperparameters are kept small so the whole suite stays fast.
SPECS = {
    "LinearRegressionSurrogate": _plain(),
    "GaussianProcessSurrogate": _plain(),
    "RandomForestSurrogate": _plain(n_estimators=20, random_state=0),
    "GradientBoostingSurrogate": _plain(n_estimators=20, random_state=0),
    "SVRSurrogate": _plain(),
    "KNeighborsSurrogate": _plain(n_neighbors=5),
    "DecisionTreeRegressorSurrogate": _plain(random_state=0),
    "ConformalPredictionNetworkSurrogate": _plain(hidden_layer_sizes=(16,), max_iter=200, random_state=0),
    "ExtraTreesRegressorSurrogate": _plain(n_estimators=20, random_state=0),
    "NGBoostSurrogate": {
        **_plain(n_estimators=30),
        "requires": ("multioutreg.surrogates.ngboost_sklearn", "_NGBOOST_AVAILABLE"),
    },
    "BNNSurrogate": {
        **_plain(hidden_layer_sizes=(16,), max_epochs=20, n_mc_samples=10, random_state=0),
        "requires": ("multioutreg.surrogates._torch_utils", "_TORCH_AVAILABLE"),
    },
    "MultiFidelitySurrogate": {"kind": "multi_fidelity"},
    "StackedVFMSurrogate": {"kind": "stacked_vfm"},
    "AdditiveCorrectionVFM": {"kind": "additive_vfm"},
    "MixtureOfExpertsSurrogate": _plain(n_experts=2, max_em_iters=3, random_state=0),
    "BayesianRidgeSurrogate": _plain(),
    "RFFGPSurrogate": _plain(n_components=50, random_state=0),
    "PolynomialBayesianRidgeSurrogate": _plain(degree=2),
    "NystroemGPSurrogate": _plain(n_components=20, random_state=0),
    "GPXSurrogate": {
        **_plain(),
        "requires": ("multioutreg.surrogates.gpx_smt", "_GPX_AVAILABLE"),
    },
    "KPLSSurrogate": {
        **_plain(n_comp=1),
        "requires": ("multioutreg.surrogates.kpls_smt", "_KPLS_AVAILABLE"),
    },
    "ARDGPSurrogate": _plain(),
    "HistGradientBoostingSurrogate": _plain(max_iter=20, min_samples_leaf=5),
    "LightGBMSurrogate": {
        **_plain(n_estimators=20, min_child_samples=5),
        "requires": ("multioutreg.surrogates.lightgbm_sklearn", "_LIGHTGBM_AVAILABLE"),
    },
    "XGBoostSurrogate": {
        **_plain(n_estimators=20),
        "requires": ("multioutreg.surrogates.xgboost_sklearn", "_XGBOOST_AVAILABLE"),
    },
    "CatBoostSurrogate": {
        **_plain(n_estimators=20, virtual_ensembles_count=5),
        "requires": ("multioutreg.surrogates.catboost_sklearn", "_CATBOOST_AVAILABLE"),
    },
    # The default alpha=1.0 shrinks every coefficient to zero on this data.
    "ElasticNetSurrogate": _plain(alpha=0.01),
    "LassoSurrogate": _plain(alpha=0.01),
    "DeepEnsembleSurrogate": {
        **_plain(n_estimators=2, hidden_layer_sizes=(16,), max_epochs=200, learning_rate=1e-2, random_state=0),
        "requires": ("multioutreg.surrogates._torch_utils", "_TORCH_AVAILABLE"),
    },
    "SparseGPSurrogate": {
        **_plain(n_inducing=10, max_iter=20, random_state=0),
        "requires": ("multioutreg.surrogates.sparse_gp_gpytorch", "_GPYTORCH_AVAILABLE"),
    },
    "QuantileRegressionSurrogate": _plain(alpha=0.01),
    "PCAOutputSurrogate": _plain(n_components=2),
    "MultiTaskGPSurrogate": {
        **_plain(max_iter=50, random_state=0),
        "requires": ("multioutreg.surrogates.multitask_gp_gpytorch", "_GPYTORCH_AVAILABLE"),
    },
    "GaussianCopulaSurrogate": _plain(cv=3, random_state=0),
    "StdRecalibratedSurrogate": _plain(criterion="nll", cv=3, random_state=0),
    "MFKSurrogate": {
        "kind": "mfk",
        "requires": ("multioutreg.surrogates.mfk_smt", "_MFK_AVAILABLE"),
    },
}

SURROGATE_NAMES = sorted(SPECS)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(80, 2))
    Y = np.column_stack([np.sin(3 * X[:, 0]) + X[:, 1], X[:, 0] * X[:, 1]])
    Y = Y + 0.05 * rng.normal(size=Y.shape)
    return X[:60], Y[:60], X[60:], Y[60:]


def _skip_if_unavailable(name):
    requires = SPECS[name].get("requires")
    if requires is None:
        return
    module, flag = requires
    if getattr(importlib.import_module(module), flag):
        return
    msg = f"{name}: optional dependency for {module} not installed"
    if REQUIRE_ALL:
        pytest.fail(msg + " (MULTIOUTREG_REQUIRE_ALL_SURROGATES=1)")
    pytest.skip(msg)


def build_surrogate(name):
    """Construct an unfitted ``name`` surrogate from its spec."""
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
    if kind == "mfk":
        return cls(["lo", "hi"], n_start=2)
    raise AssertionError(f"unknown spec kind {kind!r}")


def fit_surrogate(name, model, X, Y):
    """Fit ``model`` (built by ``build_surrogate(name)``) on (X, Y), adapting to its fit API."""
    kind = SPECS[name]["kind"]
    if kind == "plain":
        return model.fit(X, Y)
    if kind == "multi_fidelity":
        return model.fit((X, Y))
    # A cheap, deterministic low-fidelity version of Y for the VFM models.
    Y_lo = 0.8 * Y + 0.1
    if kind == "mfk":
        # Co-kriging wants nested designs: high fidelity is a subset of low.
        return model.fit({"lo": (X, Y_lo), "hi": (X[::2], Y[::2])})
    return model.fit({"lo": (X, Y_lo), "hi": (X, Y)})


def _build_and_fit(name, X, Y):
    return fit_surrogate(name, build_surrogate(name), X, Y)


def test_every_exported_surrogate_has_a_spec():
    exported = set(surrogates_pkg.__all__) - NON_SURROGATES
    assert exported - set(SPECS) == set(), "add new surrogates to SPECS"
    assert set(SPECS) - exported == set(), "SPECS lists names no longer exported"


@pytest.mark.parametrize("name", SURROGATE_NAMES)
def test_fit_predict_mean_and_std(name, data):
    _skip_if_unavailable(name)
    X, Y, X_test, _ = data
    model = _build_and_fit(name, X, Y)

    mean = np.asarray(model.predict(X_test))
    assert mean.shape == (len(X_test), N_OUTPUTS)
    assert np.all(np.isfinite(mean))

    mean_s, std = model.predict(X_test, return_std=True)
    mean_s, std = np.asarray(mean_s), np.asarray(std)
    assert mean_s.shape == std.shape == (len(X_test), N_OUTPUTS)
    assert np.all(np.isfinite(mean_s))
    assert np.all(np.isfinite(std))
    assert np.all(std >= 0)


@pytest.mark.parametrize("name", SURROGATE_NAMES)
def test_predictions_track_the_target(name, data):
    # Every surrogate should beat predicting the training mean on the
    # smooth, nearly-noiseless first output; a constant/broken fit will not.
    _skip_if_unavailable(name)
    X, Y, X_test, Y_test = data
    model = _build_and_fit(name, X, Y)
    pred = np.asarray(model.predict(X_test))[:, 0]
    mse_model = np.mean((pred - Y_test[:, 0]) ** 2)
    mse_mean = np.mean((Y[:, 0].mean() - Y_test[:, 0]) ** 2)
    assert mse_model < mse_mean


@pytest.mark.parametrize(
    "name",
    [n for n in SURROGATE_NAMES if issubclass(getattr(surrogates_pkg, n), ConformalMixin)],
)
def test_conformal_intervals_bracket_point_prediction(name, data):
    _skip_if_unavailable(name)
    X, Y, X_test, Y_test = data
    model = _build_and_fit(name, X, Y)
    model.wrap_conformal(X_test[:10], Y_test[:10])
    lower, upper = model.conformal_predict(X_test[10:], alpha=0.2)
    point = model._conformal_point_predict(X_test[10:])
    assert lower.shape == upper.shape == point.shape == (len(X_test) - 10, N_OUTPUTS)
    assert np.all(lower <= point + 1e-12)
    assert np.all(point <= upper + 1e-12)
