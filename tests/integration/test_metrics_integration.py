# Copyright (c) 2026 takotime808
"""End-to-end use of multioutreg.metrics with fitted multi-output surrogates."""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

from multioutreg.metrics import (
    RegretTracker,
    epsilon_regret,
    hypervolume,
    hypervolume_regret,
    pareto_mask,
    scalarized_regret,
)
from tests.unit.report.test_template_html import minimal_context, template_env  # noqa: F401


@pytest.fixture(scope="module")
def data():
    X, Y = make_regression(n_samples=400, n_features=6, n_targets=3, noise=5.0, random_state=0)
    return X[:300], X[300:], Y[:300], Y[300:]


@pytest.fixture(scope="module")
def predictions(data):
    X_train, X_test, Y_train, _ = data
    models = {
        "linear": LinearRegression(),
        "forest": MultiOutputRegressor(RandomForestRegressor(n_estimators=30, random_state=0)),
        "mean": DummyRegressor(strategy="mean"),
    }
    return {name: m.fit(X_train, Y_train).predict(X_test) for name, m in models.items()}


def _regret_metrics(Y_true, Y_pred, reference_point, weights):
    """Compute regret the way the auto-detect GUI page does."""
    return {
        "hv": float(hypervolume_regret(Y_true, Y_pred, reference_point=reference_point)),
        "scalar": float(scalarized_regret(Y_true, Y_pred, weights=weights, reduce="mean")),
        "eps": float(epsilon_regret(Y_true, Y_pred)),
    }


def _shared_reference(*Ys):
    allY = np.vstack(Ys)
    return allY.max(axis=0) + 0.1 * np.ptp(allY, axis=0)


def test_true_values_as_prediction_give_zero_regret(data):
    *_, Y_test = data
    ref = _shared_reference(Y_test)
    W = np.random.default_rng(7).dirichlet(np.ones(3), size=8)
    assert _regret_metrics(Y_test, Y_test, ref, W) == {"hv": 0.0, "scalar": 0.0, "eps": 0.0}


def test_better_surrogate_has_lower_regret(data, predictions):
    *_, Y_test = data
    ref = _shared_reference(Y_test, *predictions.values())
    W = np.random.default_rng(7).dirichlet(np.ones(3), size=8)
    good = _regret_metrics(Y_test, predictions["linear"], ref, W)
    bad = _regret_metrics(Y_test, predictions["mean"], ref, W)

    assert good["hv"] < bad["hv"]
    assert good["eps"] < bad["eps"]
    assert abs(good["scalar"]) < abs(bad["scalar"])
    # A near-perfect linear fit on linear data recovers the front almost exactly.
    hv_true = hypervolume(Y_test, reference_point=ref)
    assert good["hv"] / hv_true < 0.05


def test_hypervolume_regret_consistent_with_hypervolume(data, predictions):
    *_, Y_test = data
    for Y_pred in predictions.values():
        ref = _shared_reference(Y_test, Y_pred)
        diff = hypervolume(Y_test, reference_point=ref) - hypervolume(Y_pred, reference_point=ref)
        assert hypervolume_regret(Y_test, Y_pred, reference_point=ref) == pytest.approx(max(0.0, diff))


def test_regret_depends_only_on_pareto_front(data, predictions):
    *_, Y_test = data
    Y_pred = predictions["forest"]
    ref = _shared_reference(Y_test, Y_pred)
    W = np.eye(3)
    Ft, Fp = Y_test[pareto_mask(Y_test)], Y_pred[pareto_mask(Y_pred)]
    assert len(Ft) < len(Y_test)
    assert _regret_metrics(Ft, Fp, ref, W) == pytest.approx(_regret_metrics(Y_test, Y_pred, ref, W))


def test_maximization_matches_negated_minimization(data, predictions):
    *_, Y_test = data
    Y_pred = predictions["forest"]
    minimize = [True, False, True]
    flip = np.array([1.0, -1.0, 1.0])
    ref = _shared_reference(Y_test * flip, Y_pred * flip)
    W = np.random.default_rng(1).dirichlet(np.ones(3), size=5)

    assert hypervolume_regret(Y_test, Y_pred, reference_point=ref, minimize=minimize) == pytest.approx(
        hypervolume_regret(Y_test * flip, Y_pred * flip, reference_point=ref))
    assert scalarized_regret(Y_test, Y_pred, W, minimize=minimize) == pytest.approx(
        scalarized_regret(Y_test * flip, Y_pred * flip, W))
    assert epsilon_regret(Y_test, Y_pred, minimize=minimize) == pytest.approx(
        epsilon_regret(Y_test * flip, Y_pred * flip))


def test_single_target_pipeline():
    X, y = make_regression(n_samples=200, n_features=4, noise=1.0, random_state=3)
    model = LinearRegression().fit(X[:150], y[:150])
    y_test, y_pred = y[150:], model.predict(X[150:])
    ref = np.array([max(y_test.max(), y_pred.max()) + 1.0])

    metrics = _regret_metrics(y_test, y_pred, ref, weights=1.0)
    assert metrics["hv"] == pytest.approx(max(0.0, y_pred.min() - y_test.min()))
    assert metrics["eps"] == pytest.approx(max(0.0, y_pred.min() - y_test.min()))
    assert metrics["scalar"] == pytest.approx(y_test.min() - y_pred.min())


def test_tracker_over_growing_training_set(data):
    X_train, X_test, Y_train, Y_test = data
    tracker = RegretTracker(n_weight_samples=8, reference_point=_shared_reference(Y_train, Y_test) + 50)
    sizes = [10, 40, 160, 300]
    for n in sizes:
        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=30, random_state=0))
        out = tracker.step(Y_test, model.fit(X_train[:n], Y_train[:n]).predict(X_test))
        assert set(out) == {"hv", "scalar", "eps"}
        assert all(np.isfinite(v) and v >= 0 for k, v in out.items() if k != "scalar")

    assert all(len(v) == len(sizes) for v in tracker.history.values())
    assert tracker.weights.shape == (8, 3)
    # More training data should recover the true front better.
    assert tracker.history["hv"][-1] < tracker.history["hv"][0]
    assert tracker.history["eps"][-1] < tracker.history["eps"][0]


def test_regret_section_renders_in_report(template_env, minimal_context, data, predictions):  # noqa: F811
    *_, Y_test = data
    ref = _shared_reference(Y_test, *predictions.values())
    W = np.full((1, 3), 1.0 / 3)
    good = _regret_metrics(Y_test, predictions["linear"], ref, W)
    bad = _regret_metrics(Y_test, predictions["mean"], ref, W)
    bound = (good["hv"] + bad["hv"]) / 2
    settings = {"hv_ref_mode": "Auto", "reference_point": ref.tolist(), "weights_mode": "Uniform"}
    template = template_env.get_template("template.html")

    html_good = template.render(**minimal_context, regret_metrics=good,
                                regret_bounds={"hv": bound, "scalar": None, "eps": None},
                                regret_settings=settings)
    html_bad = template.render(**minimal_context, regret_metrics=bad,
                               regret_bounds={"hv": bound, "scalar": None, "eps": None},
                               regret_settings=settings)

    assert "Multi-Objective Regret" in html_good
    assert '<span class="pass">PASS</span>' in html_good and '<span class="fail">FAIL</span>' not in html_good
    assert '<span class="fail">FAIL</span>' in html_bad
