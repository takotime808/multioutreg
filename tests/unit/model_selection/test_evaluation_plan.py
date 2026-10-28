# Copyright (c) 2026 takotime808

"""Tests for sample-size-dependent evaluation plans (AUDIT_CHECKLIST S1, S2)."""

import math
import warnings

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import BayesianRidge

from multioutreg.conformal.base import BaseConformalPredictor
from multioutreg.model_selection.evaluation import (
    MIN_TEST_POINTS,
    fit_with_conformal,
    min_calibration_points,
    nested_cv_predict,
    plan_evaluation,
)
from multioutreg.surrogates import BayesianRidgeSurrogate


@pytest.mark.parametrize("alpha, need", [(0.1, 9), (0.05, 19), (0.2, 4), (0.3, 3)])
def test_min_calibration_points(alpha, need):
    assert min_calibration_points(alpha) == need


@pytest.mark.parametrize("alpha", [0.1, 0.05, 0.2])
def test_min_calibration_points_is_the_finite_interval_boundary(alpha):
    need = min_calibration_points(alpha)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isfinite(BaseConformalPredictor._conformal_quantile(np.arange(need, dtype=float), alpha))
    with pytest.warns(UserWarning, match="too few"):
        assert BaseConformalPredictor._conformal_quantile(np.arange(need - 1, dtype=float), alpha) == np.inf


def test_holdout_vs_nested_boundary():
    # ceil(0.25 n) >= 30 first holds at n = 117.
    assert plan_evaluation(116).scheme == "nested_cv"
    big = plan_evaluation(117)
    assert big.scheme == "holdout" and big.outer_cv is None
    assert math.ceil(0.25 * 117) == MIN_TEST_POINTS


def test_nested_plan_uses_seeded_shuffled_kfold():
    plan = plan_evaluation(40, random_state=3)
    assert plan.outer_cv.n_splits == 5 and plan.outer_cv.shuffle
    assert plan.outer_cv.random_state == 3
    assert plan_evaluation(4).outer_cv.n_splits == 4
    assert "nested" in plan.describe().lower() and plan.notes


def test_split_vs_cv_plus_boundary():
    # Holdout n=200: n_fit = 150, n_cal = 30 -> split at alpha 0.05 (needs 19).
    assert plan_evaluation(200, alpha=0.05).conformal == "split"
    # Holdout n=120: n_fit = 90, n_cal = 18 < 19 -> CV+.
    plan = plan_evaluation(120, alpha=0.05)
    assert plan.scheme == "holdout" and plan.conformal == "cv_plus"
    assert any("CV+" in note for note in plan.notes)
    assert plan_evaluation(120, alpha=0.1).conformal == "split"     # 18 >= 9
    assert plan_evaluation(200).conformal is None


def test_too_small_for_alpha_is_reported():
    plan = plan_evaluation(10, alpha=0.05)
    assert plan.conformal == "cv_plus"
    assert any("unbounded" in note for note in plan.notes)


def test_tiny_dataset_rejected():
    with pytest.raises(ValueError):
        plan_evaluation(1)


def _data(n, n_targets=2, seed=0):
    return make_regression(n_samples=n, n_features=3, n_targets=n_targets, noise=5.0,
                           random_state=seed)


@pytest.mark.parametrize("n, alpha, method", [(300, 0.1, "split"), (30, 0.1, "cv_plus")])
def test_fit_with_conformal_methods(n, alpha, method):
    X, y = _data(n)
    plan = plan_evaluation(n, alpha=alpha)
    fit = fit_with_conformal(BayesianRidgeSurrogate(), X, y, plan)
    assert fit.method == method
    pred, std = fit.model.predict(X[:7], return_std=True)
    lo, hi = fit.predict_interval(X[:7], alpha)
    assert lo.shape == hi.shape == pred.shape == (7, 2)
    assert np.all(np.isfinite(lo)) and np.all(lo <= pred + 1e-9) and np.all(pred <= hi + 1e-9)
    assert fit.residuals.shape[1] == 2


def test_fit_with_conformal_single_output_stays_2d():
    X, y = _data(30, n_targets=1)
    fit = fit_with_conformal(BayesianRidgeSurrogate(), X, y, plan_evaluation(30, alpha=0.1))
    assert fit.method == "cv_plus"
    lo, hi = fit.predict_interval(X[:5], 0.1)
    assert lo.shape == (5, 1) and fit.residuals.shape == (30, 1)


def test_cv_plus_covers_on_fresh_data():
    # Marginal coverage of CV+ on new draws from the same distribution.
    X, y = _data(400, n_targets=1, seed=1)
    X_tr, y_tr, X_te, y_te = X[:30], y[:30], X[30:], y[30:].reshape(-1, 1)
    fit = fit_with_conformal(BayesianRidgeSurrogate(), X_tr, y_tr,
                             plan_evaluation(30, alpha=0.1))
    assert fit.method == "cv_plus"
    lo, hi = fit.predict_interval(X_te, 0.1)
    coverage = np.mean((y_te >= lo) & (y_te <= hi))
    assert coverage >= 0.8      # guarantee is >= 1 - 2 alpha; ~0.9 in practice


def test_nested_cv_predict_never_trains_on_the_row_it_predicts():
    X, y = _data(40)
    plan = plan_evaluation(40)
    seen = []

    def fit_predict(X_tr, y_tr, X_te):
        seen.append({tuple(r) for r in X_tr})
        return {"pred": np.zeros((len(X_te), 2)), "std": np.ones((len(X_te), 2))}

    oof = nested_cv_predict(fit_predict, X, y, plan)
    assert oof.pred.shape == (40, 2) and oof.lower is None
    assert set(oof.fold) == set(range(5))
    for i, row in enumerate(X):
        assert tuple(row) not in seen[oof.fold[i]]


def test_nested_cv_predict_with_conformal_end_to_end():
    X, y = _data(50)
    plan = plan_evaluation(50, alpha=0.2)

    def fit_predict(X_tr, y_tr, X_te):
        inner = plan_evaluation(len(X_tr), alpha=plan.alpha, random_state=plan.random_state)
        fit = fit_with_conformal(BayesianRidgeSurrogate(), X_tr, y_tr, inner)
        pred, std = fit.model.predict(X_te, return_std=True)
        lo, hi = fit.predict_interval(X_te, plan.alpha)
        return {"pred": pred, "std": std, "lower": lo, "upper": hi}

    oof = nested_cv_predict(fit_predict, X, y, plan)
    assert np.all(np.isfinite(oof.pred)) and np.all(np.isfinite(oof.lower))
    assert np.mean((y >= oof.lower) & (y <= oof.upper)) > 0.6


@pytest.mark.parametrize("n, scheme, n_test", [(300, "holdout", 75), (40, "nested_cv", 40)])
def test_run_evaluation(n, scheme, n_test):
    from multioutreg.model_selection.evaluation import run_evaluation

    X, y = _data(n)
    plan = plan_evaluation(n, alpha=0.2)
    calls = []

    def preprocess(X_tr, X_te):
        calls.append(len(X_tr))
        mu = X_tr.mean(axis=0)
        return X_tr - mu, X_te - mu, {"mu": mu}

    res = run_evaluation(BayesianRidgeSurrogate, X, y, plan, preprocess)
    assert res.plan.scheme == scheme and res.n_test == n_test
    assert res.pred.shape == res.y_true.shape == res.lower.shape == (n_test, 2)
    assert np.all(np.isfinite(res.pred)) and np.all(np.isfinite(res.upper))
    assert res.model.predict(res.X_eval).shape == (n_test, 2)
    assert "mu" in res.info
    # nested: 5 outer folds + the final fit; holdout: one fit
    assert len(calls) == (6 if scheme == "nested_cv" else 1)
    if scheme == "nested_cv":
        assert res.n_train == n and calls[-1] == n


def test_run_evaluation_uses_cheaper_cv_inside_outer_folds():
    from multioutreg.model_selection.evaluation import run_evaluation

    X, y = _data(40)
    seen = []

    class Recorder(BayesianRidgeSurrogate):
        cv = "auto"

        def fit(self, X, y):
            seen.append(self.cv)
            return super().fit(X, y)

    run_evaluation(Recorder, X, y, plan_evaluation(40))
    assert seen == ["auto_single"] * 5 + ["auto"]      # 5 outer folds, then the final fit
    seen.clear()
    run_evaluation(Recorder, X, y, plan_evaluation(40), fold_cv=None)
    assert seen == ["auto"] * 6


def test_nested_cv_predict_requires_nested_plan():
    X, y = _data(300)
    with pytest.raises(ValueError, match="nested_cv"):
        nested_cv_predict(lambda *a: {}, X, y, plan_evaluation(300))
