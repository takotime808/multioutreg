# Copyright (c) 2026 takotime808

import numpy as np

from multioutreg.model_selection.screening import _blr


def test_bootstrap_lr_keeps_x_y_pairs():
    # Regression: X and y were resampled with independent indices, which
    # destroyed the X-y relationship and gave slopes near zero.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 1))
    y = 3.0 * X[:, 0] + 0.1 * rng.normal(size=200)
    model = _blr().fit(X, y)
    slopes = np.array([m.coef_[0] for m in model.models_])
    np.testing.assert_allclose(slopes, 3.0, atol=0.1)


def test_gbq_std_is_sigma_not_half_width():
    # Regression: std was the half-width of the 90% quantile interval
    # (~1.645 * sigma for Gaussian noise) instead of sigma itself.
    from multioutreg.model_selection.screening import _gbq

    rng = np.random.default_rng(0)
    X = rng.uniform(size=(2000, 1))
    y = X[:, 0] + rng.normal(size=2000)  # sigma = 1
    _, std = _gbq().fit(X, y).predict(X[:500], return_std=True)
    assert abs(std.mean() - 1.0) < 0.2


def test_normality_is_tested_on_residuals_not_y():
    # Regression: Shapiro-Wilk ran on raw y. A uniform X makes y = 5x uniform
    # (strongly non-normal) while the residuals are exactly Gaussian.
    from scipy import stats
    from multioutreg.model_selection.screening import test_normality

    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, size=(300, 1))
    y = 5.0 * X[:, 0] + 0.3 * rng.normal(size=300)
    assert stats.shapiro(y).pvalue < 1e-4
    assert test_normality(X, y).passed


def test_holm_correction_reduces_false_flags_on_noise(monkeypatch):
    # Regression: 3 tests x n_outputs at alpha=0.05 were OR'ed with no
    # correction, so pure noise was flagged in about half the runs.
    from multioutreg.model_selection import screening
    from multioutreg.model_selection.screening import ModelScreener, TestResult

    # The cross-validated RF check is not in the corrected family and is most of
    # the fit time (200 fits here), so skip it; it has its own tests.
    monkeypatch.setattr(screening, "test_nonlinear_dependency",
                        lambda *a, **k: TestResult("stub", True, 0.0, None, "stub"))

    keys = ("any_nonlinear", "any_nonnormal", "any_heteroscedastic")
    raw = holm = 0
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        X, Y = rng.normal(size=(150, 3)), rng.normal(size=(150, 5))
        raw += any(ModelScreener(correction=None).fit(X, Y).global_flags_[k] for k in keys)
        holm += any(ModelScreener().fit(X, Y).global_flags_[k] for k in keys)
    assert holm <= 3
    assert holm < raw


def test_correction_sets_adjusted_p_and_flips_only_fail_to_pass():
    from multioutreg.model_selection.screening import ModelScreener, TestResult

    def r(p):
        return TestResult("t", p >= 0.05, 0.0, p, "msg")

    s = ModelScreener(correction="holm")
    s._per_output = [{"normality": r(0.03), "linearity": r(0.001),
                      "heteroscedasticity": r(0.5)}]
    s._apply_correction()
    out = s._per_output[0]
    assert out["normality"].passed            # 0.03 * 2 = 0.06 -> not significant
    assert "holm" in out["normality"].message
    assert not out["linearity"].passed        # 0.001 * 3 = 0.003 -> still significant
    assert out["heteroscedasticity"].passed
    assert all(v.p_adjusted >= v.p_value for v in out.values())


def test_invalid_correction_rejected():
    import pytest
    from multioutreg.model_selection.screening import ModelScreener

    with pytest.raises(ValueError):
        ModelScreener(correction="bonferonni")


def test_breusch_pagan_skipped_when_mean_is_nonlinear():
    # Regression: BP on OLS residuals of a non-linear mean rejects even with
    # constant noise. It now only runs when RESET passes; otherwise it is
    # reported as not tested and heteroscedastic models stay eligible.
    from multioutreg.model_selection.screening import (
        ModelScreener, test_heteroscedasticity)

    rng = np.random.default_rng(0)
    X = rng.uniform(0, 3, size=(250, 1))
    y = 3 * np.sin(2 * X[:, 0]) + X[:, 0] ** 2 + 0.5 * rng.normal(size=250)
    assert not test_heteroscedasticity(X, y).passed   # the confounded result
    het = ModelScreener().fit(X, y)._per_output[0]["heteroscedasticity"]
    assert het.p_value is None and "not tested" in het.message


def test_breusch_pagan_detects_variance_on_linear_mean():
    from multioutreg.model_selection.screening import ModelScreener

    rng = np.random.default_rng(0)
    X = rng.uniform(0, 3, size=(300, 1))
    y = 2 * X[:, 0] + 0.5 * X[:, 0] * rng.normal(size=300)
    res = ModelScreener().fit(X, y)._per_output[0]
    assert res["linearity"].passed
    assert res["heteroscedasticity"].p_value is not None
    assert not res["heteroscedasticity"].passed


def _results(**failed):
    from multioutreg.model_selection.screening import TestResult

    return {k: TestResult(k, not failed.get(k, False), 0.0, None, "")
            for k in ("normality", "linearity", "heteroscedasticity", "nonlinear_dep")}


def test_screen_uses_nonlinear_dependency_and_vif():
    # Regression: screen() ignored the RF-LR gain test and never used VIF.
    from multioutreg.model_selection.screening import (
        ModelScreener, test_multicollinearity, test_sample_size)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    X = np.column_stack([X, X[:, 0] + 1e-3 * rng.normal(size=200)])
    s = ModelScreener()
    s._size = test_sample_size(X)
    s._vif = test_multicollinearity(X)
    s._per_output = [_results(nonlinear_dep=True)]
    s.global_flags_ = {"any_nonlinear": False, "any_nonnormal": False,
                       "any_heteroscedastic": False, "any_nonlinear_dep": True,
                       "multicollinear": not s._vif.passed}
    specs = {m.name: m for m in s.screen()}
    assert specs["RandomForest"].factory is not None
    assert "RF beats LR" in specs["RandomForest"].reason_included
    assert specs["BayesianRidge"].factory is not None
    assert "VIF" in specs["LinearRegression"].reason_included


def test_every_model_name_has_a_screening_rule():
    # Regression: names like sgpr, et, nystroem_gp, ngboost, catboost had no
    # rule and silently defaulted to eligible.
    from multioutreg.model_selection import AutoDetectMultiOutputRegressor
    from multioutreg.model_selection.screening import (
        _eligible_for_output_j, test_sample_size)

    rules = _eligible_for_output_j(_results(), test_sample_size(np.zeros((50, 2))))
    vendored = ["linear", "gp", "rf", "et", "gb", "svr", "knn", "dt", "mlp",
                "bayesian_ridge", "rfgp", "pbr", "nystroem_gp", "ard_gp", "hgb",
                "elastic_net", "lasso", "quantile", "ngboost", "gpx", "kpls",
                "lgbm", "xgb", "catboost", "sgpr"]
    gui = ["gpr", "rf", "et", "gb", "hgb", "knn", "blr", "svr", "dt", "cpn",
           "mfs_lr", "bayesian_ridge", "rfgp", "pbr", "sgp", "ard_gp",
           "elastic_net", "lasso", "quantile", "ngb", "gpx", "kpls", "lgbm",
           "xgb", "catboost", "sgpr"]
    model = AutoDetectMultiOutputRegressor.with_vendored_surrogates()
    missing = set(vendored + gui + model._model_names) - set(rules)
    assert not missing
    # linear, normal, homoscedastic data: tree models are not needed
    assert not rules["et"] and not rules["catboost"] and not rules["ngboost"]
    assert rules["sgpr"] and rules["nystroem_gp"] and rules["sgp"]


def test_pre_screen_works_without_vendored_names():
    # Regression: pre_screen=True was silently ignored unless the regressor
    # was built by with_vendored_surrogates().
    from sklearn import gaussian_process
    from sklearn.linear_model import LinearRegression
    from multioutreg.model_selection import AutoDetectMultiOutputRegressor

    class GaussianProcessRegressor(gaussian_process.GaussianProcessRegressor):
        # Same class name, so the screener maps it to the "gp" rule.
        def fit(self, X, y):
            raise AssertionError("GP should have been screened out")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2))       # N >= 300: GP is gated out
    y = X @ np.array([1.0, -2.0]) + 0.1 * rng.normal(size=400)
    model = AutoDetectMultiOutputRegressor(
        [LinearRegression(), GaussianProcessRegressor()], [{}, {}], pre_screen=True)
    model.fit(X, y)
    assert type(model.models_[0]).__name__ == "LinearRegression"


def test_ehet_std_is_heteroscedastic_and_covers():
    # Regression: _EHet's std was the spread of a bootstrap RF ensemble only
    # (epistemic, ~45% coverage of 90% intervals) and did not follow the noise.
    from multioutreg.model_selection.screening import _ehet

    rng = np.random.default_rng(0)
    X = rng.uniform(size=(1400, 2))
    noise = 0.1 + 0.5 * X[:, 0]
    y = np.sin(6 * X[:, 0]) + X[:, 1] + noise * rng.normal(size=1400)
    model = _ehet().fit(X[:400], y[:400])
    pred, std = model.predict(X[400:], return_std=True)
    coverage = np.mean(np.abs(y[400:] - pred) <= 1.645 * std)
    assert 0.83 <= coverage <= 0.97
    assert np.corrcoef(std, noise[400:])[0, 1] > 0.7
