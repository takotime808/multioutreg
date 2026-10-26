# Copyright (c) 2026 takotime808
"""Threshold boundaries and failure points of the screening statistics.

The screening functions are named ``test_*``; they are reached through the
module so pytest does not collect them as tests.
"""

import numpy as np
import pytest

import multioutreg.model_selection.screening as scr


def _X(n, p=2):
    return np.random.default_rng(0).normal(size=(n, p))


# ---------------------------------------------------------------------------
# Sample-size rules: exact threshold boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key, n, expected",
    [
        ("gp_feasible", 299, True),
        ("gp_feasible", 300, False),
        ("svr_feasible", 1999, True),
        ("svr_feasible", 2000, False),
        ("mlp_feasible", 100, False),
        ("mlp_feasible", 101, True),
        ("bootstrap_needed", 499, True),
        ("bootstrap_needed", 500, False),
    ],
)
def test_sample_size_thresholds_are_strict(key, n, expected):
    assert scr.test_sample_size(np.zeros((n, 1)))[key].passed is expected


@pytest.mark.parametrize("n, p, expected", [(20, 2, False), (21, 2, True)])
def test_knn_ratio_threshold_is_strict(n, p, expected):
    res = scr.test_sample_size(np.zeros((n, p)))["knn_ratio_ok"]
    assert res.passed is expected
    assert res.statistic == pytest.approx(n / p)


@pytest.mark.parametrize("p, expected", [(19, True), (20, False)])
def test_pbr_threshold_and_feature_count_message(p, expected):
    res = scr.test_sample_size(np.zeros((10, p)))["pbr_feasible"]
    assert res.passed is expected
    assert f"C(p+2,2)={(p + 2) * (p + 1) // 2}" in res.message


def test_sample_size_rules_have_no_p_value():
    assert all(r.p_value is None for r in scr.test_sample_size(np.zeros((5, 1))).values())


def test_sample_size_zero_features_fails():
    with pytest.raises(ZeroDivisionError):
        scr.test_sample_size(np.zeros((5, 0)))


# ---------------------------------------------------------------------------
# Statistical tests: p-values and statistics bounds
# ---------------------------------------------------------------------------


def test_normality_passes_on_gaussian_noise_and_p_in_unit_interval():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 2))
    y = X @ [1.0, -1.0] + rng.normal(size=200)
    res = scr.test_normality(X, y)
    assert 0 <= res.p_value <= 1
    assert 0 < res.statistic <= 1  # Shapiro W
    assert res.passed


def test_normality_fails_on_heavy_tailed_residuals():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 1))
    y = X[:, 0] + rng.standard_cauchy(300)
    assert not scr.test_normality(X, y).passed


def test_normality_switches_to_dagostino_above_5000():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(5001, 1))
    res = scr.test_normality(X, X[:, 0] + rng.normal(size=5001))
    assert "D'Agostino" in res.name


def test_normality_alpha_zero_always_passes_and_alpha_one_always_fails():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(50, 1))
    y = X[:, 0] + rng.normal(size=50)
    assert scr.test_normality(X, y, alpha=0.0).passed
    assert not scr.test_normality(X, y, alpha=1.0).passed


def test_normality_too_few_points_is_nan_and_conservative_fail():
    # Shapiro-Wilk needs 3 points; with 2 it yields NaN, which compares as a
    # failure, so non-linear models stay eligible (no crash, no false PASS).
    res = scr.test_normality(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))
    assert np.isnan(res.statistic) and np.isnan(res.p_value)
    assert not res.passed


def test_linearity_detects_quadratic_and_accepts_linear():
    rng = np.random.default_rng(5)
    X = rng.uniform(-2, 2, size=(200, 1))
    lin = scr.test_linearity(X, 2 * X[:, 0] + 0.1 * rng.normal(size=200))
    quad = scr.test_linearity(X, X[:, 0] ** 2 + 0.1 * rng.normal(size=200))
    assert lin.passed and not quad.passed
    assert 0 <= lin.p_value <= 1 and 0 <= quad.p_value <= 1


def test_linearity_falls_back_to_rank_gap_when_reset_fails(monkeypatch):
    import statsmodels.stats.diagnostic as diag

    def boom(*a, **k):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(diag, "linear_reset", boom)
    rng = np.random.default_rng(6)
    X = rng.normal(size=(60, 1))
    res = scr.test_linearity(X, 3 * X[:, 0])
    assert "Spearman" in res.name
    assert res.p_value is None
    assert res.statistic >= 0 and res.passed


def test_heteroscedasticity_statistic_non_negative():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(100, 2))
    res = scr.test_heteroscedasticity(X, X.sum(axis=1) + rng.normal(size=100))
    assert res.statistic >= 0 and 0 <= res.p_value <= 1


def test_heteroscedasticity_not_tested_is_a_conservative_fail():
    res = scr.heteroscedasticity_not_tested(alpha=0.1)
    assert res.passed is False and np.isnan(res.statistic) and res.alpha == 0.1


def test_vif_is_at_least_one_for_independent_features():
    res = scr.test_multicollinearity(_X(200, 3))
    assert res.statistic >= 1.0 and res.passed


def test_vif_perfect_collinearity_fails():
    X = _X(50, 2)
    X = np.column_stack([X, X[:, 0] + X[:, 1]])
    res = scr.test_multicollinearity(X)
    assert not res.passed
    assert res.statistic > 1e6 or np.isinf(res.statistic)


def test_vif_threshold_boundary_is_strict():
    X = _X(100, 3)
    vif = scr.test_multicollinearity(X).statistic
    assert not scr.test_multicollinearity(X, threshold=vif).passed
    assert scr.test_multicollinearity(X, threshold=vif + 1e-9).passed


def test_nonlinear_dependency_gain_sign():
    rng = np.random.default_rng(8)
    X = rng.uniform(-3, 3, size=(150, 1))
    lin = scr.test_nonlinear_dependency(X, 2 * X[:, 0] + 0.1 * rng.normal(size=150))
    nonlin = scr.test_nonlinear_dependency(X, np.sin(2 * X[:, 0]) + 0.05 * rng.normal(size=150))
    assert lin.passed and not nonlin.passed
    assert nonlin.statistic > lin.statistic


def test_nonlinear_dependency_more_folds_than_samples_raises():
    with pytest.raises(ValueError):
        scr.test_nonlinear_dependency(_X(4, 1), np.arange(4.0), cv=10)


# ---------------------------------------------------------------------------
# TestResult repr
# ---------------------------------------------------------------------------


def test_result_repr_shows_status_and_adjusted_p():
    r = scr.TestResult("x", False, 1.5, 0.01, "msg", p_adjusted=0.03)
    s = repr(r)
    assert "FAIL" in s and "p=0.0100" in s and "p_adj=0.0300" in s


def test_result_repr_without_p_value():
    s = repr(scr.TestResult("x", True, 2.0, None, "ok"))
    assert "PASS" in s and "p=" not in s


# ---------------------------------------------------------------------------
# ModelScreener failure points
# ---------------------------------------------------------------------------


def test_screener_methods_before_fit_raise():
    s = scr.ModelScreener()
    with pytest.raises(RuntimeError, match="fit"):
        s.screen_per_output()
    with pytest.raises(RuntimeError, match="fit"):
        s.eligible_indices_for_output(0, ["LinearRegression"])


def test_screener_unknown_model_name_is_always_tried_with_warning():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(60, 2))
    Y = np.column_stack([X[:, 0], X[:, 1] ** 2]) + 0.1 * rng.normal(size=(60, 2))
    s = scr.ModelScreener().fit(X, Y)
    with pytest.warns(UserWarning, match="No screening rule"):
        idx = s.eligible_indices_for_output(0, ["NotARealModel"])
    assert idx == [0]


def test_screener_per_output_matrix_is_boolean_with_given_names():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(60, 2))
    Y = np.column_stack([X[:, 0], X[:, 1]]) + 0.1 * rng.normal(size=(60, 2))
    df = scr.ModelScreener().fit(X, Y).screen_per_output(["a", "b"])
    assert list(df.columns) == ["a", "b"]
    assert df.dtypes.map(lambda t: t == bool).all()
