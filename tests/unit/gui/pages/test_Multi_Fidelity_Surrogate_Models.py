# Copyright (c) 2026 takotime808

import importlib.util
import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Load the page module without running Streamlit
# ---------------------------------------------------------------------------

_PAGE_PATH = os.path.abspath(
    "multioutreg/gui/pages/02_Multi_Fidelity_Surrogate_Models.py"
)
_spec = importlib.util.spec_from_file_location("mf_page", _PAGE_PATH)
_mf_page = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mf_page)

_split_by_fidelity = _mf_page._split_by_fidelity
_default_level_order = _mf_page._default_level_order
_compute_metrics = _mf_page._compute_metrics
generate_html_report = _mf_page.generate_html_report


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_level_csv_df():
    """DataFrame with integer fidelity column, 2 inputs, 1 output, 2 levels."""
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
        "fidelity": [0] * (n // 2) + [1] * (n // 2),
        "y": rng.standard_normal(n),
    })
    return df


@pytest.fixture
def three_level_csv_df():
    """DataFrame with integer fidelity column, 3 levels, 2 outputs."""
    rng = np.random.default_rng(7)
    n_per = 40
    rows = []
    for lvl in range(3):
        X = rng.standard_normal((n_per, 2))
        Y = X + lvl * 0.1
        for i in range(n_per):
            rows.append({"x1": X[i, 0], "x2": X[i, 1], "fidelity": lvl,
                         "y1": Y[i, 0], "y2": Y[i, 1]})
    return pd.DataFrame(rows)


@pytest.fixture
def string_fidelity_df():
    """DataFrame with string fidelity column."""
    rng = np.random.default_rng(3)
    n = 50
    df = pd.DataFrame({
        "x1": rng.standard_normal(n),
        "fidelity": ["lo"] * (n // 2) + ["hi"] * (n // 2),
        "y": rng.standard_normal(n),
    })
    return df


# ---------------------------------------------------------------------------
# _split_by_fidelity tests
# ---------------------------------------------------------------------------


def test_split_by_fidelity_two_levels(two_level_csv_df):
    df = two_level_csv_df
    result = _split_by_fidelity(df, "fidelity", ["x1", "x2"], ["y"], [0, 1])
    assert set(result.keys()) == {"0", "1"}
    X0, Y0 = result["0"]
    X1, Y1 = result["1"]
    assert X0.shape == (30, 2)
    assert Y0.shape == (30, 1)
    assert X1.shape == (30, 2)


def test_split_by_fidelity_three_levels(three_level_csv_df):
    df = three_level_csv_df
    result = _split_by_fidelity(df, "fidelity", ["x1", "x2"], ["y1", "y2"], [0, 1, 2])
    assert set(result.keys()) == {"0", "1", "2"}
    for k in ["0", "1", "2"]:
        X_k, Y_k = result[k]
        assert X_k.shape[1] == 2
        assert Y_k.shape[1] == 2


def test_split_by_fidelity_string_levels(string_fidelity_df):
    """String fidelity values work the same as integers."""
    df = string_fidelity_df
    result = _split_by_fidelity(df, "fidelity", ["x1"], ["y"], ["lo", "hi"])
    assert set(result.keys()) == {"lo", "hi"}
    X_lo, _ = result["lo"]
    X_hi, _ = result["hi"]
    assert X_lo.shape == (25, 1)
    assert X_hi.shape == (25, 1)


def test_split_by_fidelity_preserves_order(three_level_csv_df):
    """Returns exactly the levels in ordered_levels, in order."""
    df = three_level_csv_df
    result = _split_by_fidelity(df, "fidelity", ["x1", "x2"], ["y1", "y2"], [2, 0, 1])
    assert list(result.keys()) == ["2", "0", "1"]


def test_split_by_fidelity_float_dtype(two_level_csv_df):
    """Output arrays are float64."""
    df = two_level_csv_df
    result = _split_by_fidelity(df, "fidelity", ["x1", "x2"], ["y"], [0, 1])
    for X_k, Y_k in result.values():
        assert X_k.dtype == np.float64
        assert Y_k.dtype == np.float64


# ---------------------------------------------------------------------------
# _compute_metrics tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values, expected",
    [
        ([2, 10, 0, 1, 10, None], [0, 1, 2, 10]),  # numeric, not "10" < "2"
        ([1.5, 0.0, 0.25], [0.0, 0.25, 1.5]),
        (np.array([3, 0, 11, 2]), [0, 2, 3, 11]),
        (["hi", "lo", "med"], ["hi", "lo", "med"]),  # text: user re-orders
        ([1, "lo"], [1, "lo"]),  # mixed falls back to text order
        ([], []),
    ],
)
def test_default_level_order(values, expected):
    assert _default_level_order(values) == expected


def test_compute_metrics_keys():
    rng = np.random.default_rng(0)
    y_test = rng.standard_normal((20, 2))
    y_pred = y_test + rng.standard_normal((20, 2)) * 0.1
    y_std = np.abs(rng.standard_normal((20, 2))) * 0.05
    metrics = _compute_metrics(y_test, y_pred, y_std, ["out1", "out2"])
    assert set(metrics.keys()) == {"out1", "out2"}
    for name in ["out1", "out2"]:
        assert set(metrics[name].keys()) == {"r2", "rmse", "mae", "mean_predicted_std"}


def test_compute_metrics_perfect_prediction():
    y = np.ones((10, 1)) * 3.0
    metrics = _compute_metrics(y, y, np.zeros((10, 1)), ["out"])
    assert pytest.approx(metrics["out"]["r2"], abs=1e-6) == 1.0
    assert pytest.approx(metrics["out"]["rmse"], abs=1e-10) == 0.0
    assert pytest.approx(metrics["out"]["mae"], abs=1e-10) == 0.0


def test_compute_metrics_std_recorded():
    rng = np.random.default_rng(1)
    y = rng.standard_normal((15, 1))
    std = np.full((15, 1), 0.5)
    metrics = _compute_metrics(y, y, std, ["out"])
    assert pytest.approx(metrics["out"]["mean_predicted_std"]) == 0.5


# ---------------------------------------------------------------------------
# End-to-end fit tests (pure Python, no Streamlit)
# ---------------------------------------------------------------------------


def test_stacked_vfm_fits_and_predicts(two_level_csv_df):
    """StackedVFMSurrogate is instantiated and fitted correctly from level_data."""
    from multioutreg.surrogates import RandomForestSurrogate, StackedVFMSurrogate
    from sklearn.model_selection import train_test_split

    df = two_level_csv_df
    level_data = _split_by_fidelity(df, "fidelity", ["x1", "x2"], ["y"], [0, 1])
    level_names = ["0", "1"]

    train_data, test_data = {}, {}
    for lv, (X_k, Y_k) in level_data.items():
        X_tr, X_te, Y_tr, Y_te = train_test_split(X_k, Y_k, test_size=0.25, random_state=0)
        train_data[lv] = (X_tr, Y_tr)
        test_data[lv] = (X_te, Y_te)

    model = StackedVFMSurrogate(
        fidelity_levels=level_names, surrogate_cls=RandomForestSurrogate
    )
    model.fit(train_data)

    X_test, y_test = test_data["1"]
    y_pred, y_std = model.predict(X_test, return_std=True)
    assert y_pred.shape == y_test.shape
    assert y_std.shape == y_test.shape


def test_additive_correction_fits_two_levels(two_level_csv_df):
    """AdditiveCorrectionVFM fits from lo/hi split."""
    from multioutreg.surrogates import AdditiveCorrectionVFM, RandomForestSurrogate
    from sklearn.model_selection import train_test_split

    df = two_level_csv_df
    level_data = _split_by_fidelity(df, "fidelity", ["x1", "x2"], ["y"], [0, 1])
    level_names = ["0", "1"]

    train_data: dict = {}
    test_data: dict = {}
    for lv, (X_k, Y_k) in level_data.items():
        X_tr, X_te, Y_tr, Y_te = train_test_split(X_k, Y_k, test_size=0.25, random_state=0)
        train_data[lv] = (X_tr, Y_tr)
        test_data[lv] = (X_te, Y_te)

    lo_name, hi_name = level_names[0], level_names[1]
    model = AdditiveCorrectionVFM(
        lo_surrogate_cls=RandomForestSurrogate,
        hi_surrogate_cls=RandomForestSurrogate,
    )
    model.fit({"lo": train_data[lo_name], "hi": train_data[hi_name]})

    X_test, y_test = test_data[hi_name]
    y_pred, y_std = model.predict(X_test, return_std=True)
    assert y_pred.shape == y_test.shape


def test_multi_fidelity_surrogate_fits(two_level_csv_df):
    """MultiFidelitySurrogate (independent) fits from level_data."""
    from multioutreg.surrogates import MultiFidelitySurrogate, RandomForestSurrogate
    from sklearn.model_selection import train_test_split

    df = two_level_csv_df
    level_data = _split_by_fidelity(df, "fidelity", ["x1", "x2"], ["y"], [0, 1])
    level_names = ["0", "1"]

    train_data, test_data = {}, {}
    for lv, (X_k, Y_k) in level_data.items():
        X_tr, X_te, Y_tr, Y_te = train_test_split(X_k, Y_k, test_size=0.25, random_state=0)
        train_data[lv] = (X_tr, Y_tr)
        test_data[lv] = (X_te, Y_te)

    model = MultiFidelitySurrogate(RandomForestSurrogate, level_names)
    model.fit(train_data)

    X_test, y_test = test_data["1"]
    y_pred = model.predict(X_test, level="1")
    assert y_pred.shape == y_test.shape


def test_three_level_stacked_vfm(three_level_csv_df):
    """Three-level stacked VFM works end-to-end."""
    from multioutreg.surrogates import RandomForestSurrogate, StackedVFMSurrogate
    from sklearn.model_selection import train_test_split

    df = three_level_csv_df
    level_data = _split_by_fidelity(df, "fidelity", ["x1", "x2"], ["y1", "y2"], [0, 1, 2])
    level_names = ["0", "1", "2"]

    train_data, test_data = {}, {}
    for lv, (X_k, Y_k) in level_data.items():
        X_tr, X_te, Y_tr, Y_te = train_test_split(X_k, Y_k, test_size=0.25, random_state=0)
        train_data[lv] = (X_tr, Y_tr)
        test_data[lv] = (X_te, Y_te)

    model = StackedVFMSurrogate(
        fidelity_levels=level_names, surrogate_cls=RandomForestSurrogate
    )
    model.fit(train_data)

    X_test, y_test = test_data["2"]
    y_pred, y_std = model.predict(X_test, return_std=True)
    assert y_pred.shape == y_test.shape


def test_metrics_evaluated_on_highest_fidelity(two_level_csv_df):
    """Confirm that metrics are computed against the highest-fidelity test set."""
    from multioutreg.surrogates import RandomForestSurrogate, StackedVFMSurrogate
    from sklearn.model_selection import train_test_split

    df = two_level_csv_df
    level_data = _split_by_fidelity(df, "fidelity", ["x1", "x2"], ["y"], [0, 1])
    level_names = ["0", "1"]

    train_data, test_data = {}, {}
    for lv, (X_k, Y_k) in level_data.items():
        X_tr, X_te, Y_tr, Y_te = train_test_split(X_k, Y_k, test_size=0.25, random_state=0)
        train_data[lv] = (X_tr, Y_tr)
        test_data[lv] = (X_te, Y_te)

    model = StackedVFMSurrogate(fidelity_levels=level_names, surrogate_cls=RandomForestSurrogate)
    model.fit(train_data)

    X_test, y_test = test_data[level_names[-1]]  # highest fidelity
    y_pred, y_std = model.predict(X_test, return_std=True)
    y_pred = np.asarray(y_pred)
    y_std = np.asarray(y_std)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_std.ndim == 1:
        y_std = y_std.reshape(-1, 1)

    metrics = _compute_metrics(y_test, y_pred, y_std, ["y"])
    assert "y" in metrics
    assert -1.0 <= metrics["y"]["r2"] <= 1.0  # valid R² range


# ---------------------------------------------------------------------------
# generate_html_report test
# ---------------------------------------------------------------------------


def test_generate_html_report_runs(tmp_path):
    """generate_html_report returns HTML without error."""
    from multioutreg.surrogates import RandomForestSurrogate

    rng = np.random.default_rng(42)
    n = 20
    X = rng.standard_normal((n, 2))
    Y = rng.standard_normal((n, 1))
    model = RandomForestSurrogate()
    model.fit(X, Y)
    y_pred = model.predict(X)
    if isinstance(y_pred, tuple):
        y_pred = y_pred[0]
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    y_std = np.zeros_like(y_pred)
    metrics = _compute_metrics(Y, y_pred, y_std, ["out"])

    template_path = tmp_path / "template.html"
    template_path.write_text("{{ model_type }} report", encoding="utf-8")
    original = os.environ.get("MOR_TEMPLATE_PATH")
    os.environ["MOR_TEMPLATE_PATH"] = str(template_path)
    try:
        html = generate_html_report(
            model_type="StackedVFMSurrogate",
            fidelity_levels=["0", "1"],
            output_names=["out"],
            description="test",
            metrics=metrics,
            uncertainty_metrics={},
            y_test=Y,
            best_pred=y_pred,
            best_std=y_std,
            best_model=model,
            X_train=X,
            n_train=n,
            n_test=n,
            cross_validation="None",
            seed=0,
            notes="test",
        )
    finally:
        if original is not None:
            os.environ["MOR_TEMPLATE_PATH"] = original
        else:
            del os.environ["MOR_TEMPLATE_PATH"]

    assert isinstance(html, str)
    assert "StackedVFMSurrogate" in html


# ---------------------------------------------------------------------------
# Model selection (audit 1c): pick on validation, never on the test set
# ---------------------------------------------------------------------------

_select_best_candidate = _mf_page._select_best_candidate
_predict_highest = _mf_page._predict_highest
_HighestFidelityView = _mf_page._HighestFidelityView
_compute_uncertainty_metrics = _mf_page._compute_uncertainty_metrics


class _MeanSurrogate:
    """Predicts the training mean; also records the rows it was fitted on."""

    fitted_rows: list = []

    def fit(self, X, Y):
        _MeanSurrogate.fitted_rows.append(np.asarray(X).copy())
        self.mu = np.asarray(Y).mean(axis=0)
        return self

    def predict(self, X, return_std=False):
        pred = np.tile(self.mu, (len(X), 1))
        return (pred, np.zeros_like(pred)) if return_std else pred


def _linear_train_data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    data = {}
    for lv, sign in (("lo", -1.0), ("hi", 1.0)):
        X = rng.standard_normal((n, 1))
        data[lv] = (X, sign * 3.0 * X)
    return data


def _independent(name, sur_cls):
    return {"name": name, "mf_type": "independent",
            "surrogate_cls": sur_cls, "augment_with_std": False}


def test_select_best_candidate_scores_on_validation():
    from multioutreg.surrogates import LinearRegressionSurrogate

    train_data = _linear_train_data()
    results, best, _ = _select_best_candidate(
        [_independent("mean", _MeanSurrogate),
         _independent("linear", LinearRegressionSurrogate)],
        train_data, ["lo", "hi"],
    )
    assert best["name"] == "linear"
    assert {"CV MSE", "CV RMSE", "CV R²"} <= set(results[0])
    assert all(r["Status"] == "OK" for r in results)
    assert results[1]["CV R²"] > 0.99 and results[0]["CV R²"] < 0.1


def test_select_best_candidate_cross_validates_hifi_rows_only():
    from sklearn.model_selection import KFold

    _MeanSurrogate.fitted_rows = []
    train_data = _linear_train_data(n=40)
    _select_best_candidate(
        [_independent("mean", _MeanSurrogate)], train_data, ["lo", "hi"],
        cv=KFold(4, shuffle=True, random_state=0),
    )
    fits = _MeanSurrogate.fitted_rows
    assert len(fits) == 8              # (lo, hi) per fold
    lo_rows, hi_rows = fits[0::2], fits[1::2]
    assert all(len(r) == 40 for r in lo_rows)  # lower levels use all their data
    assert all(len(r) == 30 for r in hi_rows)  # 1/4 of hi-fi rows held out per fold
    hi_X = set(map(float, train_data["hi"][0].ravel()))
    held_out = [hi_X - set(map(float, r.ravel())) for r in hi_rows]
    assert set().union(*held_out) == hi_X      # every hi-fi row validated once
    assert sum(len(h) for h in held_out) == 40


@pytest.mark.parametrize("n, expected", [
    (12, "Repeated 3-Fold CV (x3 repeats)"),
    (60, "Repeated 5-Fold CV (x3 repeats)"),
])
def test_select_best_candidate_auto_cv_depends_on_hifi_size(n, expected):
    from multioutreg.model_selection.auto_detect import describe_cv

    _, best, splitter = _select_best_candidate(
        [_independent("mean", _MeanSurrogate)], _linear_train_data(n=n), ["lo", "hi"]
    )
    assert best["name"] == "mean"
    assert describe_cv(splitter) == expected


def test_select_best_candidate_too_few_hifi_samples():
    train_data = _linear_train_data(n=4)
    with pytest.raises(ValueError, match="at least 5"):
        _select_best_candidate(
            [_independent("mean", _MeanSurrogate)], train_data, ["lo", "hi"]
        )


def test_predict_highest_uses_highest_level_for_independent():
    from multioutreg.surrogates import LinearRegressionSurrogate, MultiFidelitySurrogate

    train_data = _linear_train_data()
    cand = _independent("linear", LinearRegressionSurrogate)
    model = MultiFidelitySurrogate(LinearRegressionSurrogate, ["lo", "hi"]).fit(train_data)
    X = np.array([[1.0]])
    y_pred, y_std = _predict_highest(model, cand, X, ["lo", "hi"])
    # hi level is y = 3x; lo level is y = -3x
    assert y_pred.shape == (1, 1) and y_std.shape == (1, 1)
    assert y_pred[0, 0] == pytest.approx(3.0, abs=1e-6)


def test_highest_fidelity_view_explains_highest_level():
    """SHAP / PDP must see the highest level, not MultiFidelitySurrogate's default."""
    from sklearn.base import is_regressor
    from sklearn.inspection import partial_dependence
    from multioutreg.surrogates import LinearRegressionSurrogate, MultiFidelitySurrogate

    train_data = _linear_train_data()
    cand = _independent("linear", LinearRegressionSurrogate)
    model = MultiFidelitySurrogate(LinearRegressionSurrogate, ["lo", "hi"]).fit(train_data)
    view = _HighestFidelityView(model, cand, ["lo", "hi"], n_outputs=1)
    X = np.array([[1.0], [2.0]])
    np.testing.assert_allclose(view.predict(X)[:, 0], [3.0, 6.0], atol=1e-6)
    est = view.estimators_[0]
    assert is_regressor(est)
    np.testing.assert_allclose(est.predict(X), [3.0, 6.0], atol=1e-6)
    pd_res = partial_dependence(est, X, [0], grid_resolution=2)
    np.testing.assert_allclose(pd_res["average"][0], [3.0, 6.0], atol=1e-6)


def test_compute_uncertainty_metrics():
    y = np.zeros((4, 1))
    pred = np.array([[0.0], [0.0], [0.0], [3.0]])
    std = np.ones((4, 1))
    m = _compute_uncertainty_metrics(y, pred, std)
    assert m["Empirical coverage of 95% Gaussian interval"] == pytest.approx(0.75)
    assert m["Miscoverage at 95% (|nominal − empirical|)"] == pytest.approx(0.2)
    assert m["Sharpness (mean predicted std)"] == pytest.approx(1.0)
    assert "Gaussian NLL (mean per point)" in m
    # NLL is omitted when any std is zero (no native uncertainty)
    assert "Gaussian NLL (mean per point)" not in _compute_uncertainty_metrics(
        y, pred, np.zeros((4, 1))
    )


# ---------------------------------------------------------------------------
# Report summary, fidelity-level table, HF-only baseline, model selection
# ---------------------------------------------------------------------------

_single_level_predictions = _mf_page._single_level_predictions
_level_summary = _mf_page._level_summary
_baseline_comparison = _mf_page._baseline_comparison
_summary_points = _mf_page._summary_points


def _gs_rows():
    return [
        {"Model": "B", "CV MSE": 2.0, "CV RMSE": 1.41, "CV R²": 0.5, "Status": "OK"},
        {"Model": "A", "CV MSE": 1.0, "CV RMSE": 1.0, "CV R²": 0.8, "Status": "OK"},
        {"Model": "C", "CV MSE": None, "CV RMSE": None, "CV R²": None, "Status": "Error: x"},
    ]


def _summary(mf_type="stacked", mf_rmse=1.0, hf_rmse=2.0, r=0.9, cov=0.95, n_test=50,
             std=1.0):
    metrics = {"y": {"r2": 0.9, "rmse": mf_rmse, "mae": 0.5, "mean_predicted_std": std}}
    base = {"y": {"r2": 0.8, "rmse": hf_rmse, "mae": 0.6, "mean_predicted_std": std}}
    levels = [
        {"level": "lo", "output": "y", "r": r},
        {"level": "hi", "output": "y", "r": 0.99},
    ]
    unc = {"Empirical coverage of 95% Gaussian interval": cov}
    return _summary_points(
        {"name": "A", "mf_type": mf_type}, _gs_rows(), metrics,
        _baseline_comparison(metrics, base), "RF", levels, ["lo", "hi"], unc,
        np.full((n_test, 1), std), n_test,
    )


def test_summary_best_and_runner_up():
    first = _summary()[0]
    assert "Best configuration: A" in first
    assert "runner-up B is 100% worse" in first
    assert "1 failed to fit" in first


def test_summary_mf_vs_baseline_branches():
    assert any("reduced mean test RMSE by 50%" in p for p in _summary(mf_rmse=1.0, hf_rmse=2.0))
    assert any("100% worse than an HF-only RF" in p for p in _summary(mf_rmse=2.0, hf_rmse=1.0))
    assert any("within 5%" in p for p in _summary(mf_rmse=1.0, hf_rmse=1.02))
    assert any("is the HF-only RF model" in p for p in _summary(mf_type="independent"))


def test_summary_levels_uncertainty_and_small_test_set():
    assert any("Weakly related lower level(s): 'lo'" in p for p in _summary(r=0.2))
    assert any("track the highest fidelity closely" in p for p in _summary(r=0.9))
    assert any("overconfident" in p for p in _summary(cov=0.5))
    assert any("conservative" in p for p in _summary(cov=1.0))
    assert any("close to nominal" in p for p in _summary(cov=0.94))
    assert any("no predictive uncertainty" in p for p in _summary(std=0.0))
    assert any("only 10 highest-fidelity points" in p for p in _summary(n_test=10))
    assert not any("highest-fidelity points" in p for p in _summary(n_test=50))


def test_single_level_predictions_and_level_summary():
    from multioutreg.surrogates import LinearRegressionSurrogate

    train_data = _linear_train_data()          # lo: y = -3x, hi: y = 3x
    X_test = np.linspace(-1, 1, 10).reshape(-1, 1)
    y_test = 3.0 * X_test
    preds = _single_level_predictions(LinearRegressionSurrogate, train_data, X_test)
    np.testing.assert_allclose(preds["hi"][0], y_test, atol=1e-6)
    test_data = {lv: (X_test, y_test) for lv in train_data}
    rows = _level_summary(train_data, test_data, preds, y_test, ["y"])
    by_level = {r["level"]: r for r in rows}
    assert by_level["lo"]["r"] == pytest.approx(-1.0)   # perfectly (anti-)correlated
    assert by_level["lo"]["r2"] < 0                     # but useless as-is
    assert by_level["hi"]["r2"] == pytest.approx(1.0)
    assert by_level["hi"]["n_test"] == 10


def test_single_level_predictions_reuses_fitted_models_and_reports_errors():
    class _Boom:
        def fit(self, X, Y):
            raise ValueError("boom")

    class _Fixed:
        def predict(self, X, return_std=False):
            return np.ones((len(X), 1)), np.zeros((len(X), 1))

    train_data = _linear_train_data()
    out = _single_level_predictions(
        _Boom, train_data, np.zeros((3, 1)), fitted_models={"hi": _Fixed()}
    )
    assert out["lo"] == "ValueError: boom"
    np.testing.assert_allclose(out["hi"][0], 1.0)


def test_report_renders_new_sections_with_real_template():
    from multioutreg.surrogates import LinearRegressionSurrogate

    rng = np.random.default_rng(0)
    X = rng.random((30, 2))
    Y = X @ np.array([[1.0], [2.0]])
    model = LinearRegressionSurrogate().fit(X, Y)
    pred, std = model.predict(X, return_std=True)
    pred, std = np.asarray(pred).reshape(-1, 1), np.asarray(std).reshape(-1, 1) + 0.1
    metrics = _compute_metrics(Y, pred, std, ["y"])
    common = dict(
        model_type="m", fidelity_levels=["lo", "hi"], output_names=["y"],
        description="", metrics=metrics, uncertainty_metrics={}, y_test=Y,
        best_pred=pred, best_std=std, best_model=model, X_train=X, n_train=30,
        n_test=30, cross_validation="", seed=0, notes="", input_names=["a", "b"],
        report_style="classic",
    )
    plain = generate_html_report(**common)
    for heading in ("<h2>Summary", "<h2>Fidelity Levels", "High-Fidelity-Only Baseline",
                    "<h2>Model Selection"):
        assert heading not in plain
    html = generate_html_report(
        **common,
        summary_points=["Point one."],
        level_summary=[{"level": "lo", "output": "y", "n_train": 5, "n_test": 2,
                        "mean": 0.0, "std": 1.0, "r": None, "r2": None,
                        "error": "ValueError: boom"},
                       {"level": "hi", "output": "y", "n_train": 5, "n_test": 2,
                        "mean": 0.0, "std": 1.0, "r": 0.5, "r2": 0.25, "error": None}],
        baseline_comparison=_baseline_comparison(metrics, metrics),
        baseline_label="Linear",
        grid_search_results=_gs_rows(),
        best_candidate_name="A",
    )
    assert "<li>Point one.</li>" in html
    assert "ValueError: boom" in html and "0.250" in html
    assert "+0.0%" in html and "(Linear)" in html
    assert 'class="best"' in html and "Error: x" in html


def test_v2_report_renders_mf_sections():
    from multioutreg.surrogates import LinearRegressionSurrogate

    rng = np.random.default_rng(0)
    X = rng.random((30, 2))
    Y = X @ np.array([[1.0], [2.0]])
    model = LinearRegressionSurrogate().fit(X, Y)
    pred, std = model.predict(X, return_std=True)
    pred, std = np.asarray(pred).reshape(-1, 1), np.asarray(std).reshape(-1, 1) + 0.1
    metrics = _compute_metrics(Y, pred, std, ["y"])
    html = generate_html_report(
        model_type="m", fidelity_levels=["lo", "hi"], output_names=["y"],
        description="", metrics=metrics, uncertainty_metrics={}, y_test=Y,
        best_pred=pred, best_std=std, best_model=model, X_train=X, n_train=30,
        n_test=30, cross_validation="", seed=0, notes="", input_names=["a", "b"],
        summary_points=["Point one."],
        level_summary=[{"level": "lo", "output": "y", "n_train": 5, "n_test": 2,
                        "mean": 0.0, "std": 1.0, "r": None, "r2": None,
                        "error": "ValueError: boom"},
                       {"level": "hi", "output": "y", "n_train": 5, "n_test": 2,
                        "mean": 0.0, "std": 1.0, "r": 0.5, "r2": 0.25, "error": None}],
        baseline_comparison=_baseline_comparison(metrics, metrics),
        baseline_label="Linear",
        grid_search_results=_gs_rows(),
        best_candidate_name="A",
        report_style="v2",
    )
    assert 'id="report-data"' in html and "Verdict" in html
    assert "<li>Point one.</li>" in html
    assert "ValueError: boom" in html and "(Linear)" in html
    assert "No gain" in html  # MF RMSE equals the HF-only RMSE
    assert 'class="best"' in html and "Error: x" in html


# ---------------------------------------------------------------------------
# Random seed control
# ---------------------------------------------------------------------------

_build_candidates = _mf_page._build_candidates
_seeded = _mf_page._seeded


def test_seeded_factory_kwargs():
    from multioutreg.surrogates import LinearRegressionSurrogate, RandomForestSurrogate
    from multioutreg.surrogates.catboost_sklearn import CatBoostSurrogate

    assert _seeded(RandomForestSurrogate, None) is RandomForestSurrogate
    assert _seeded(RandomForestSurrogate, 3).keywords == {"random_state": 3}
    assert _seeded(CatBoostSurrogate, 3).keywords == {"random_seed": 3}
    assert _seeded(LinearRegressionSurrogate, 3) is LinearRegressionSurrogate


def test_every_seeded_candidate_constructs_and_fits():
    """Each base surrogate must accept the seed argument it is given."""
    rng = np.random.default_rng(0)
    X, Y = rng.random((40, 2)), rng.random((40, 1))
    seen = set()
    for cand in _build_candidates(["lo", "hi"], skip_expensive=False, seed=7):
        factory = cand["surrogate_cls"]
        key = getattr(factory, "func", factory)
        if key in seen:
            continue
        seen.add(key)
        factory().fit(X, Y)


def test_seeded_candidates_are_reproducible():
    from multioutreg.surrogates import MultiFidelitySurrogate, RandomForestSurrogate
    from multioutreg.surrogates.sparse_gp_gpytorch import _GPYTORCH_AVAILABLE, SparseGPSurrogate

    train_data = _linear_train_data()
    X = np.linspace(-1, 1, 5).reshape(-1, 1)
    classes = [RandomForestSurrogate] + ([SparseGPSurrogate] if _GPYTORCH_AVAILABLE else [])
    for cls in classes:
        preds = [
            MultiFidelitySurrogate(_seeded(cls, 11), ["lo", "hi"])
            .fit(train_data).predict(X, level="hi")
            for _ in range(2)
        ]
        np.testing.assert_array_equal(preds[0], preds[1])


def test_additive_candidates_offer_fitted_and_unit_rho():
    """Audit 28: AdditiveVFM is offered with fitted rho and with rho=1."""
    cands = [c for c in _build_candidates(["lo", "hi"], skip_expensive=True, seed=0)
             if c["mf_type"] == "additive"]
    by_name = {c["name"]: c for c in cands}
    assert by_name["AdditiveVFM+RF"]["rho"] == "fit"
    assert by_name["AdditiveVFM+RF+rho1"]["rho"] == 1.0
    model = _mf_page._fit_candidate(by_name["AdditiveVFM+RF+rho1"], _linear_train_data(), ["lo", "hi"])
    np.testing.assert_array_equal(model.rho_, 1.0)


# ---------------------------------------------------------------------------
# Sampling section covers every fidelity level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_dim", [1, 2, 3])
def test_sampling_plot_covers_every_level(n_dim):
    rng = np.random.default_rng(0)
    designs = {"lo": rng.random((40, n_dim)) * 5, "mid": rng.random((20, n_dim)), "hi": rng.random((8, n_dim))}
    img, explanation = _mf_page._sampling_plot(designs, [f"x{j}" for j in range(n_dim)])
    assert img
    for lv, X in designs.items():
        assert f"<b>{lv}</b> (n={len(X)})" in explanation


def test_sampling_plot_escapes_level_names():
    rng = np.random.default_rng(0)
    _, explanation = _mf_page._sampling_plot({"<lo>": rng.random((10, 2))}, ["a", "b"])
    assert "&lt;lo&gt;" in explanation and "<lo>" not in explanation


def test_conformal_summary_chart_in_report(tmp_path):
    """The report's conformal section includes the summary chart."""
    from multioutreg.surrogates import RandomForestSurrogate

    rng = np.random.default_rng(0)
    X = rng.random((30, 2))
    Y = np.c_[X[:, 0], X[:, 1]]
    model = RandomForestSurrogate().fit(X, Y)
    pred = np.asarray(model.predict(X))
    std = np.full_like(pred, 0.1)
    template_path = tmp_path / "t.html"
    template_path.write_text(
        "{% for p in conformal_plots %}[{{ p.title }}]{% endfor %}", encoding="utf-8"
    )
    os.environ["MOR_TEMPLATE_PATH"] = str(template_path)
    try:
        html = generate_html_report(
            model_type="m", fidelity_levels=["lo", "hi"], output_names=["a", "b"],
            description="", metrics=_compute_metrics(Y, pred, std, ["a", "b"]),
            uncertainty_metrics={}, y_test=Y, best_pred=pred, best_std=std,
            best_model=model, X_train=X, n_train=30, n_test=30,
            cross_validation="", seed=0, notes="",
            conformal_intervals=(pred - 0.2, pred + 0.2), conformal_alpha=0.1,
        )
    finally:
        del os.environ["MOR_TEMPLATE_PATH"]
    assert "[Conformal Prediction Summary]" in html
