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
