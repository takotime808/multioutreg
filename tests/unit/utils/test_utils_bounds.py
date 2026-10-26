# Copyright (c) 2026 takotime808
"""Bounds, invariants and failure points of the utils helpers."""

import base64

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from multioutreg.utils.figure_utils import plot_to_b64
from multioutreg.utils.imputation import (
    apply_imputation,
    detect_missing,
    drop_unusable_rows,
    impute_train_test,
)
from multioutreg.utils.metric_utils import flatten_metrics_df, gaussian_uncertainty_metrics
from multioutreg.utils.pca_utils import fit_input_pca
from multioutreg.utils.surrogate_utils import predict_with_std

PNG_MAGIC = b"\x89PNG"

# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------


def test_detect_missing_reports_only_columns_with_gaps():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan], "b": [1, 2, 3, 4], "c": [None] * 4})
    out = detect_missing(df)
    assert list(out.index) == ["a", "c"]
    assert out.loc["a", "missing_count"] == 2
    assert out.loc["a", "missing_pct"] == 50.0
    assert out.loc["c", "missing_pct"] == 100.0


def test_detect_missing_percentages_are_bounded():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(30, 5)))
    df = df.mask(rng.uniform(size=df.shape) < 0.3)
    pct = detect_missing(df)["missing_pct"]
    assert ((pct > 0) & (pct <= 100)).all()


def test_detect_missing_complete_and_empty_frames_return_empty():
    assert detect_missing(pd.DataFrame({"a": [1, 2]})).empty
    assert detect_missing(pd.DataFrame({"a": []})).empty


def test_apply_imputation_fills_selected_and_drops_others():
    df = pd.DataFrame({
        "x": [1.0, np.nan, 3.0, 4.0, 5.0],
        "y": [1.0, 2.0, np.nan, 4.0, 5.0],
        "z": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    out = apply_imputation(df, cols_to_impute=["x", "z"], cols_to_drop_rows=["y"], n_neighbors=2)
    assert len(out) == 4
    assert not out[["x", "y"]].isna().any().any()
    # Original is untouched.
    assert df["x"].isna().sum() == 1


def test_apply_imputation_imputed_values_lie_within_observed_range():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.uniform(10, 20, size=(40, 3)), columns=list("abc"))
    df.iloc[::5, 0] = np.nan
    out = apply_imputation(df, ["a", "b", "c"], [], n_neighbors=3)
    assert out["a"].between(df["a"].min(), df["a"].max()).all()


def test_apply_imputation_no_columns_is_identity_copy():
    df = pd.DataFrame({"a": [1.0, np.nan]})
    out = apply_imputation(df, [], [])
    pd.testing.assert_frame_equal(out, df)
    assert out is not df


def test_apply_imputation_unknown_column_raises():
    with pytest.raises(KeyError):
        apply_imputation(pd.DataFrame({"a": [1.0]}), ["missing"], [])


def test_apply_imputation_zero_neighbors_raises():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    with pytest.raises(ValueError):
        apply_imputation(df, ["a"], [], n_neighbors=0)


def test_drop_unusable_rows_with_no_columns_returns_copy():
    df = pd.DataFrame({"a": [np.nan, 1.0]})
    out = drop_unusable_rows(df, [], [])
    pd.testing.assert_frame_equal(out, df)
    assert out is not df


def test_drop_unusable_rows_overlapping_column_lists():
    df = pd.DataFrame({"y": [1.0, np.nan, 3.0], "x": [np.nan, 1.0, 2.0]})
    out = drop_unusable_rows(df, ["y"], ["y", "x"])
    assert out.index.tolist() == [2]


def test_impute_train_test_all_nan_column_is_kept_not_dropped():
    X_tr = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    X_te = np.array([[1.5, np.nan]])
    a, b = impute_train_test(X_tr, X_te, n_neighbors=2)
    assert a.shape == X_tr.shape and b.shape == X_te.shape
    assert not np.isnan(a).any() and not np.isnan(b).any()


def test_impute_train_test_more_neighbors_than_rows_still_works():
    X_tr = np.array([[1.0, 2.0], [np.nan, 4.0]])
    a, _ = impute_train_test(X_tr, X_tr.copy(), n_neighbors=10)
    assert a[1, 0] == pytest.approx(1.0)


def test_impute_train_test_feature_count_mismatch_raises():
    with pytest.raises(ValueError):
        impute_train_test(np.array([[np.nan, 1.0]]), np.array([[1.0, 2.0, 3.0]]))


# ---------------------------------------------------------------------------
# Gaussian uncertainty metrics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nominal", [0.5, 0.9, 0.95, 0.99])
def test_gaussian_metrics_coverage_and_miscoverage_are_bounded(nominal):
    rng = np.random.default_rng(2)
    y = rng.normal(size=(200, 2))
    m = gaussian_uncertainty_metrics(y, np.zeros_like(y), np.ones_like(y), nominal=nominal)
    cov = next(v for k, v in m.items() if k.startswith("Empirical coverage"))
    mis = next(v for k, v in m.items() if k.startswith("Miscoverage"))
    assert 0.0 <= cov <= 1.0
    assert 0.0 <= mis <= max(nominal, 1 - nominal)
    # Well-specified model: coverage is near nominal.
    assert cov == pytest.approx(nominal, abs=0.08)


def test_gaussian_metrics_infinite_std_covers_everything():
    y = np.array([[100.0]])
    m = gaussian_uncertainty_metrics(y, np.zeros_like(y), np.full_like(y, np.inf))
    assert m["Empirical coverage of 95% Gaussian interval"] == 1.0


def test_gaussian_metrics_nll_lower_for_better_calibrated_std():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(500, 1))
    good = gaussian_uncertainty_metrics(y, np.zeros_like(y), np.ones_like(y))
    bad = gaussian_uncertainty_metrics(y, np.zeros_like(y), np.full_like(y, 10.0))
    assert good["Gaussian NLL (mean per point)"] < bad["Gaussian NLL (mean per point)"]


def test_gaussian_metrics_negative_std_omits_nll():
    m = gaussian_uncertainty_metrics(np.zeros((2, 1)), np.zeros((2, 1)), np.array([[1.0], [-1.0]]))
    assert "Gaussian NLL (mean per point)" not in m


def test_gaussian_metrics_shape_mismatch_raises():
    with pytest.raises(ValueError):
        gaussian_uncertainty_metrics(np.zeros((3, 2)), np.zeros((4, 2)), np.ones((3, 2)))


def test_flatten_metrics_df_prefixes_duplicate_keys():
    df = pd.DataFrame({
        "acc": [{"mae": 1.0, "rmse": 2.0}],
        "other": [{"mae": 3.0}],
        "output": [0],
    })
    flat = flatten_metrics_df(df)
    assert set(flat.columns) == {"mae", "rmse", "other_mae", "output"}
    assert flat.loc[0, "other_mae"] == 3.0


def test_flatten_metrics_df_missing_key_in_later_row_is_none():
    df = pd.DataFrame({"acc": [{"mae": 1.0}, {}], "output": [0, 1]})
    flat = flatten_metrics_df(df)
    assert flat["mae"].isna().tolist() == [False, True]


def test_flatten_metrics_df_without_output_column_raises():
    with pytest.raises(KeyError):
        flatten_metrics_df(pd.DataFrame({"acc": [{"mae": 1.0}]}))


# ---------------------------------------------------------------------------
# Input PCA bounds
# ---------------------------------------------------------------------------


@pytest.fixture
def X():
    return np.random.default_rng(4).normal(size=(30, 4))


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (dict(method="Explained variance threshold", threshold=5.0), 4),
        (dict(method="Explained variance threshold", threshold=1.0), 4),
        (dict(method="Explained variance threshold", threshold=0.0), 1),
        (dict(method="Explained variance threshold", threshold=-1.0), 1),
        (dict(method="Manual", n_components=0), 1),
        (dict(method="Manual", n_components=-3), 1),
        (dict(method="Manual", n_components=99), 4),
    ],
)
def test_pca_component_count_is_clamped_to_valid_range(X, kwargs, expected):
    assert fit_input_pca(X, X[:3], **kwargs).n_components == expected


def test_pca_threshold_without_value_falls_back_to_kaiser(X):
    res = fit_input_pca(X, X[:3], method="Explained variance threshold")
    assert res.n_components == max(1, res.kaiser_k)


def test_pca_kaiser_on_uncorrelated_data_never_exceeds_feature_count(X):
    res = fit_input_pca(X, X[:3])
    assert 0 <= res.kaiser_k <= X.shape[1]
    assert 1 <= res.n_components <= X.shape[1]


def test_pca_more_features_than_samples_caps_at_sample_count():
    X = np.random.default_rng(5).normal(size=(5, 10))
    res = fit_input_pca(X, X[:2], method="Manual", n_components=10)
    assert res.n_components == 5
    assert res.X_train.shape == (5, 5)


def test_pca_constant_column_does_not_produce_nan():
    X = np.random.default_rng(6).normal(size=(20, 3))
    X[:, 1] = 7.0
    res = fit_input_pca(X, X[:4], method="Manual", n_components=2)
    assert np.isfinite(res.X_train).all() and np.isfinite(res.X_test).all()


def test_pca_feature_count_mismatch_raises(X):
    with pytest.raises(ValueError):
        fit_input_pca(X, np.zeros((2, 3)))


def test_pca_nan_input_raises(X):
    X = X.copy()
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_input_pca(X, X[:3])


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def test_plot_to_b64_returns_png_and_closes_figure():
    before = len(plt.get_fignums())
    s = plot_to_b64(lambda: plt.plot([0, 1], [0, 1]))
    assert base64.b64decode(s).startswith(PNG_MAGIC)
    assert len(plt.get_fignums()) == before


def test_plot_to_b64_propagates_plot_errors():
    def boom():
        raise RuntimeError("bad plot")

    with pytest.raises(RuntimeError, match="bad plot"):
        plot_to_b64(boom)


class _Est:
    def __init__(self, k):
        self.k = k

    def predict(self, X, return_std=False):
        n = len(X)
        return np.full(n, self.k), np.full(n, self.k / 10)


class _Multi:
    estimators_ = [_Est(1.0), _Est(2.0), _Est(3.0)]


def test_predict_with_std_stacks_one_column_per_estimator():
    mean, std = predict_with_std(_Multi(), np.zeros((4, 2)))
    assert mean.shape == std.shape == (4, 3)
    np.testing.assert_allclose(mean[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(std[0], [0.1, 0.2, 0.3])


def test_predict_with_std_requires_fitted_multioutput():
    with pytest.raises(AttributeError):
        predict_with_std(object(), np.zeros((2, 2)))
