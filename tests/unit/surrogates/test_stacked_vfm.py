# Copyright (c) 2026 takotime808

import numpy as np
import pytest

from multioutreg.surrogates.stacked_vfm import AdditiveCorrectionVFM, StackedVFMSurrogate
from multioutreg.surrogates.gp_sklearn import GaussianProcessSurrogate
from multioutreg.surrogates.linear_sklearn import LinearRegressionSurrogate
from multioutreg.surrogates.rf_sklearn import RandomForestSurrogate


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_level_data():
    """Minimal 2-level data: 60 lo samples, 20 hi samples, 1D output."""
    rng = np.random.default_rng(0)
    X_lo = rng.standard_normal((60, 3))
    Y_lo = np.sin(X_lo).sum(axis=1, keepdims=True)
    X_hi = rng.standard_normal((20, 3))
    Y_hi = np.sin(X_hi).sum(axis=1, keepdims=True) * 1.1 + 0.05
    return {"lo": (X_lo, Y_lo), "hi": (X_hi, Y_hi)}


@pytest.fixture
def three_level_data():
    """3-level data: 100 lo, 40 mid, 15 hi samples, 1D output."""
    rng = np.random.default_rng(7)
    X_lo = rng.standard_normal((100, 4))
    Y_lo = X_lo[:, :2].sum(axis=1, keepdims=True)
    X_mid = rng.standard_normal((40, 4))
    Y_mid = X_mid[:, :2].sum(axis=1, keepdims=True) + 0.2
    X_hi = rng.standard_normal((15, 4))
    Y_hi = X_hi[:, :2].sum(axis=1, keepdims=True) + 0.5
    return {"lo": (X_lo, Y_lo), "mid": (X_mid, Y_mid), "hi": (X_hi, Y_hi)}


@pytest.fixture
def multi_output_data():
    """2-level data with 3 outputs."""
    rng = np.random.default_rng(13)
    X_lo = rng.standard_normal((60, 2))
    Y_lo = np.hstack([X_lo, X_lo[:, :1] ** 2])
    X_hi = rng.standard_normal((20, 2))
    Y_hi = np.hstack([X_hi, X_hi[:, :1] ** 2]) + 0.1
    return {"lo": (X_lo, Y_lo), "hi": (X_hi, Y_hi)}


# ===========================================================================
# StackedVFMSurrogate tests
# ===========================================================================


def test_stacked_fit_predict_shape_two_level(two_level_data):
    """Predicted shape matches Y_hi."""
    m = StackedVFMSurrogate(fidelity_levels=["lo", "hi"])
    m.fit(two_level_data)
    X_hi, Y_hi = two_level_data["hi"]
    preds = m.predict(X_hi)
    assert preds.shape == Y_hi.shape


def test_stacked_fit_predict_shape_three_level(three_level_data):
    """Three-level chain produces correct output shape."""
    m = StackedVFMSurrogate(fidelity_levels=["lo", "mid", "hi"])
    m.fit(three_level_data)
    X_hi, Y_hi = three_level_data["hi"]
    preds = m.predict(X_hi)
    assert preds.shape == Y_hi.shape


def test_stacked_predict_at_lo_level(two_level_data):
    """Can predict at lo fidelity using raw X."""
    m = StackedVFMSurrogate(fidelity_levels=["lo", "hi"])
    m.fit(two_level_data)
    X_lo, Y_lo = two_level_data["lo"]
    preds = m.predict(X_lo, level="lo")
    assert preds.shape == Y_lo.shape


def test_stacked_predict_at_mid_level(three_level_data):
    """Can predict at mid fidelity."""
    m = StackedVFMSurrogate(fidelity_levels=["lo", "mid", "hi"])
    m.fit(three_level_data)
    X_mid, Y_mid = three_level_data["mid"]
    preds = m.predict(X_mid, level="mid")
    assert preds.shape == Y_mid.shape


def test_stacked_return_std_shape(two_level_data):
    """return_std=True returns two arrays of matching shape with non-negative stds."""
    m = StackedVFMSurrogate(
        fidelity_levels=["lo", "hi"],
        surrogate_cls=GaussianProcessSurrogate,
    )
    m.fit(two_level_data)
    X_hi, Y_hi = two_level_data["hi"]
    preds, stds = m.predict(X_hi, return_std=True)
    assert preds.shape == Y_hi.shape
    assert stds.shape == Y_hi.shape
    assert np.all(stds >= 0)


def _top_level_std(m, X):
    """The top surrogate's own std, without lower-level propagation."""
    X_aug = m._augment(np.asarray(X, dtype=np.float64), len(m.fidelity_levels) - 1)
    return m.surrogates_[m.fidelity_levels[-1]].predict(X_aug, return_std=True)[1]


def test_stacked_mc_zero_samples_returns_top_level_std(three_level_data):
    """n_mc_samples=0 keeps the old std: the top surrogate's own."""
    m = StackedVFMSurrogate(["lo", "mid", "hi"], GaussianProcessSurrogate, n_mc_samples=0)
    m.fit(three_level_data)
    X_hi, _ = three_level_data["hi"]
    _, stds = m.predict(X_hi, return_std=True)
    np.testing.assert_array_equal(stds, _top_level_std(m, X_hi))


def test_stacked_mc_std_propagates_lower_levels(three_level_data):
    """MC std is wider than the top level's own std; the mean is unchanged."""
    m = StackedVFMSurrogate(["lo", "mid", "hi"], GaussianProcessSurrogate)
    m.fit(three_level_data)
    X = np.random.default_rng(1).standard_normal((40, 4))
    preds, stds = m.predict(X, return_std=True)
    np.testing.assert_array_equal(preds, m.predict(X))
    assert stds.shape == preds.shape
    assert stds.mean() > _top_level_std(m, X).mean()


def test_stacked_mc_std_deterministic(two_level_data):
    """Repeated calls give the same std (draws are seeded by random_state)."""
    m = StackedVFMSurrogate(["lo", "hi"], GaussianProcessSurrogate)
    m.fit(two_level_data)
    X_hi, _ = two_level_data["hi"]
    np.testing.assert_array_equal(
        m.predict(X_hi, return_std=True)[1], m.predict(X_hi, return_std=True)[1]
    )


def test_stacked_mc_std_chunked(two_level_data, monkeypatch):
    """Draws split into chunks still give a finite std of the right shape."""
    import multioutreg.surrogates.stacked_vfm as svfm

    m = StackedVFMSurrogate(["lo", "hi"], GaussianProcessSurrogate, n_mc_samples=7)
    m.fit(two_level_data)
    X_hi, Y_hi = two_level_data["hi"]
    monkeypatch.setattr(svfm, "_MC_MAX_ROWS", 3 * len(X_hi))
    _, stds = m.predict(X_hi, return_std=True)
    assert stds.shape == Y_hi.shape
    assert np.all(np.isfinite(stds)) and np.all(stds > 0)


def test_stacked_mc_missing_attribute_uses_top_level_std(two_level_data):
    """Models pickled before n_mc_samples existed keep the old std."""
    m = StackedVFMSurrogate(["lo", "hi"], GaussianProcessSurrogate)
    m.fit(two_level_data)
    del m.n_mc_samples
    X_hi, _ = two_level_data["hi"]
    np.testing.assert_array_equal(m.predict(X_hi, return_std=True)[1], _top_level_std(m, X_hi))


def test_stacked_mc_coverage_forrester():
    """4-level GP stack on Forrester (examples/mf_surrogates.ipynb, seed 0).

    The top level's own std covered about half the grid at 95%; with
    lower-level uncertainty propagated it reaches nominal coverage.
    """
    f3 = lambda x: (6 * x - 2) ** 2 * np.sin(12 * x - 4)  # noqa: E731
    fns = {
        "f3": f3,
        "f2": lambda x: (5.5 * x - 2.5) ** 2 * np.sin(12 * x - 4),
        "f1": lambda x: 0.75 * f3(x) + 5 * (x - 0.5) - 2,
        "f0": lambda x: 0.5 * f3(x) + 10 * (x - 0.5) - 5,
    }
    n = {"f3": 8, "f2": 25, "f1": 50, "f0": 80}
    sigma = {"f3": 0.02, "f2": 0.05, "f1": 0.08, "f0": 0.10}
    rng = np.random.default_rng(0)
    data = {}
    for lvl, k in n.items():
        X = rng.uniform(0, 1, size=(k, 1))
        data[lvl] = (X, fns[lvl](X) + rng.normal(scale=sigma[lvl], size=(k, 1)))

    levels = ["f0", "f1", "f2", "f3"]
    m = StackedVFMSurrogate(levels, GaussianProcessSurrogate)
    m.fit({k: data[k] for k in levels})
    X_test = np.linspace(0, 1, 300).reshape(-1, 1)
    y_hat, y_std = m.predict(X_test, return_std=True)
    coverage = np.mean(np.abs(f3(X_test) - y_hat) <= 1.96 * y_std)
    assert coverage >= 0.95


def test_stacked_augment_with_std(two_level_data):
    """augment_with_std=True fits without error and returns correct shape."""
    m = StackedVFMSurrogate(
        fidelity_levels=["lo", "hi"],
        augment_with_std=True,
    )
    m.fit(two_level_data)
    X_hi, Y_hi = two_level_data["hi"]
    preds = m.predict(X_hi)
    assert preds.shape == Y_hi.shape


def test_stacked_augmented_feature_dim_recorded(two_level_data):
    """augmented_n_features_per_level_ stores correct dims."""
    m = StackedVFMSurrogate(fidelity_levels=["lo", "hi"])
    m.fit(two_level_data)
    X_lo, Y_lo = two_level_data["lo"]
    assert m.augmented_n_features_per_level_["lo"] == X_lo.shape[1]
    expected_hi = X_lo.shape[1] + Y_lo.shape[1]
    assert m.augmented_n_features_per_level_["hi"] == expected_hi


def test_stacked_augmented_feature_dim_with_std(two_level_data):
    """With augment_with_std=True, hi level gets original + pred + std columns."""
    m = StackedVFMSurrogate(fidelity_levels=["lo", "hi"], augment_with_std=True)
    m.fit(two_level_data)
    X_lo, Y_lo = two_level_data["lo"]
    # hi level: original features + n_outputs (preds) + n_outputs (stds)
    expected_hi = X_lo.shape[1] + 2 * Y_lo.shape[1]
    assert m.augmented_n_features_per_level_["hi"] == expected_hi


def test_stacked_multi_output_shape(multi_output_data):
    """Multi-output target shapes propagate correctly through the chain."""
    m = StackedVFMSurrogate(fidelity_levels=["lo", "hi"])
    m.fit(multi_output_data)
    X_hi, Y_hi = multi_output_data["hi"]
    preds = m.predict(X_hi)
    assert preds.shape == Y_hi.shape


def test_stacked_heterogeneous_surrogate_cls(two_level_data):
    """Different surrogate classes per level work."""
    m = StackedVFMSurrogate(
        fidelity_levels=["lo", "hi"],
        surrogate_cls=[RandomForestSurrogate, LinearRegressionSurrogate],
    )
    m.fit(two_level_data)
    X_hi, Y_hi = two_level_data["hi"]
    preds = m.predict(X_hi)
    assert preds.shape == Y_hi.shape


def test_stacked_predict_before_fit_raises():
    m = StackedVFMSurrogate(fidelity_levels=["lo", "hi"])
    with pytest.raises(AttributeError, match="not fitted"):
        m.predict(np.random.rand(5, 3))


def test_stacked_invalid_level_raises(two_level_data):
    m = StackedVFMSurrogate(fidelity_levels=["lo", "hi"])
    m.fit(two_level_data)
    with pytest.raises(ValueError, match="Unknown fidelity level"):
        m.predict(np.random.rand(5, 3), level="medium")


def test_stacked_requires_at_least_two_levels():
    with pytest.raises(ValueError, match="at least 2"):
        StackedVFMSurrogate(fidelity_levels=["only_one"])


def test_stacked_missing_level_in_data_raises():
    """fit raises if a required level key is missing from data dict."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3))
    Y = rng.standard_normal((30, 1))
    m = StackedVFMSurrogate(fidelity_levels=["lo", "hi"])
    with pytest.raises(ValueError, match="Missing fidelity level"):
        m.fit({"lo": (X, Y)})  # missing "hi"


def test_stacked_1d_y_input(two_level_data):
    """1D Y arrays are reshaped to (n, 1) without error."""
    rng = np.random.default_rng(5)
    X_lo = rng.standard_normal((50, 2))
    Y_lo = rng.standard_normal(50)  # 1D
    X_hi = rng.standard_normal((15, 2))
    Y_hi = rng.standard_normal(15)  # 1D
    m = StackedVFMSurrogate(fidelity_levels=["lo", "hi"])
    m.fit({"lo": (X_lo, Y_lo), "hi": (X_hi, Y_hi)})
    preds = m.predict(X_hi)
    assert preds.shape == (15, 1)


def test_stacked_conformal_wrap_and_predict(two_level_data):
    """wrap_conformal + conformal_predict work end-to-end."""
    X_hi, Y_hi = two_level_data["hi"]
    X_train, X_cal = X_hi[:12], X_hi[12:]
    Y_train, Y_cal = Y_hi[:12], Y_hi[12:]
    X_lo, Y_lo = two_level_data["lo"]
    data_train = {"lo": (X_lo, Y_lo), "hi": (X_train, Y_train)}
    m = StackedVFMSurrogate(fidelity_levels=["lo", "hi"])
    m.fit(data_train)
    m.wrap_conformal(X_cal, Y_cal)
    lower, upper = m.conformal_predict(X_cal, alpha=0.1)
    assert lower.shape == Y_cal.shape
    assert upper.shape == Y_cal.shape
    assert np.all(upper >= lower)


def test_stacked_get_set_params():
    """get_params / set_params round-trip correctly."""
    m = StackedVFMSurrogate(fidelity_levels=["a", "b"], augment_with_std=True)
    params = m.get_params()
    assert params["augment_with_std"] is True
    assert params["fidelity_levels"] == ["a", "b"]
    assert params["n_mc_samples"] == 100
    m.set_params(augment_with_std=False)
    assert m.augment_with_std is False


def test_stacked_multi_output_attribute():
    assert StackedVFMSurrogate._multi_output is True


def test_stacked_n_features_in_recorded(two_level_data):
    """n_features_in_ is set to the raw feature count."""
    m = StackedVFMSurrogate(fidelity_levels=["lo", "hi"])
    m.fit(two_level_data)
    X_lo, _ = two_level_data["lo"]
    assert m.n_features_in_ == X_lo.shape[1]


def test_stacked_surrogates_dict_keyed_by_level(two_level_data):
    """surrogates_ dict has exactly the right keys after fitting."""
    m = StackedVFMSurrogate(fidelity_levels=["lo", "hi"])
    m.fit(two_level_data)
    assert set(m.surrogates_.keys()) == {"lo", "hi"}


def test_stacked_three_level_augmented_dims(three_level_data):
    """Three-level chain accumulates feature dims correctly."""
    m = StackedVFMSurrogate(fidelity_levels=["lo", "mid", "hi"])
    m.fit(three_level_data)
    n_feat = 4  # original features
    n_out_lo = 1
    n_out_mid = 1
    assert m.augmented_n_features_per_level_["lo"] == n_feat
    assert m.augmented_n_features_per_level_["mid"] == n_feat + n_out_lo
    assert m.augmented_n_features_per_level_["hi"] == n_feat + n_out_lo + n_out_mid


# ===========================================================================
# AdditiveCorrectionVFM tests
# ===========================================================================


def test_additive_fit_predict_shape(two_level_data):
    """Predicted shape matches Y_hi."""
    m = AdditiveCorrectionVFM()
    m.fit(two_level_data)
    X_hi, Y_hi = two_level_data["hi"]
    preds = m.predict(X_hi)
    assert preds.shape == Y_hi.shape


def test_additive_return_std_shape(two_level_data):
    """return_std=True returns correct shape with non-negative stds."""
    m = AdditiveCorrectionVFM(
        lo_surrogate_cls=GaussianProcessSurrogate,
        hi_surrogate_cls=GaussianProcessSurrogate,
    )
    m.fit(two_level_data)
    X_hi, Y_hi = two_level_data["hi"]
    preds, stds = m.predict(X_hi, return_std=True)
    assert preds.shape == Y_hi.shape
    assert stds.shape == Y_hi.shape
    assert np.all(stds >= 0)


def test_additive_quadrature_std_formula(two_level_data):
    """Combined std equals sqrt(rho^2 s_lo^2 + s_delta^2) exactly."""
    m = AdditiveCorrectionVFM(
        lo_surrogate_cls=GaussianProcessSurrogate,
        hi_surrogate_cls=GaussianProcessSurrogate,
    )
    m.fit(two_level_data)
    X_hi, _ = two_level_data["hi"]
    _, s_lo = m.surrogate_lo_.predict(X_hi, return_std=True)
    _, s_delta = m.surrogate_delta_.predict(X_hi, return_std=True)
    _, s_combined = m.predict(X_hi, return_std=True)
    s_lo = np.asarray(s_lo)
    s_delta = np.asarray(s_delta)
    if s_lo.ndim == 1:
        s_lo = s_lo.reshape(-1, 1)
    if s_delta.ndim == 1:
        s_delta = s_delta.reshape(-1, 1)
    expected = np.sqrt((m.rho_ * s_lo) ** 2 + s_delta ** 2)
    np.testing.assert_allclose(s_combined, expected, atol=1e-10)


def test_additive_predict_components_consistency(two_level_data):
    """predict_components: y_lo + delta == predict() output."""
    m = AdditiveCorrectionVFM()
    m.fit(two_level_data)
    X_hi, Y_hi = two_level_data["hi"]
    y_lo, delta = m.predict_components(X_hi)
    y_hi_pred = m.predict(X_hi)
    np.testing.assert_allclose(y_lo + delta, y_hi_pred, atol=1e-10)


def test_additive_predict_components_shapes(two_level_data):
    """predict_components returns two arrays of the correct shape."""
    m = AdditiveCorrectionVFM()
    m.fit(two_level_data)
    X_hi, Y_hi = two_level_data["hi"]
    y_lo, delta = m.predict_components(X_hi)
    assert y_lo.shape == Y_hi.shape
    assert delta.shape == Y_hi.shape


def test_additive_output_dim_mismatch_raises():
    """Mismatched lo/hi output dims raises ValueError at fit time."""
    rng = np.random.default_rng(99)
    X_lo = rng.standard_normal((30, 3))
    Y_lo = rng.standard_normal((30, 2))   # 2 outputs
    X_hi = rng.standard_normal((10, 3))
    Y_hi = rng.standard_normal((10, 3))   # 3 outputs — mismatch
    m = AdditiveCorrectionVFM()
    with pytest.raises(ValueError, match="same output dimension"):
        m.fit({"lo": (X_lo, Y_lo), "hi": (X_hi, Y_hi)})


def test_additive_missing_lo_key_raises():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 2))
    Y = rng.standard_normal((20, 1))
    m = AdditiveCorrectionVFM()
    with pytest.raises(ValueError, match="'lo'"):
        m.fit({"hi": (X, Y)})


def test_additive_missing_hi_key_raises():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 2))
    Y = rng.standard_normal((20, 1))
    m = AdditiveCorrectionVFM()
    with pytest.raises(ValueError, match="'hi'"):
        m.fit({"lo": (X, Y)})


def test_additive_multi_output_shape(multi_output_data):
    """Multi-output target shapes propagate correctly."""
    m = AdditiveCorrectionVFM()
    m.fit(multi_output_data)
    X_hi, Y_hi = multi_output_data["hi"]
    preds = m.predict(X_hi)
    assert preds.shape == Y_hi.shape


def test_additive_1d_y_input():
    """1D Y arrays are reshaped and handled correctly."""
    rng = np.random.default_rng(3)
    X_lo = rng.standard_normal((50, 2))
    Y_lo = rng.standard_normal(50)  # 1D
    X_hi = rng.standard_normal((15, 2))
    Y_hi = rng.standard_normal(15)  # 1D
    m = AdditiveCorrectionVFM()
    m.fit({"lo": (X_lo, Y_lo), "hi": (X_hi, Y_hi)})
    preds = m.predict(X_hi)
    assert preds.shape == (15, 1)


def test_additive_predict_before_fit_raises():
    m = AdditiveCorrectionVFM()
    with pytest.raises(AttributeError, match="not fitted"):
        m.predict(np.random.rand(5, 3))


def test_additive_predict_components_before_fit_raises():
    m = AdditiveCorrectionVFM()
    with pytest.raises(AttributeError, match="not fitted"):
        m.predict_components(np.random.rand(5, 3))


def test_additive_conformal_wrap_and_predict(two_level_data):
    """wrap_conformal + conformal_predict work end-to-end."""
    X_hi, Y_hi = two_level_data["hi"]
    X_train, X_cal = X_hi[:14], X_hi[14:]
    Y_train, Y_cal = Y_hi[:14], Y_hi[14:]
    X_lo, Y_lo = two_level_data["lo"]
    data_train = {"lo": (X_lo, Y_lo), "hi": (X_train, Y_train)}
    m = AdditiveCorrectionVFM()
    m.fit(data_train)
    m.wrap_conformal(X_cal, Y_cal)
    lower, upper = m.conformal_predict(X_cal, alpha=0.1)
    assert lower.shape == Y_cal.shape
    assert upper.shape == Y_cal.shape
    assert np.all(upper >= lower)


def test_additive_get_set_params():
    """get_params / set_params round-trip."""
    m = AdditiveCorrectionVFM(lo_surrogate_cls=LinearRegressionSurrogate)
    params = m.get_params()
    assert params["lo_surrogate_cls"] is LinearRegressionSurrogate
    assert params["hi_surrogate_cls"] is None
    m.set_params(lo_surrogate_cls=None)
    assert m.lo_surrogate_cls is None


def test_additive_multi_output_attribute():
    assert AdditiveCorrectionVFM._multi_output is True


def test_additive_n_outputs_recorded(two_level_data):
    """n_outputs_ is set correctly after fitting."""
    m = AdditiveCorrectionVFM()
    m.fit(two_level_data)
    _, Y_hi = two_level_data["hi"]
    assert m.n_outputs_ == Y_hi.shape[1]


def test_additive_n_features_in_recorded(two_level_data):
    """n_features_in_ is set to the raw feature count."""
    m = AdditiveCorrectionVFM()
    m.fit(two_level_data)
    X_lo, _ = two_level_data["lo"]
    assert m.n_features_in_ == X_lo.shape[1]


def test_additive_custom_lo_and_hi_surrogate_cls(two_level_data):
    """Custom lo and hi surrogate classes are instantiated correctly."""
    m = AdditiveCorrectionVFM(
        lo_surrogate_cls=LinearRegressionSurrogate,
        hi_surrogate_cls=RandomForestSurrogate,
    )
    m.fit(two_level_data)
    assert isinstance(m.surrogate_lo_, LinearRegressionSurrogate)
    assert isinstance(m.surrogate_delta_, RandomForestSurrogate)


# ---------------------------------------------------------------------------
# Import / registry test
# ---------------------------------------------------------------------------


def test_importable_from_surrogates_package():
    """Both classes are importable from the top-level surrogates package."""
    from multioutreg.surrogates import AdditiveCorrectionVFM as A
    from multioutreg.surrogates import StackedVFMSurrogate as S

    assert S is StackedVFMSurrogate
    assert A is AdditiveCorrectionVFM
