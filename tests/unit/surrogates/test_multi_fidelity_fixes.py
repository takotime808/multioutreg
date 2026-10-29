# Copyright (c) 2026 takotime808
"""Audit item 28: out-of-fold stacking features and fitted rho in the VFM models."""

import numpy as np
import pytest

from multioutreg.surrogates import MultiFidelitySurrogate
from multioutreg.surrogates.gp_sklearn import GaussianProcessSurrogate
from multioutreg.surrogates.linear_sklearn import LinearRegressionSurrogate
from multioutreg.surrogates.stacked_vfm import (
    AdditiveCorrectionVFM,
    StackedVFMSurrogate,
    _OOFPredictor,
)


def _f(X):
    return np.sin(3 * X[:, :1]) + X[:, 1:2] ** 2


class _NearestNeighbour:
    """1-NN regressor: in-sample predictions reproduce Y exactly."""

    fit_inputs: list = []

    def fit(self, X, Y):
        self.X_, self.Y_ = np.asarray(X), np.asarray(Y)
        _NearestNeighbour.fit_inputs.append(self.X_)
        return self

    def predict(self, X, return_std=False):
        d = ((np.asarray(X)[:, None, :] - self.X_[None, :, :]) ** 2).sum(-1)
        pred = self.Y_[d.argmin(axis=1)]
        return (pred, np.zeros_like(pred)) if return_std else pred


@pytest.fixture
def nested_data():
    rng = np.random.default_rng(0)
    X_lo = rng.uniform(-1, 1, (60, 2))
    Y_lo = _f(X_lo) + 0.3 * rng.standard_normal((60, 1))
    idx = rng.choice(60, 20, replace=False)
    X_hi = X_lo[idx]
    return {"lo": (X_lo, Y_lo), "hi": (X_hi, 2 * _f(X_hi))}, idx


# ---------------------------------------------------------------------------
# _OOFPredictor
# ---------------------------------------------------------------------------


def test_oof_predictor_replaces_only_overlapping_rows():
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(30, 2))
    Y = rng.standard_normal((30, 1))
    full = _NearestNeighbour().fit(X, Y)
    oof = _OOFPredictor(_NearestNeighbour, X, X, Y, cv=5, random_state=0)

    X_new = rng.uniform(size=(10, 2)) + 5.0
    Xq = np.vstack([X[:10], X_new])
    pred, std = oof.predict(full, Xq, Xq, return_std=False)

    assert std is None
    # Training rows: 1-NN in-sample would return Y exactly; OOF must not.
    assert not np.allclose(pred[:10], Y[:10])
    # Rows the model never saw keep the full model's prediction.
    np.testing.assert_array_equal(pred[10:], full.predict(X_new))


def test_oof_predictor_disabled_or_no_overlap_uses_full_model():
    rng = np.random.default_rng(2)
    X = rng.uniform(size=(20, 2))
    Y = rng.standard_normal((20, 1))
    full = _NearestNeighbour().fit(X, Y)
    for cv in (None, 0, 1):
        pred, _ = _OOFPredictor(_NearestNeighbour, X, X, Y, cv, 0).predict(full, X, X, False)
        np.testing.assert_array_equal(pred, Y)


def test_oof_predictor_duplicate_rows_share_a_fold():
    X = np.repeat(np.arange(10, dtype=float).reshape(-1, 1), 2, axis=0)
    Y = X.copy()
    oof = _OOFPredictor(_NearestNeighbour, X, X, Y, cv=5, random_state=0)
    folds = oof._train_folds
    assert all(folds[2 * i] == folds[2 * i + 1] for i in range(10))
    # So the held-out copy cannot be matched by its twin.
    pred, _ = oof.predict(_NearestNeighbour().fit(X, Y), X, X, False)
    assert not np.allclose(pred, Y)


# ---------------------------------------------------------------------------
# StackedVFMSurrogate
# ---------------------------------------------------------------------------


def test_stacked_hi_level_trained_on_oof_lo_predictions(nested_data):
    data, _ = nested_data
    X_hi = data["hi"][0]

    _NearestNeighbour.fit_inputs = []
    StackedVFMSurrogate(["lo", "hi"], surrogate_cls=_NearestNeighbour, oof_cv=None).fit(data)
    in_sample_feature = _NearestNeighbour.fit_inputs[-1][:, -1]

    _NearestNeighbour.fit_inputs = []
    model = StackedVFMSurrogate(["lo", "hi"], surrogate_cls=_NearestNeighbour).fit(data)
    oof_feature = _NearestNeighbour.fit_inputs[-1][:, -1]

    # In-sample 1-NN feature is the noisy Y_lo itself; the OOF one is not.
    lo_at_hi = model.surrogates_["lo"].predict(X_hi).ravel()
    np.testing.assert_array_equal(in_sample_feature, lo_at_hi)
    assert not np.allclose(oof_feature, lo_at_hi)
    # Fold models are not kept after fit.
    assert not hasattr(model, "_oof_predictors")


def test_stacked_oof_no_effect_when_levels_disjoint():
    rng = np.random.default_rng(3)
    X_lo, X_hi = rng.uniform(size=(50, 2)), rng.uniform(size=(15, 2)) + 0.01
    data = {"lo": (X_lo, _f(X_lo)), "hi": (X_hi, 2 * _f(X_hi))}
    Xt = rng.uniform(size=(20, 2))
    a = StackedVFMSurrogate(["lo", "hi"], surrogate_cls=LinearRegressionSurrogate, oof_cv=None)
    b = StackedVFMSurrogate(["lo", "hi"], surrogate_cls=LinearRegressionSurrogate, oof_cv=5)
    np.testing.assert_allclose(a.fit(data).predict(Xt), b.fit(data).predict(Xt))


def test_stacked_three_levels_with_oof(nested_data):
    data, idx = nested_data
    X_lo, Y_lo = data["lo"]
    X_mid = X_lo[idx[:15]]
    data3 = {"lo": data["lo"], "mid": (X_mid, 1.5 * _f(X_mid)), "hi": data["hi"]}
    model = StackedVFMSurrogate(["lo", "mid", "hi"], surrogate_cls=LinearRegressionSurrogate)
    assert model.fit(data3).predict(X_lo[:7]).shape == (7, 1)


def test_stacked_get_params_has_new_keys():
    params = StackedVFMSurrogate(["a", "b"], oof_cv=3, random_state=1).get_params()
    assert params["oof_cv"] == 3 and params["random_state"] == 1


# ---------------------------------------------------------------------------
# AdditiveCorrectionVFM
# ---------------------------------------------------------------------------


def _ar1_data(seed=0, lo_noise=0.0, n_lo=80, n_hi=20):
    rng = np.random.default_rng(seed)
    X_lo = rng.uniform(-1, 1, (n_lo, 2))
    Y_lo = _f(X_lo) + lo_noise * rng.standard_normal((n_lo, 1))
    X_hi = X_lo[rng.choice(n_lo, n_hi, replace=False)]
    Xt = rng.uniform(-1, 1, (300, 2))
    return {"lo": (X_lo, Y_lo), "hi": (X_hi, 2 * _f(X_hi) + 0.3)}, Xt, 2 * _f(Xt) + 0.3


def test_additive_recovers_rho_and_beats_rho_one():
    data, Xt, Yt = _ar1_data()
    kw = dict(lo_surrogate_cls=GaussianProcessSurrogate, hi_surrogate_cls=GaussianProcessSurrogate)
    fitted = AdditiveCorrectionVFM(**kw).fit(data)
    fixed = AdditiveCorrectionVFM(rho=1.0, **kw).fit(data)

    assert fitted.rho_ == pytest.approx([2.0], abs=0.1)
    np.testing.assert_array_equal(fixed.rho_, [1.0])
    mse = lambda m: np.mean((m.predict(Xt) - Yt) ** 2)
    assert mse(fitted) < mse(fixed)


def test_additive_rho_per_output_and_components_sum():
    rng = np.random.default_rng(4)
    X_lo, X_hi = rng.uniform(-1, 1, (60, 2)), rng.uniform(-1, 1, (25, 2))
    lo = lambda X: np.hstack([X[:, :1], X[:, 1:]])
    data = {"lo": (X_lo, lo(X_lo)), "hi": (X_hi, lo(X_hi) * [3.0, -0.5])}
    m = AdditiveCorrectionVFM(LinearRegressionSurrogate, LinearRegressionSurrogate).fit(data)
    np.testing.assert_allclose(m.rho_, [3.0, -0.5], atol=1e-8)
    scaled_lo, delta = m.predict_components(X_hi)
    np.testing.assert_allclose(scaled_lo + delta, m.predict(X_hi), atol=1e-10)


def test_additive_constant_lo_prediction_falls_back_to_rho_one():
    rng = np.random.default_rng(5)
    X_lo, X_hi = rng.uniform(size=(30, 2)), rng.uniform(size=(10, 2))
    data = {"lo": (X_lo, np.ones((30, 1))), "hi": (X_hi, X_hi[:, :1])}
    m = AdditiveCorrectionVFM(LinearRegressionSurrogate, LinearRegressionSurrogate).fit(data)
    np.testing.assert_array_equal(m.rho_, [1.0])


def _forrester_data(n_hi=8, seed=0):
    from multioutreg.active.benchmarks import forrester, forrester_lo

    rng = np.random.default_rng(seed)
    X_lo, X_hi = rng.uniform(0, 1, (80, 1)), rng.uniform(0, 1, (n_hi, 1))
    data = {"lo": (X_lo, forrester_lo(X_lo)), "hi": (X_hi, forrester(X_hi))}
    Xt = np.linspace(0, 1, 200).reshape(-1, 1)
    return data, Xt, forrester(Xt)


def test_additive_joint_rho_recovers_forrester_where_slope_is_biased():
    """f_hi - 2 f_lo is linear in x, so the plain slope is biased."""
    data, Xt, Yt = _forrester_data()
    kw = dict(lo_surrogate_cls=GaussianProcessSurrogate, hi_surrogate_cls=GaussianProcessSurrogate)
    joint = AdditiveCorrectionVFM(**kw).fit(data)
    slope = AdditiveCorrectionVFM(rho="slope", **kw).fit(data)

    assert joint.rho_ == pytest.approx([2.0], abs=0.05)
    assert abs(slope.rho_[0] - 2.0) > 0.3
    mse = lambda m: np.mean((m.predict(Xt) - Yt) ** 2)
    assert mse(joint) < 0.1 * mse(slope)


def test_additive_joint_rho_falls_back_to_slope_with_few_hi_points():
    data, _, _ = _forrester_data(n_hi=3)  # n_hi <= n_features + 2
    kw = dict(lo_surrogate_cls=LinearRegressionSurrogate, hi_surrogate_cls=LinearRegressionSurrogate)
    np.testing.assert_array_equal(
        AdditiveCorrectionVFM(**kw).fit(data).rho_,
        AdditiveCorrectionVFM(rho="slope", **kw).fit(data).rho_,
    )


def test_additive_rho_rejects_unknown_string():
    data, _, _ = _forrester_data()
    with pytest.raises(ValueError, match="rho must be"):
        AdditiveCorrectionVFM(LinearRegressionSurrogate, LinearRegressionSurrogate, rho="ml").fit(data)


def test_additive_noisy_nested_lo_oof_avoids_in_sample_blowup():
    """With noisy, nested lo data, in-sample f_lo gave a badly wrong rho."""
    data, Xt, Yt = _ar1_data(seed=0, lo_noise=0.5, n_lo=120, n_hi=25)
    kw = dict(lo_surrogate_cls=GaussianProcessSurrogate, hi_surrogate_cls=GaussianProcessSurrogate)
    mse = lambda m: np.mean((m.fit(data).predict(Xt) - Yt) ** 2)
    assert mse(AdditiveCorrectionVFM(**kw)) < 0.5 * mse(AdditiveCorrectionVFM(oof_cv=None, **kw))


def test_additive_get_params_has_new_keys():
    params = AdditiveCorrectionVFM(rho=1.0, oof_cv=None).get_params()
    assert params["rho"] == 1.0 and params["oof_cv"] is None and params["random_state"] == 0


# ---------------------------------------------------------------------------
# MultiFidelitySurrogate stays the uncoupled baseline
# ---------------------------------------------------------------------------


def test_multi_fidelity_surrogate_hi_ignores_lo_data():
    rng = np.random.default_rng(6)
    X_lo, X_hi = rng.uniform(size=(40, 2)), rng.uniform(size=(15, 2))
    Y_hi = X_hi[:, :1]
    a = MultiFidelitySurrogate(LinearRegressionSurrogate, ["lo", "hi"])
    b = MultiFidelitySurrogate(LinearRegressionSurrogate, ["lo", "hi"])
    a.fit({"lo": (X_lo, X_lo[:, :1]), "hi": (X_hi, Y_hi)})
    b.fit({"lo": (X_lo, -5 * X_lo[:, 1:]), "hi": (X_hi, Y_hi)})
    np.testing.assert_allclose(a.predict(X_lo, level="hi"), b.predict(X_lo, level="hi"))
    assert "no information transfer" in MultiFidelitySurrogate.__doc__
