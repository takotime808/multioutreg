# Copyright (c) 2026 takotime808

"""PLS, joint-tree random forest, reduced-rank regression and regressor chain."""

import numpy as np
import pytest
from sklearn.base import clone

from multioutreg.model_selection.auto_detect import (
    AutoDetectMultiOutputRegressor,
    joint_candidates,
)
from multioutreg.surrogates import (
    BayesianRidgeSurrogate,
    JointRandomForestSurrogate,
    LinearRegressionSurrogate,
    PLSSurrogate,
    ReducedRankRegressionSurrogate,
    RegressorChainSurrogate,
)

JOINT = [PLSSurrogate, JointRandomForestSurrogate, ReducedRankRegressionSurrogate,
         RegressorChainSurrogate]


def _low_rank_data(n=120, p=6, m=5, rank=1, noise=0.1, seed=0):
    """Linear outputs driven by ``rank`` combinations of the inputs."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    B = rng.normal(size=(p, rank)) @ rng.normal(size=(rank, m))
    return X, X @ B + noise * rng.normal(size=(n, m))


@pytest.mark.parametrize("cls", JOINT)
def test_flagged_as_joint_and_accepts_1d_target(cls):
    assert cls._multi_output is True
    X, Y = _low_rank_data(n=40, m=1)
    model = cls().fit(X, Y[:, 0])
    mean, std = model.predict(X[:5], return_std=True)
    assert mean.shape == std.shape == (5, 1)


@pytest.mark.parametrize("cls", JOINT)
def test_clone_is_unfitted_and_keeps_params(cls):
    model = cls(random_state=3) if "random_state" in cls().get_params() else cls()
    copy = clone(model)
    assert copy.get_params() == model.get_params()
    assert not any(k.endswith("_") and not k.startswith("_") for k in vars(copy))


def test_pls_recovers_linear_outputs_with_tight_std():
    X, Y = _low_rank_data(noise=0.01)
    model = PLSSurrogate(n_components=6).fit(X[:100], Y[:100])
    mean, std = model.predict(X[100:], return_std=True)
    assert np.sqrt(np.mean((mean - Y[100:]) ** 2)) < 0.05
    np.testing.assert_allclose(std.mean(), 0.01, rtol=0.5)


def test_pls_clamps_components_to_input_dimension():
    X, Y = _low_rank_data(n=30, p=2)
    assert PLSSurrogate(n_components=10).fit(X, Y).n_components_ == 2


def test_pls_std_grows_away_from_the_design():
    X, Y = _low_rank_data()
    model = PLSSurrogate().fit(X, Y)
    _, near = model.predict(np.zeros((1, X.shape[1])), return_std=True)
    _, far = model.predict(np.full((1, X.shape[1]), 10.0), return_std=True)
    assert np.all(far > near)


def test_joint_forest_uses_one_forest_for_all_outputs():
    X, Y = _low_rank_data(n=60, m=3)
    model = JointRandomForestSurrogate(n_estimators=10, random_state=0).fit(X, Y)
    assert model.forest_.n_outputs_ == 3
    assert model.noise_var_.shape == (3,)


def test_joint_forest_without_noise_is_tree_spread_only():
    X, Y = _low_rank_data(n=60, m=3)
    kw = dict(n_estimators=10, random_state=0)
    with_noise = JointRandomForestSurrogate(**kw).fit(X, Y)
    spread = JointRandomForestSurrogate(include_noise=False, **kw).fit(X, Y)
    _, s_noise = with_noise.predict(X[:5], return_std=True)
    _, s_spread = spread.predict(X[:5], return_std=True)
    np.testing.assert_array_equal(spread.noise_var_, 0.0)
    assert np.all(s_noise >= s_spread - 1e-12)


def test_reduced_rank_picks_the_true_rank():
    X, Y = _low_rank_data(rank=2, noise=0.3)
    model = ReducedRankRegressionSurrogate().fit(X, Y)
    assert model.rank_ == 2
    assert len(model.cv_mse_) == min(X.shape[1], Y.shape[1])


def test_reduced_rank_beats_independent_least_squares_on_low_rank_data():
    X, Y = _low_rank_data(n=440, p=10, m=10, rank=1, noise=1.0, seed=2)
    rrr = ReducedRankRegressionSurrogate(rank=1).fit(X[:40], Y[:40])
    ols = LinearRegressionSurrogate().fit(X[:40], Y[:40])
    assert (np.mean((rrr.predict(X[40:]) - Y[40:]) ** 2)
            < np.mean((ols.predict(X[40:]) - Y[40:]) ** 2))


def test_full_rank_matches_least_squares():
    X, Y = _low_rank_data(rank=5)
    rrr = ReducedRankRegressionSurrogate(rank=5, alpha=0.0).fit(X, Y)
    ols = LinearRegressionSurrogate().fit(X, Y)
    np.testing.assert_allclose(rrr.predict(X), ols.predict(X), atol=1e-8)


def test_reduced_rank_rejects_bad_rank():
    X, Y = _low_rank_data(n=20)
    with pytest.raises(ValueError):
        ReducedRankRegressionSurrogate(rank=0).fit(X, Y)


def _chain_data(n=150, seed=0):
    """y1 is a noisy function of x; y2 depends on y1 itself, noise included."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 1))
    y1 = 2 * X[:, 0] + 0.3 * rng.normal(size=n)
    y2 = 3 * y1 + 0.1 * rng.normal(size=n)
    return X, np.column_stack([y1, y2])


def test_chain_propagates_upstream_noise():
    X, Y = _chain_data()
    model = RegressorChainSurrogate(BayesianRidgeSurrogate, n_samples=400).fit(X, Y)
    _, std = model.predict(X[:20], return_std=True)
    # y2 = 3*y1 + noise: its std must carry 3x y1's noise, not just 0.1.
    np.testing.assert_allclose(std[:, 0], 0.3, rtol=0.25)
    np.testing.assert_allclose(std[:, 1], np.hypot(3 * 0.3, 0.1), rtol=0.25)


def test_chain_cov_captures_output_correlation():
    X, Y = _chain_data()
    model = RegressorChainSurrogate(BayesianRidgeSurrogate, n_samples=400).fit(X, Y)
    mean, cov = model.predict(X[:10], return_cov=True)
    assert cov.shape == (10, 2, 2)
    corr = cov[:, 0, 1] / np.sqrt(cov[:, 0, 0] * cov[:, 1, 1])
    assert np.all(corr > 0.9)
    np.testing.assert_allclose(mean, model.predict(X[:10]))


def test_chain_order_and_determinism():
    X, Y = _chain_data(n=60)
    model = RegressorChainSurrogate(BayesianRidgeSurrogate, order=[1, 0]).fit(X, Y)
    np.testing.assert_array_equal(model.order_, [1, 0])
    # Link 0 predicts output 1 from X alone; link 1 sees X and output 1.
    assert model.links_[1].model.estimators_[0].n_features_in_ == 2
    _, s1 = model.predict(X[:5], return_std=True)
    _, s2 = model.predict(X[:5], return_std=True)
    np.testing.assert_array_equal(s1, s2)


@pytest.mark.parametrize("order", [[0, 0], [0, 2], [0]])
def test_chain_rejects_bad_order(order):
    X, Y = _chain_data(n=20)
    with pytest.raises(ValueError):
        RegressorChainSurrogate(BayesianRidgeSurrogate, order=order).fit(X, Y)


def test_joint_candidates_compete_in_auto_detect():
    X, Y = _low_rank_data(n=60, p=3, m=2)
    model = AutoDetectMultiOutputRegressor(
        [LinearRegressionSurrogate().model.estimator], [{}], cv=3)
    candidates = joint_candidates()
    assert {type(c) for c in candidates} == set(JOINT)
    model.register_multi_output_candidates(candidates).fit(X, Y)
    names = set(model.selection_scores_[0])
    assert {f"{c.__name__}[{k}]" for k, c in enumerate(JOINT)} <= names
    mean, std = model.predict(X[:5], return_std=True)
    assert mean.shape == std.shape == (5, 2)
