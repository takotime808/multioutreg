# Copyright (c) 2026 takotime808

import warnings

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from multioutreg.active import ActiveLearner, CommitteeSurrogate, get_benchmark
from multioutreg.surrogates import GaussianProcessSurrogate, KNeighborsSurrogate


def gp():
    kernel = ConstantKernel(1.0) * Matern(1.0, length_scale_bounds=(0.1, 100.0), nu=2.5) + WhiteKernel(1e-6, (1e-10, 1e-2))
    return GaussianProcessSurrogate(kernel=kernel, n_restarts_optimizer=2)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_ask_tell_shapes_and_bounds():
    b = get_benchmark("branin_currin")
    al = ActiveLearner(gp, b.bounds, batch_size=3, n_candidates=256)
    X0 = al.initial_design(6)
    al.tell(X0, b(X0))
    X_new = al.ask()
    assert X_new.shape == (3, 2)
    assert np.all(X_new >= b.bounds[:, 0]) and np.all(X_new <= b.bounds[:, 1])
    al.tell(X_new, b(X_new))
    assert al.X_.shape == (9, 2) and al.Y_.shape == (9, 2)


def test_tell_validates_rows_and_fit_needs_data():
    al = ActiveLearner(gp, [[0, 1]])
    with pytest.raises(RuntimeError):
        al.fit()
    with pytest.raises(ValueError):
        al.tell(np.zeros((3, 1)), np.zeros((2, 1)))


def test_invalid_arguments():
    with pytest.raises(ValueError):
        ActiveLearner(gp, [0, 1])
    with pytest.raises(ValueError):
        ActiveLearner(gp, [[0, 1]], acquisition="thompson")


def test_run_is_reproducible_and_respects_budget():
    b = get_benchmark("forrester")
    runs = []
    for _ in range(2):
        al = ActiveLearner(gp, b.bounds, batch_size=2, n_candidates=128, random_state=3)
        al.run(b, n_init=4, n_iter=10, max_evals=11)
        runs.append(al.X_)
    np.testing.assert_array_equal(*runs)
    assert len(runs[0]) == 11


def test_active_beats_random_on_forrester():
    b = get_benchmark("forrester")
    Xt = b.sample(256, seed=99)
    Yt = b(Xt)
    err = {}
    for acq in ("max_variance", "random"):
        scores = []
        for seed in range(3):
            al = ActiveLearner(gp, b.bounds, acquisition=acq, n_candidates=256, random_state=seed)
            scores.append(al.run(b, n_init=4, n_iter=8, X_test=Xt, Y_test=Yt)[-1]["nrmse"])
        err[acq] = np.median(scores)
    assert err["max_variance"] < err["random"]
    assert err["max_variance"] < 0.1


def test_expected_improvement_finds_branin_minimum_region():
    b = get_benchmark("branin")
    al = ActiveLearner(gp, b.bounds, acquisition="ei", n_candidates=512, random_state=0)
    hist = al.run(b, n_init=6, n_iter=15)
    assert hist[-1]["best"] < 1.0
    assert hist[-1]["best"] <= hist[0]["best"]


def test_rtol_stops_early():
    b = get_benchmark("forrester")
    Xt = b.sample(128, seed=5)
    al = ActiveLearner(gp, b.bounds, n_candidates=128)
    hist = al.run(b, n_init=6, n_iter=40, X_test=Xt, Y_test=b(Xt), rtol=0.5, patience=1)
    assert len(hist) < 41


def test_committee_learner_runs_without_native_std():
    b = get_benchmark("ishigami")
    with pytest.raises(ValueError, match="CommitteeSurrogate"):
        ActiveLearner(KNeighborsSurrogate, b.bounds).run(b, n_init=8, n_iter=1)
    al = ActiveLearner(lambda: CommitteeSurrogate(KNeighborsSurrogate, 4), b.bounds, penalize_train=True, n_candidates=128)
    al.run(b, n_init=8, n_iter=2)
    assert len(al.X_) == 10
