# Copyright (c) 2026 takotime808

import numpy as np
import pytest

from multioutreg.active.batch import default_length_scale, select_batch
from multioutreg.active.benchmarks import BENCHMARKS, get_benchmark


def _grid(n=21):
    g = np.linspace(0, 1, n)
    return np.array([[a, b] for a in g for b in g])


def test_batch_spreads_on_flat_scores():
    U = _grid()
    idx = select_batch(U, np.ones(len(U)), 6)
    assert len(set(idx.tolist())) == 6
    picks = U[idx]
    d = np.sqrt(((picks[:, None] - picks[None]) ** 2).sum(-1))
    assert d[np.triu_indices(6, 1)].min() > default_length_scale(6, 2) / 2


def test_batch_first_pick_is_the_maximum():
    U = _grid()
    scores = -((U - 0.3) ** 2).sum(axis=1)
    idx = select_batch(U, scores, 3)
    assert idx[0] == np.argmax(scores)


def test_batch_without_penalty_clusters():
    U = _grid()
    scores = -((U - 0.3) ** 2).sum(axis=1)
    tight = select_batch(U, scores, 4, length_scale=1e-6)
    spread = select_batch(U, scores, 4)
    spread_d = np.ptp(U[spread], axis=0).sum()
    assert np.ptp(U[tight], axis=0).sum() < spread_d


def test_batch_penalize_train_avoids_existing_points():
    U = _grid()
    train = np.array([[0.5, 0.5]])
    idx = select_batch(U, np.ones(len(U)), 1, U_train=train, penalize_train=True)
    assert np.linalg.norm(U[idx[0]] - train[0]) > 0.1


def test_batch_size_too_large():
    with pytest.raises(ValueError):
        select_batch(np.zeros((3, 2)), np.zeros(3), 4)


@pytest.mark.parametrize("name", BENCHMARKS)
def test_benchmarks_shapes_and_bounds(name):
    b = get_benchmark(name)
    X = b.sample(32, seed=0)
    assert np.all(X >= b.bounds[:, 0]) and np.all(X <= b.bounds[:, 1])
    for level in b.levels:
        Y = b(X, level)
        assert Y.shape[0] == 32 and np.all(np.isfinite(Y))
    assert b.costs[b.high] >= max(b.costs.values())


def test_benchmark_known_optima():
    assert get_benchmark("branin")(np.array([[np.pi, 2.275]]))[0, 0] == pytest.approx(0.397887, abs=1e-5)
    h3 = get_benchmark("hartmann3")
    assert h3(np.array([[0.114614, 0.555649, 0.852547]]))[0, 0] == pytest.approx(-3.86278, abs=1e-4)
    f = get_benchmark("forrester")
    assert f(np.array([[0.757249]]))[0, 0] == pytest.approx(-6.02074, abs=1e-4)


def test_low_fidelity_is_correlated_with_high():
    for name in ("forrester", "currin", "park", "borehole"):
        b = get_benchmark(name)
        X = b.sample(256, seed=1)
        r = np.corrcoef(b(X, "lo").ravel(), b(X, "hi").ravel())[0, 1]
        assert r > 0.5, name


def test_unknown_benchmark_and_sampler():
    with pytest.raises(ValueError):
        get_benchmark("rosenbrock")
    with pytest.raises(ValueError):
        get_benchmark("branin").sample(4, method="grid")
