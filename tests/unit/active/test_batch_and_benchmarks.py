# Copyright (c) 2026 takotime808

import numpy as np
import pytest

from multioutreg.active.batch import default_length_scale, select_batch
from multioutreg.active.benchmarks import (
    BENCHMARKS,
    HIMMELBLAU_MINIMA,
    branin,
    get_benchmark,
    zdt_front,
)
from multioutreg.metrics import pareto_mask


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


def test_new_benchmark_known_values():
    gl = get_benchmark("gramacy_lee")
    assert gl(np.array([[0.548563]]))[0, 0] == pytest.approx(-0.869011, abs=1e-5)
    assert np.allclose(get_benchmark("himmelblau")(HIMMELBLAU_MINIMA), 0.0, atol=1e-8)
    # 10 sin(pi/4) + 0 + 5 + 2.5 at the centre
    f = get_benchmark("friedman1")
    assert f(np.full((1, 10), 0.5))[0, 0] == pytest.approx(10 * np.sin(np.pi / 4) + 7.5)
    V = get_benchmark("viennet2")(np.zeros((1, 2)))
    assert np.allclose(V, [[2 + 1 / 13 + 3, 0.25 + 0.5 - 17, 1 / 175 - 13]])


def test_friedman1_ignores_last_five_inputs():
    f = get_benchmark("friedman1")
    X = f.sample(16, seed=0)
    X2 = X.copy()
    X2[:, 5:] = np.random.default_rng(1).random((16, 5))
    assert np.allclose(f(X), f(X2))


def test_zdt_on_front_and_dimension():
    b = get_benchmark("zdt1", n_features=3)
    assert b.n_features == 3 and get_benchmark("zdt2").n_features == 6
    # x2..xd = 0 gives g = 1, so f2 = 1 - sqrt(f1)
    assert np.allclose(b(np.array([[0.25, 0.0, 0.0]])), [[0.25, 0.5]])
    with pytest.raises(ValueError):
        get_benchmark("zdt1", n_features=1)
    with pytest.raises(ValueError):
        get_benchmark("branin", n_features=3)


@pytest.mark.parametrize("name", ["zdt1", "zdt2", "zdt3"])
def test_zdt_front_is_non_dominated_and_dominates_samples(name):
    F = zdt_front(name)
    assert pareto_mask(F).all()
    # Random designs (g > 1) never beat the front.
    Y = get_benchmark(name).__call__(get_benchmark(name).sample(256, seed=0))
    assert not pareto_mask(np.vstack([F, Y]))[len(F):].any()


def test_zdt3_front_is_disconnected():
    F = zdt_front("zdt3", n=2000)
    assert len(F) < 2000
    gaps = np.diff(F[:, 0])
    assert (gaps > 0.05).sum() == 4  # five pieces
    with pytest.raises(ValueError):
        zdt_front("viennet2")


def test_mf_branin_and_heterogeneous_relations():
    b = get_benchmark("branin_mf")
    X = b.sample(8, seed=0)
    expected_lo = 10 * branin(1.2 * X) + (5 * X[:, 0] + 10 * X[:, 1])[:, None]
    assert np.allclose(b(X, "lo"), expected_lo)
    assert np.allclose(b(X), branin(X))
    h = get_benchmark("heterogeneous")
    x = h.sample(8, seed=0)
    assert np.allclose(h(x, "hi"), (h(x, "lo") - 1 + x) / (1 + 0.25 * x))


def test_low_fidelity_is_correlated_with_high():
    for name in ("forrester", "currin", "park", "borehole", "branin_mf", "heterogeneous"):
        b = get_benchmark(name)
        X = b.sample(256, seed=1)
        r = np.corrcoef(b(X, "lo").ravel(), b(X, "hi").ravel())[0, 1]
        assert r > 0.5, name


def test_unknown_benchmark_and_sampler():
    with pytest.raises(ValueError):
        get_benchmark("rosenbrock")
    with pytest.raises(ValueError):
        get_benchmark("branin").sample(4, method="grid")
