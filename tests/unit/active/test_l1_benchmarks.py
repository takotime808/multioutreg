# Copyright (c) 2026 takotime808
import numpy as np
import pytest

from multioutreg.active.benchmarks import L1_BENCHMARKS, get_benchmark

# Rows of AVT331_L1Problem_MinMaxValues.csv from the authors' repository
# (gitlab.com/qudo046/avt-331-l1-benchmarks): minimizer and f_min, and where it
# is a single point, the maximizer and f_max (the CSV stores -f_max).
REFERENCE = {
    "mf1.1": ([0.757248751], -6.020740056, [1.0], 15.82973195),
    "mf4.1": ([0.275501812], -0.624998901, [0.445870846], 0.361513623),
    "mf2.1": ([1, 1], 0.0, [-2, -2], 3609.0),
    "mf3.1": ([0.1, 0.1], 0.0, [-0.018112915, 0.021701618], 4.020040611),
    "mf4.2": ([0, 0], -0.562712306, [1, 1], 1.835000108),
    "mf5.1": ([2.46740108, 2.19324544], -0.999999999, None, None),
    "mf4.3": ([0, 0.061716432, 0.012401164], -0.562712306, [1, 1, 1], 4.359413062),
    "mf5.2": ([1, 3.946018148, 4, 3.28627657], -1.0, None, None),
    "mf2.2": ([1] * 5, 0.0, [-2] * 5, 14436.0),
    "mf3.2": ([0.1] * 5, 0.0, None, None),
    "mf2.3": ([1] * 10, 0.0, [-2] * 10, 32481.0),
    "mf3.3": ([0.1] * 10, 0.0, None, None),
}


@pytest.mark.parametrize("name", sorted(REFERENCE))
def test_matches_authors_reference_values(name):
    b = get_benchmark(name)
    x_min, f_min, x_max, f_max = REFERENCE[name]
    assert b.true_value(np.array([x_min]))[0, 0] == pytest.approx(f_min, abs=1e-7)
    assert b.minimum == pytest.approx(f_min, abs=1e-6)
    if x_max is not None:
        assert b.true_value(np.array([x_max]))[0, 0] == pytest.approx(f_max, rel=1e-8)
        assert b.maximum == pytest.approx(f_max, rel=1e-8)


@pytest.mark.parametrize("name", L1_BENCHMARKS)
def test_l1_structure(name):
    b = get_benchmark(name, seed=0)
    X = b.sample(64, seed=0)
    assert np.all(X >= b.bounds[:, 0]) and np.all(X <= b.bounds[:, 1])
    assert b.high == "hi" and b.levels[0] == "lo"
    assert b.costs["hi"] == 1.0 and all(b.costs[l] < 1 for l in b.levels[:-1])
    assert b.budget is not None and b.budget >= 100
    for lvl in b.levels:
        Y = b(X, lvl)
        assert Y.shape == (64, 1) and np.all(np.isfinite(Y))
    # Nothing sampled beats the stated minimum or exceeds the stated maximum.
    F = b.true_value(b.sample(2048, seed=1))
    assert F.min() >= b.minimum - 1e-9 and F.max() <= b.maximum + 1e-9
    # The stated optimum scores zero error.
    if b.argmin is not None:
        e = b.optimum_errors(b.argmin[:1])
        assert e["Ex"] == pytest.approx(0, abs=1e-9) and e["Ef"] == pytest.approx(0, abs=1e-6)


def test_level_counts_costs_and_dimensions():
    assert get_benchmark("mf1.1").levels == ["lo", "mid_lo", "mid_hi", "hi"]
    assert get_benchmark("mf1.1").costs == {"lo": 0.05, "mid_lo": 0.1, "mid_hi": 0.5, "hi": 1.0}
    assert get_benchmark("mf3.3").levels == ["lo", "mid", "hi"]
    assert get_benchmark("mf3.3").costs["lo"] == pytest.approx(3.91e-3)
    assert [get_benchmark(n).n_features for n in ("mf2.1", "mf2.2", "mf2.3")] == [2, 5, 10]
    assert get_benchmark("mf5.2").budget == 400


def test_forrester_levels_follow_the_paper():
    b = get_benchmark("mf1.1")
    x = np.linspace(0, 1, 11)[:, None]
    f1 = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
    assert np.allclose(b(x, "hi"), f1)
    assert np.allclose(b(x, "mid_hi"), (5.5 * x - 2.5) ** 2 * np.sin(12 * x - 4))
    assert np.allclose(b(x, "mid_lo"), 0.75 * f1 + 5 * (x - 0.5) - 2)
    assert np.allclose(b(x, "lo"), 0.5 * f1 + 10 * (x - 0.5) - 5)
    # MF4.1's high level is the existing heterogeneous benchmark.
    h, h_old = get_benchmark("mf4.1"), get_benchmark("heterogeneous")
    assert np.allclose(h(x), h_old(x)) and np.allclose(h(x, "lo"), h_old(x, "lo"))


def test_jump_forrester_and_rastrigin_details():
    j = get_benchmark("mf1.2")
    # +10 jump in the high level just right of 0.5; the low level follows the authors' code.
    left, right = j(np.array([[0.5]]))[0, 0], j(np.array([[0.5 + 1e-9]]))[0, 0]
    assert right - left == pytest.approx(10, abs=1e-6)
    assert j(np.array([[0.75]]), "lo")[0, 0] == pytest.approx(0.5 * j(np.array([[0.75]]))[0, 0] + 2.5 - 7)
    r = get_benchmark("mf3.2")
    # Only the last two coordinates are rotated, so moving x1 off the optimum
    # changes only its own term. At the high level the error term vanishes (Theta = 0).
    X = np.full((1, 5), 0.1)
    X2 = X.copy()
    X2[0, 0] = 0.15
    d = r.true_value(X2) - r.true_value(X)
    assert d[0, 0] == pytest.approx(0.05 ** 2 + 1 - np.cos(10 * np.pi * 0.05), abs=1e-12)


def test_heterogeneous_optimum_is_the_x1_face():
    b = get_benchmark("mf4.3")
    rng = np.random.default_rng(0)
    face = np.column_stack([np.zeros(16), rng.random((16, 2))])
    assert np.allclose(b.true_value(face), b.minimum, atol=1e-9)
    e = b.optimum_errors([[0.0, 0.7, 0.3]])
    assert e == {"Ex": 0.0, "Ef": pytest.approx(0, abs=1e-9), "Et": pytest.approx(0, abs=1e-9)}
    assert b.optimum_errors([[0.3, 0.7, 0.3]])["Ex"] == pytest.approx(0.3 / np.sqrt(3))


def test_spring_masses_has_a_second_global_minimizer():
    # Found by a global search on our implementation; far from the Table 4 point.
    b = get_benchmark("mf5.2")
    other = np.array([[3.2647, 2.8543, 1.2267, 1.4122]])
    assert b.true_value(other)[0, 0] == pytest.approx(-1.0, abs=1e-3)
    assert b.optimum_errors(other)["Ex"] > 0.3


def test_paciorek_noise_and_truth():
    b = get_benchmark("mf6", seed=0)
    X = b.sample(500, seed=0)
    truth = b.true_value(X)
    assert np.allclose(truth, np.sin(1 / (X[:, :1] * X[:, 1:])))
    assert np.std(b(X) - truth) == pytest.approx(0.0125, rel=0.15)
    lo_resid = b(X, "lo") - (truth - 2.25 * np.cos(1 / (X[:, :1] * X[:, 1:])))
    assert np.std(lo_resid) == pytest.approx(0.075, rel=0.15)
    # Same seed, same noise; the optimum curves score Ex = 0.
    assert np.allclose(get_benchmark("mf6", seed=3)(X), get_benchmark("mf6", seed=3)(X))
    on_curve = np.array([[0.5, 2 / (3 * np.pi) / 0.5]])
    assert b.true_value(on_curve)[0, 0] == pytest.approx(-1.0)
    assert b.optimum_errors(on_curve)["Ex"] < 1e-3


def test_l1_rejects_cost_ratio_and_n_features():
    with pytest.raises(ValueError):
        get_benchmark("mf2.1", cost_ratio=5)
    with pytest.raises(ValueError):
        get_benchmark("mf2.1", n_features=3)
    with pytest.raises(ValueError):
        get_benchmark("branin").optimum_errors([[0, 0]])
    assert get_benchmark("forrester").costs == {"lo": 1.0, "hi": 10.0}
