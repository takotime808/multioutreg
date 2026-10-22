# Copyright (c) 2025 takotime808

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import qmc

from multioutreg.sampling.detect import (
    default_n_sim,
    detect_sampling_method,
    guess_input_columns,
)

DATA_DIR = Path(__file__).resolve().parents[3] / "docs" / "_static" / "example_datasets"


def _design(method, n, d, seed):
    rng = np.random.default_rng(seed)
    if method == "Random":
        return rng.random((n, d))
    if method == "LHS":
        return qmc.LatinHypercube(d=d, seed=rng).random(n)
    if method == "Sobol":
        return qmc.Sobol(d=d, seed=rng).random(n)
    raise ValueError(method)


@pytest.mark.parametrize("method", ["Random", "LHS", "Sobol"])
def test_detects_method_on_unknown_box(method):
    # Rescaled to an arbitrary box so the bounds have to be estimated.
    X = _design(method, 100, 3, seed=1) * [2.0, 50.0, 0.1] + [-1.0, 10.0, 3.0]
    result = detect_sampling_method(X, n_sim=99)
    assert result.method == method
    assert result.explanation.endswith(f"-> {method}")
    assert result.p_values[method] >= 0.05


def test_known_bounds_lhs_fills_every_stratum():
    U = _design("LHS", 60, 4, seed=2)
    result = detect_sampling_method(U, bounds=(0.0, 1.0), n_sim=99)
    assert result.statistics["strata"] == pytest.approx(1.0)
    assert result.method == "LHS"


def test_example_lhs_dataset_detected_without_response():
    """Audit E9: the repo's own LHS file was classified as "not LHS"."""
    df = pd.read_csv(DATA_DIR / "lhs_sampled_dataset.csv")
    assert detect_sampling_method(df[["x1", "x2"]].to_numpy()).method == "LHS"
    # The response is not part of the design; including it hides the LHS.
    assert detect_sampling_method(df.to_numpy()).method != "LHS"


def test_full_factorial_grid():
    g = np.linspace(0, 1, 5)
    X = np.array(np.meshgrid(g, g, g)).reshape(3, -1).T
    result = detect_sampling_method(X)
    assert result.method == "Grid"
    assert "full-factorial" in result.explanation


def test_grid_with_derived_columns():
    g = np.linspace(0, 1, 10)
    x1, x2 = (a.ravel() for a in np.meshgrid(g, g))
    X = np.column_stack([x1, x2, np.sin(np.pi * x1) * np.cos(np.pi * x2)])
    result = detect_sampling_method(X)
    assert result.method == "Grid"
    assert result.columns_used == [0, 1]


def test_unscrambled_sobol_matched_exactly():
    X = qmc.Sobol(d=2, scramble=False).random(64) * 3 + 1
    result = detect_sampling_method(X[::-1])
    assert result.method == "Sobol"
    assert "unscrambled Sobol" in result.explanation


def test_non_uniform_marginals_are_uncertain():
    X = np.random.default_rng(3).normal(size=(200, 3))
    assert detect_sampling_method(X, n_sim=99).method == "Uncertain"


def test_too_few_points_is_uncertain():
    X = np.random.default_rng(4).random((6, 2))
    assert detect_sampling_method(X).method == "Uncertain"


def test_rejects_empty_input():
    with pytest.raises(ValueError):
        detect_sampling_method(np.empty((0, 3)))
    with pytest.raises(ValueError):
        detect_sampling_method(None)


def test_seeded_result_is_reproducible():
    X = _design("LHS", 50, 2, seed=5)
    a = detect_sampling_method(X, n_sim=49, random_state=7)
    b = detect_sampling_method(X, n_sim=49, random_state=7)
    assert a.p_values == b.p_values


@pytest.mark.parametrize("n", [10, 500, 501, 2000, 2001, 10_000])
def test_default_n_sim_can_reject_at_5_percent(n):
    # Smallest attainable Monte Carlo p-value is 1 / (B + 1).
    assert 1 / (default_n_sim(n) + 1) < 0.05


def test_guess_input_columns():
    assert guess_input_columns(["x1", "x2", "y", "y2", "response", "target_a"]) == ["x1", "x2"]
    assert guess_input_columns(["y1", "y2"]) == ["y1", "y2"]
