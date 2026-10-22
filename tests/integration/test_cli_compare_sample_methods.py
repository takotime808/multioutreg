# Copyright (c) 2025 takotime808

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

import mor_cli.use_ks_for_sample_bias as ks

CFG = {
    "dependent_variables": ["a", "b", "c", "d"],
    "distributions": ["normal", "gamma", "beta", "uniform"],
    "stat1": [5, 0, 2, -1],
    "stat2": [4, 2, 5, 3],
    "stat3": ["None", 3, 10, "None"],
}


@pytest.fixture
def descs():
    return ks.build_bounds_dists(CFG)


def _run(tmp_path, X):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml.safe_dump(CFG))
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(X, columns=CFG["dependent_variables"]).to_csv(csv_path, index=False)
    return CliRunner().invoke(
        ks.app, ["--csv", str(csv_path), "--yaml", str(yaml_path), "--n-sim", "99"]
    )


@pytest.mark.parametrize("method, factory", [
    ("LHS", ks.create_lhs_samples),
    ("Random", ks.create_random_samples),
    ("Grid", ks.create_grid_samples),
])
def test_compare_sample_methods_non_uniform_marginals(tmp_path, descs, method, factory):
    X = factory(descs, 120, np.random.default_rng(1))
    result = _run(tmp_path, X)
    assert result.exit_code == 0, result.output
    assert f"matches the '{method}' sampling technique" in result.output


def test_compare_sample_methods_misspecified_yaml(tmp_path):
    result = _run(tmp_path, np.random.default_rng(2).random((80, 4)))
    assert result.exit_code == 0, result.output
    assert "does not clearly match" in result.output


def test_string_none_is_missing(descs):
    assert descs[0]["stat3"] is None
    assert descs[3]["stat3"] is None


def test_generators_are_seeded(descs):
    a = ks.create_lhs_samples(descs, 20, np.random.default_rng(0))
    b = ks.create_lhs_samples(descs, 20, np.random.default_rng(0))
    np.testing.assert_array_equal(a, b)


def test_grid_levels_depend_on_dimension(descs):
    # 4 continuous inputs and n=120 -> 3 levels each (81 points), not cbrt(120) = 4 (256 > 120).
    X = ks.create_grid_samples(descs, 120)
    assert X.shape == (81, 4)
    assert all(np.unique(X[:, j]).size == 3 for j in range(4))


def test_beta_grid_respects_loc():
    desc = {"name": "c", "dist": "beta", "stat1": 2, "stat2": 5, "stat3": 10}
    X = ks.create_grid_samples([desc], 9)
    assert X.min() > 10 and X.max() < 11


def test_marginal_ks_reports_p_values(descs):
    X = ks.create_random_samples(descs, 200, np.random.default_rng(3))
    res = ks.marginal_ks_tests(pd.DataFrame(X, columns=CFG["dependent_variables"]), descs)
    assert list(res["variable"]) == CFG["dependent_variables"]
    assert (res["p_value"] > 0.001).all()
