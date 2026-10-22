# Copyright (c) 2025 takotime808

import numpy as np
import pandas as pd
from scipy.stats import qmc
from typer.testing import CliRunner
from mor_cli.detect_sampling import app as detect_app


def test_cli_detect_sampling(tmp_path):
    data = pd.DataFrame(np.random.rand(10, 3), columns=["a", "b", "c"])
    csv_path = tmp_path / "data.csv"
    data.to_csv(csv_path, index=False)

    runner = CliRunner()
    result = runner.invoke(detect_app, [str(csv_path)])
    assert result.exit_code == 0
    assert "Detected sampling:" in result.stdout


def test_cli_detect_sampling_excludes_response(tmp_path):
    X = qmc.LatinHypercube(d=2, seed=0).random(100)
    df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "y": np.sin(6 * X[:, 0]) + X[:, 1] ** 2})
    csv_path = tmp_path / "lhs.csv"
    df.to_csv(csv_path, index=False)

    runner = CliRunner()
    result = runner.invoke(detect_app, [str(csv_path), "--exclude", "y", "--n-sim", "99"])
    assert result.exit_code == 0, result.output
    assert "Columns tested: x1, x2" in result.stdout
    assert "Detected sampling: LHS" in result.stdout

    result = runner.invoke(detect_app, [str(csv_path), "--columns", "x1,x2", "--n-sim", "99"])
    assert "Detected sampling: LHS" in result.stdout


def test_cli_detect_sampling_unknown_column(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(np.random.rand(10, 2), columns=["a", "b"]).to_csv(csv_path, index=False)
    result = CliRunner().invoke(detect_app, [str(csv_path), "--columns", "a,zzz"])
    assert result.exit_code == 1
    assert "zzz" in result.stdout
