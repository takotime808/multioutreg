# Copyright (c) 2026 takotime808

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from mor_cli.grid_search import app as grid_search_app
from mor_cli.grid_search_auto_detect import app as auto_detect_app

INPUTS = "x0,x1,x2,x3,x4,x5"
OUTPUTS = "y0,y1"


@pytest.fixture
def csv_with_gaps(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(40, 6))
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(6)])
    df["y0"] = X[:, 0] + X[:, 1] ** 2 + 0.05 * rng.normal(size=40)
    df["y1"] = X[:, 2] * X[:, 3] + 0.05 * rng.normal(size=40)
    df.loc[[3, 10, 30], "x1"] = np.nan   # inputs: imputed after the split
    df.loc[[7, 20], "y0"] = np.nan       # targets: rows must be dropped
    path = tmp_path / "gaps.csv"
    df.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("app", [grid_search_app, auto_detect_app], ids=["grid_search", "auto_detect"])
def test_cli_imputation_drops_missing_targets(app, csv_with_gaps, tmp_path):
    out_html = tmp_path / "report.html"
    result = CliRunner().invoke(
        app,
        [str(csv_with_gaps), INPUTS, OUTPUTS, "--use-imputation", "--out-html", str(out_html)],
    )
    assert result.exit_code == 0, result.output
    html = out_html.read_text(encoding="utf-8")
    assert "Rows dropped (missing target)" in html
    assert "Imputed (KNN, fit on training split)" in html
