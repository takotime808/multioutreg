# Copyright (c) 2026 takotime808

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from mor_cli.grid_search import app as grid_search_app
from mor_cli.grid_search_auto_detect import app as auto_detect_app

INPUTS = "x0,x1,x2,x3,x4,x5"
OUTPUTS = "y0,y1"


@pytest.fixture(autouse=True)
def cheap_candidates(monkeypatch):
    """Search only fast linear candidates, as the GUI tests do.

    40 rows take the nested-CV path, which refits the whole selection per
    outer fold; the full candidate lists would take tens of minutes.
    """
    from sklearn.linear_model import BayesianRidge, LinearRegression

    import mor_cli.grid_search as grid_search_cli
    from multioutreg.model_selection import AutoDetectMultiOutputRegressor as ADR
    from multioutreg.model_selection.auto_detect import drop_candidates

    monkeypatch.setattr(grid_search_cli, "default_surrogate_defs", lambda: [
        ("linear", LinearRegression, {}), ("bayesian_ridge", BayesianRidge, {})])
    real_factory = ADR.__dict__["with_vendored_surrogates"]

    def trimmed(cls, *args, **kwargs):
        model = real_factory.__func__(cls, *args, **kwargs)
        drop_candidates(model, set(model._model_names) - {"linear", "bayesian_ridge"})
        return model

    monkeypatch.setattr(ADR, "with_vendored_surrogates", classmethod(trimmed))


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
    # 38 rows: a 25% test split would hold 10 rows, so the CLI uses nested CV.
    assert "nested 5-fold CV" in result.output
    assert "Nested 5-fold CV" in html


def test_cli_auto_detect_small_n_uses_cv_plus(csv_with_gaps, tmp_path):
    out_html = tmp_path / "report.html"
    result = CliRunner().invoke(
        auto_detect_app,
        [str(csv_with_gaps), INPUTS, OUTPUTS, "--use-imputation", "--conformal",
         "--conformal-alpha", "0.1", "--out-html", str(out_html)],
    )
    assert result.exit_code == 0, result.output
    assert "CV+ is used instead" in result.output
    assert "CV+ conformal intervals" in out_html.read_text(encoding="utf-8")
