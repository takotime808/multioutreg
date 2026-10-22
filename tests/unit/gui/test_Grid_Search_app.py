# Copyright (c) 2026 takotime808

"""End-to-end runs of the grid-search Streamlit app (see app_harness.py)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from multioutreg.gui.grid_search_core import RandomForestWithUncertainty

from .app_harness import run_grid_search_app

FAST_DEFS = [
    ("linear", LinearRegression, {}),
    ("dt", DecisionTreeRegressor, {"max_depth": [2]}),
    ("rf", RandomForestWithUncertainty, {"n_estimators": [20], "max_depth": [3]}),
]


@pytest.fixture
def csv_path(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(120, 2))
    df = pd.DataFrame({
        "x0": X[:, 0], "x1": X[:, 1],
        "y_lin": 3 * X[:, 0] - X[:, 1] + 0.05 * rng.normal(size=120),
        "y_step": np.where(X[:, 0] > 0, 5.0, -5.0) + 0.05 * rng.normal(size=120),
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


def _best_models(at):
    combo = next(j.value for j in at.json if isinstance(j.value, (list, str)))
    if isinstance(combo, str):
        import json
        combo = json.loads(combo)
    return [c["model"] for c in combo]


def test_app_selects_per_output_and_reports(csv_path):
    at = run_grid_search_app(csv_path, ["x0", "x1"], ["y_lin", "y_step"],
                             surrogate_defs=FAST_DEFS)
    assert not at.exception, at.exception
    assert _best_models(at) == ["linear", "dt"]
    metrics = at.dataframe[-2].value
    assert list(metrics.index) == ["y_lin", "y_step"]
    assert (metrics["r2"] > 0.9).all()
    # Real calibration metrics, not the old {"dummy_metric": 0.0} placeholder (MF12).
    uq = at.dataframe[-1].value["value"]
    assert "dummy_metric" not in uq.index
    assert 0.0 <= uq["Empirical coverage of 95% Gaussian interval"] <= 1.0
    assert any(b.label == "Download HTML Report" for b in at.get("download_button"))


def test_app_conformal_path_runs(csv_path):
    # Smoke test only: the calibration split itself (formerly test residuals)
    # is not observable from the UI.
    at = run_grid_search_app(csv_path, ["x0", "x1"], ["y_lin", "y_step"],
                             checkboxes={"use_conformal": True},
                             surrogate_defs=FAST_DEFS)
    assert not at.exception, at.exception
    assert any("Conformal Prediction Summary" in m.value for m in at.markdown)


def test_app_runs_with_default_candidates(csv_path):
    # The real candidate list, with expensive models skipped to stay fast.
    at = run_grid_search_app(
        csv_path, ["x0", "x1"], ["y_lin"],
        checkboxes={"Skip computationally expensive models": True}, timeout=300)
    assert not at.exception, at.exception
    assert len(_best_models(at)) == 1
