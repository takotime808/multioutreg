# Copyright (c) 2026 takotime808

"""End-to-end runs of the grid-search Streamlit app (see app_harness.py)."""

import io
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from multioutreg.gui.grid_search_core import RandomForestWithUncertainty

from streamlit.testing.v1 import AppTest

from .app_harness import APP_PATH, _script, run_grid_search_app

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
    # Picked on the validation split, not the test set: rf edges out dt on y_step.
    assert _best_models(at) == ["linear", "rf"]
    metrics = at.dataframe[-2].value
    assert list(metrics.index) == ["y_lin", "y_step"]
    assert (metrics["r2"] > 0.9).all()
    # Real calibration metrics, not the old {"dummy_metric": 0.0} placeholder (MF12).
    uq = at.dataframe[-1].value["value"]
    assert "dummy_metric" not in uq.index
    assert 0.0 <= uq["Empirical coverage of 95% Gaussian interval"] <= 1.0
    assert any(b.label == "Download HTML Report" for b in at.get("download_button"))


def test_app_reports_progress(csv_path):
    at = run_grid_search_app(csv_path, ["x0", "x1"], ["y_lin", "y_step"],
                             surrogate_defs=FAST_DEFS)
    assert not at.exception, at.exception
    status = at.status[0]
    assert status.state == "complete"
    assert status.label.startswith("Grid search complete in")


def test_app_results_survive_rerun(csv_path):
    # Any widget interaction (e.g. the download button) reruns the script; the
    # results used to vanish because they were only drawn on the submit run.
    at = run_grid_search_app(csv_path, ["x0", "x1"], ["y_lin", "y_step"],
                             checkboxes={"use_conformal": True},
                             surrogate_defs=FAST_DEFS)
    before = _best_models(at)
    at.run()
    assert not at.exception, at.exception
    assert _best_models(at) == before
    assert any("Conformal Prediction Summary" in m.value for m in at.markdown)
    assert any(b.label == "Download HTML Report" for b in at.get("download_button"))
    assert any("Showing the last run" in c.value for c in at.caption)


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


def test_app_sample_data_runs_without_upload():
    # Nothing uploaded (the stubbed uploader returns None): one click loads the
    # bundled sample with inputs/outputs pre-selected, and a run needs no setup.
    at = AppTest.from_function(_script, args=(APP_PATH, None, FAST_DEFS), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert not at.multiselect  # no data, no column pickers yet
    next(b for b in at.button if b.label == "Try with sample data").click()
    at.run()
    assert any("Using the sample dataset" in i.value for i in at.info)
    by_label = {m.label: m for m in at.multiselect}
    assert by_label["Select input features"].value == [f"feature_{i}" for i in range(5)]
    outputs = ["output_pressure", "output_temperature", "output_velocity"]
    assert by_label["Select output targets"].value == outputs
    # Before any run: the pair plot is drawn from the pre-selected columns,
    # the sample can be downloaded, and the report gets a description.
    assert [e.label for e in at.expander if e.label == "Pair plot of the data"]
    assert at.get("imgs")
    assert any(b.label == "Download sample CSV" for b in at.get("download_button"))
    assert at.text_area[0].value.startswith("Sample dataset:")
    assert not any(b.label == "Try with sample data" for b in at.button)

    next(b for b in at.button if b.label == "Run Grid Search").click()
    at.run()
    assert not at.exception, at.exception
    assert len(_best_models(at)) == 3
    assert list(at.dataframe[-2].value.index) == outputs
    assert any(b.label == "Download HTML Report" for b in at.get("download_button"))
    assert [e.label for e in at.expander if e.label == "Pair plot of the data"]
    # The sample stays loaded on later reruns, like an upload would.
    at.run()
    assert any("Using the sample dataset" in i.value for i in at.info)
    assert len(_best_models(at)) == 3


def test_app_sample_data_can_be_turned_off():
    at = AppTest.from_function(_script, args=(APP_PATH, None, FAST_DEFS), default_timeout=120)
    at.run()
    next(b for b in at.button if b.label == "Try with sample data").click()
    at.run()
    next(b for b in at.button if b.label == "Stop using sample data").click()
    at.run()
    assert not at.exception, at.exception
    assert not at.multiselect
    assert any(b.label == "Try with sample data" for b in at.button)


def test_app_exports_model_with_all_preprocessing(tmp_path):
    # The downloaded pipeline must take raw rows (gaps included) to the same
    # predictions the app scored: imputer -> scaler + PCA -> per-output models.
    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 2, size=(160, 3))
    df = pd.DataFrame({
        "x0": X[:, 0], "x1": X[:, 1], "x2": X[:, 2],
        "y_a": 2 * X[:, 0] + X[:, 2] + 0.05 * rng.normal(size=160),
        "y_b": X[:, 1] ** 2 + 0.05 * rng.normal(size=160),
    })
    df.loc[::11, "x1"] = np.nan
    path = tmp_path / "gaps.csv"
    df.to_csv(path, index=False)

    at = run_grid_search_app(str(path), ["x0", "x1", "x2"], ["y_a", "y_b"],
                             checkboxes={"use_pca": True, "use_conformal": True},
                             surrogate_defs=FAST_DEFS)
    assert not at.exception, at.exception
    labels = [b.label for b in at.get("download_button")]
    assert "Download Trained Model (.joblib)" in labels
    assert "Download Model Config (.json)" in labels

    res = at.session_state["gs_results"]
    bundle = joblib.load(io.BytesIO(res["model_bytes"]))
    pipe = bundle["pipeline"]
    assert [name for name, _ in pipe.steps] == ["columns", "imputer", "pca", "model"]
    assert bundle["input_columns"] == ["x0", "x1", "x2"]
    assert set(bundle["conformal"]["quantiles"]) == {"y_a", "y_b"}

    # Same split as the app; raw test rows through the pipeline reproduce its metrics.
    _, df_test = train_test_split(df, test_size=0.25, random_state=0)
    pred = pipe.predict(df_test)
    for i, name in enumerate(["y_a", "y_b"]):
        assert r2_score(df_test[name], pred[:, i]) == pytest.approx(res["metrics"].loc[name, "r2"])

    config = json.loads(res["model_config"])
    assert [m["model"] for m in config["models"]] == _best_models(at)
    assert config["preprocessing"]["pca"]["n_components"] == 2
    assert config["conformal"] == bundle["conformal"]
