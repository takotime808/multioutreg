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


def _table(at, column):
    """The displayed dataframe that has *column* (tables move between tabs, so not by position)."""
    return next(d.value for d in at.dataframe if column in d.value.columns)


def _best_models(at):
    # The Summary tab's "Chosen model per output" table, in output order.
    return list(_table(at, "Model")["Model"])


def test_app_selects_per_output_and_reports(csv_path):
    at = run_grid_search_app(csv_path, ["x0", "x1"], ["y_lin", "y_step"],
                             surrogate_defs=FAST_DEFS)
    assert not at.exception, at.exception
    # Picked on the validation split, not the test set: rf edges out dt on y_step.
    assert _best_models(at) == ["linear", "rf"]
    metrics = _table(at, "r2")
    assert list(metrics.index) == ["y_lin", "y_step"]
    assert (metrics["r2"] > 0.9).all()
    # Real calibration metrics, not the old {"dummy_metric": 0.0} placeholder (MF12).
    uq = _table(at, "value")["value"]
    assert "dummy_metric" not in uq.index
    assert 0.0 <= uq["Empirical coverage of 95% Gaussian interval"] <= 1.0
    assert any(b.label == "Download HTML Report" for b in at.get("download_button"))


def test_app_builds_both_report_formats(csv_path):
    at = run_grid_search_app(csv_path, ["x0", "x1"], ["y_lin", "y_step"],
                             radios={"Report format": "Both"}, surrogate_defs=FAST_DEFS)
    assert not at.exception, at.exception
    labels = [b.label for b in at.get("download_button")]
    # Once in the Report tab and once in the export section.
    assert labels.count("Download HTML Report (Redesigned)") == 2
    assert labels.count("Download HTML Report (Classic)") == 2
    reports = at.session_state["gs_results"]["html"]
    assert set(reports) == {"v2", "classic"}
    assert 'id="report-data"' in reports["v2"]
    assert 'id="report-data"' not in reports["classic"]


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
    assert list(_table(at, "r2").index) == outputs
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


# ---------------------------------------------------------------------------
# Checks before a run (GUI roadmap 3b): each problem is shown in the form and
# the grid search does not start.
# ---------------------------------------------------------------------------

def _did_run(at):
    try:
        at.session_state["gs_results"]
    except KeyError:
        return False
    return True


def _messages(at):
    return [m.value for m in list(at.error) + list(at.warning)]


def _small_csv(tmp_path, n, extra=None):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x0": rng.normal(size=n), "x1": rng.normal(size=n)})
    df["y"] = 2 * df["x0"] + 0.1 * rng.normal(size=n)
    for name, values in (extra or {}).items():
        df[name] = values
    path = tmp_path / f"small_{n}.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_check_blocks_column_used_as_input_and_output(csv_path):
    at = run_grid_search_app(csv_path, ["x0", "y_lin"], ["y_lin"], surrogate_defs=FAST_DEFS)
    assert not at.exception, at.exception
    assert any("'y_lin' is chosen as both an input and an output" in m for m in _messages(at))
    assert not _did_run(at) and not at.status


def test_check_names_non_numeric_columns(tmp_path):
    path = _small_csv(tmp_path, 40, {"site": ["north", "south"] * 20})
    at = run_grid_search_app(path, ["x0", "site"], ["y"], surrogate_defs=FAST_DEFS)
    assert not at.exception, at.exception
    msg = next(m for m in _messages(at) if "Non-numeric" in m)
    assert "'site'" in msg and "'north'" in msg
    assert not _did_run(at) and not at.status


def test_check_warns_on_too_few_rows(tmp_path):
    at = run_grid_search_app(_small_csv(tmp_path, 8), ["x0", "x1"], ["y"],
                             surrogate_defs=FAST_DEFS)
    assert not at.exception, at.exception
    assert any("Only 8 usable row(s); at least 10 are needed" in w.value for w in at.warning)
    assert not _did_run(at) and not at.status


def test_check_warns_when_conformal_split_is_too_small(tmp_path):
    # 11 rows at alpha=0.1 leave 8 training rows; finite intervals need 9.
    at = run_grid_search_app(_small_csv(tmp_path, 11), ["x0", "x1"], ["y"],
                             checkboxes={"use_conformal": True}, surrogate_defs=FAST_DEFS)
    assert not at.exception, at.exception
    msg = next(w.value for w in at.warning if "Conformal intervals at alpha=0.1" in w.value)
    assert "raise the conformal alpha" in msg
    assert not _did_run(at) and not at.status


def test_check_counts_rows_left_after_dropping_missing_targets(tmp_path):
    y = np.r_[np.arange(9.0), [np.nan] * 6]
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x0": rng.normal(size=15), "x1": rng.normal(size=15), "y": y})
    path = tmp_path / "gappy.csv"
    df.to_csv(path, index=False)
    at = run_grid_search_app(str(path), ["x0", "x1"], ["y"], surrogate_defs=FAST_DEFS)
    msg = next(w.value for w in at.warning if "usable row" in w.value)
    assert "Only 9 usable row(s)" in msg and "6 of 15 rows are dropped" in msg
    assert not _did_run(at)


def test_check_passes_clean_inputs(csv_path):
    at = run_grid_search_app(csv_path, ["x0", "x1"], ["y_lin"],
                             checkboxes={"use_conformal": True}, surrogate_defs=FAST_DEFS)
    assert not at.exception, at.exception
    assert not at.error
    assert _did_run(at)


# ---------------------------------------------------------------------------
# Results in tabs (GUI roadmap 4).
# ---------------------------------------------------------------------------

def test_results_are_in_tabs(csv_path):
    from multioutreg.gui.results_tabs import TABS

    at = run_grid_search_app(csv_path, ["x0", "x1"], ["y_lin", "y_step"],
                             checkboxes={"use_conformal": True}, surrogate_defs=FAST_DEFS)
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == TABS
    # Summary: a metric card per output and a readable model table, not raw JSON.
    assert [m.label for m in at.metric].count("Test R²") == 2
    assert not at.json
    table = _table(at, "Model")
    assert list(table.index) == ["y_lin", "y_step"] and "Settings" in table.columns
    # Predictions, Uncertainty, Explainability and model selection are Plotly charts.
    assert len(at.get("plotly_chart")) >= 8
    # SHAP and partial dependence are computed once and kept with the results.
    explain = at.session_state["gs_results"]["explain"]
    assert set(explain["shap"]) == {"y_lin", "y_step"} and explain["pdp"]["features"]
    assert not explain["errors"]


def test_report_tab_previews_the_report(csv_path):
    at = run_grid_search_app(csv_path, ["x0", "x1"], ["y_lin"], surrogate_defs=FAST_DEFS)
    assert not at.exception, at.exception
    assert any(b.label == "Download HTML Report" for b in at.get("download_button"))
    previews = at.get("iframe")
    assert previews and 'id="report-data"' in previews[0].proto.srcdoc


def test_rerun_redraws_tabs_without_refitting(csv_path, monkeypatch):
    at = run_grid_search_app(csv_path, ["x0", "x1"], ["y_lin", "y_step"],
                             surrogate_defs=FAST_DEFS)
    res = at.session_state["gs_results"]
    snapshot = (res["report_data"], res["explain"])

    import multioutreg.model_selection.evaluation as evaluation

    def _no_refit(*args, **kwargs):
        raise AssertionError("run_evaluation was called on a rerun")

    monkeypatch.setattr(evaluation, "run_evaluation", _no_refit)
    at.run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 5 and len(at.get("plotly_chart")) >= 6
    res = at.session_state["gs_results"]
    assert (res["report_data"], res["explain"]) == snapshot
