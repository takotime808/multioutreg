# Copyright (c) 2026 takotime808
"""Drive the Multi-Fidelity Streamlit page (02) end to end with a cheap candidate grid.

The real page grid-searches every multi-fidelity type x base surrogate, which
takes minutes. Here the optional backends (LightGBM, XGBoost, CatBoost,
SparseGP) are reported missing and the tree surrogates are swapped for the
linear one, so a run takes seconds while executing the same page code
(column selection, imputation, grid search, conformal, plots, report).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
PAGE = os.path.join(ROOT, "multioutreg", "gui", "pages", "02_Multi_Fidelity_Surrogate_Models.py")


def _script(page_path, csv_path):
    # Runs inside AppTest; source is copied verbatim, so imports live here.
    import importlib
    import runpy

    import streamlit as st

    import multioutreg.surrogates as surrogates

    patches = [(st, "file_uploader", lambda *args, **kwargs: csv_path)]
    for module, flag in [("lightgbm_sklearn", "_LIGHTGBM_AVAILABLE"),
                         ("xgboost_sklearn", "_XGBOOST_AVAILABLE"),
                         ("catboost_sklearn", "_CATBOOST_AVAILABLE"),
                         ("sparse_gp_gpytorch", "_GPYTORCH_AVAILABLE")]:
        patches.append((importlib.import_module(f"multioutreg.surrogates.{module}"), flag, False))
    for name in ("RandomForestSurrogate", "HistGradientBoostingSurrogate"):
        patches.append((surrogates, name, surrogates.LinearRegressionSurrogate))

    saved = [(obj, attr, getattr(obj, attr)) for obj, attr, _ in patches]
    for obj, attr, value in patches:
        setattr(obj, attr, value)
    try:
        runpy.run_path(page_path, run_name="__main__")
    finally:
        for obj, attr, value in saved:
            setattr(obj, attr, value)


def run_page(csv_path, inputs, outputs, checkboxes=None, selects=None):
    at = AppTest.from_function(_script, args=(PAGE, csv_path), default_timeout=300)
    at.run()
    at.selectbox(key="fidelity_col_select").set_value("fidelity")
    for key, value in (selects or {}).items():
        at.selectbox(key=key).set_value(value)
    at.run()
    by_label = {m.label: m for m in at.multiselect}
    by_label["Input feature columns"].set_value(inputs)
    by_label["Output target columns"].set_value(outputs)
    by_cb = {c.label: c for c in at.checkbox}
    for label, value in (checkboxes or {}).items():
        by_cb[label].set_value(value)
    next(b for b in at.button if b.label == "Run Grid Search").click()
    at.run()
    return at


def _text(at):
    return " ".join(str(e.value) for e in [*at.markdown, *at.error, *at.warning, *at.info, *at.success])


def _write_csv(path, n_hi=30, gaps=()):
    # Low fidelity: biased copy of the truth on many points; high: the truth on fewer.
    rng = np.random.default_rng(0)
    frames = []
    for level, n, bias in [(0, 60, 0.5), (1, n_hi, 0.0)]:
        X = rng.uniform(-1, 1, size=(n, 2))
        frames.append(pd.DataFrame({
            "a": X[:, 0], "b": X[:, 1], "fidelity": level,
            "y1": 2 * X[:, 0] - X[:, 1] + bias + 0.05 * rng.normal(size=n),
            "y2": X[:, 0] * 0.5 + bias + 0.05 * rng.normal(size=n),
        }))
    df = pd.concat(frames, ignore_index=True)
    df.loc[list(gaps), "a"] = np.nan
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def csv(tmp_path):
    return _write_csv(tmp_path / "mf.csv")


CONFORMAL = {"Compute conformal prediction intervals": True}


def test_default_run_completes_and_reports(csv):
    at = run_page(csv, ["a", "b"], ["y1", "y2"])
    assert not at.exception, [e.message for e in at.exception]
    text = _text(at)
    assert "Grid Search Results" in text
    assert "best:" in text
    assert "Best Model Metrics" in text
    assert len(at.dataframe) >= 2


def test_conformal_run_with_imputed_input(tmp_path):
    # 80 hi-fi rows -> 60 train -> 12 calibration rows, enough for alpha=0.1.
    path = _write_csv(tmp_path / "gappy.csv", n_hi=80, gaps=(2, 40, 70))
    at = run_page(path, ["a", "b"], ["y1"], checkboxes=CONFORMAL,
                  selects={"imp_a": "Impute (KNN)"})
    assert not at.exception, [e.message for e in at.exception]
    assert any("missing value(s) detected" in w.value for w in at.warning)
    text = _text(at)
    assert "Grid Search Results" in text
    assert "Conformal Prediction Summary" in text


def test_conformal_with_too_few_calibration_rows_is_reported_not_raised(csv):
    # 30 hi-fi rows -> 22 train -> 5 calibration rows (9 needed at alpha=0.1).
    at = run_page(csv, ["a", "b"], ["y1"], checkboxes=CONFORMAL)
    assert not at.exception, [e.message for e in at.exception]
    assert any("Conformal prediction skipped: 5 calibration rows" in w.value
               and "at least 9 are needed" in w.value for w in at.warning)
    text = _text(at)
    assert "Grid Search Results" in text
    assert "Conformal Prediction Summary" not in text


@pytest.mark.parametrize("inputs, outputs, message", [
    ([], ["y1"], "at least one input feature"),
    (["a"], [], "at least one output target"),
])
def test_missing_selection_is_reported(csv, inputs, outputs, message):
    at = run_page(csv, inputs, outputs)
    assert not at.exception
    assert any(message in e.value for e in at.error)
