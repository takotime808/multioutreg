# Copyright (c) 2026 takotime808
"""Drive the AutoDetect Streamlit page (01) end to end with a trimmed candidate list.

The real page grid-searches every vendored surrogate, which takes minutes.
Here ``with_vendored_surrogates`` keeps only the cheap linear and Bayesian-ridge
candidates, so each run takes seconds while exercising the same page code
(imputation, PCA, conformal intervals, regret metrics, report export).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
PAGE = os.path.join(ROOT, "multioutreg", "gui", "pages", "01_Use_Auto_Detect_MultiOutput_Regressor.py")
KEEP = ("linear", "bayesian_ridge")


def _script(page_path, csv_path, keep):
    # Runs inside AppTest; source is copied verbatim, so imports live here.
    import runpy

    import streamlit as st

    from multioutreg.model_selection import AutoDetectMultiOutputRegressor as ADR

    real_uploader = st.file_uploader
    real_factory = ADR.__dict__["with_vendored_surrogates"]

    def trimmed(cls, *args, **kwargs):
        m = real_factory.__func__(cls, *args, **kwargs)
        idx = [i for i, n in enumerate(m._model_names) if n in keep]
        m.estimators = [m.estimators[i] for i in idx]
        m.param_spaces = [m.param_spaces[i] for i in idx]
        m._surrogate_constructors = [m._surrogate_constructors[i] for i in idx]
        m._model_names = [m._model_names[i] for i in idx]
        return m

    st.file_uploader = lambda *args, **kwargs: csv_path
    ADR.with_vendored_surrogates = classmethod(trimmed)
    try:
        runpy.run_path(page_path, run_name="__main__")
    finally:
        st.file_uploader = real_uploader
        ADR.with_vendored_surrogates = real_factory


def run_page(csv_path, inputs, outputs, checkboxes=None, selects=None):
    at = AppTest.from_function(_script, args=(PAGE, csv_path, KEEP), default_timeout=300)
    at.run()
    for key, value in (checkboxes or {}).items():
        at.checkbox(key=key).set_value(value)
    at.run()
    for key, value in (selects or {}).items():
        at.selectbox(key=key).set_value(value)
    if selects:
        at.run()
    by_label = {m.label: m for m in at.multiselect}
    by_label["Select input features"].set_value(inputs)
    by_label["Select output targets"].set_value(outputs)
    next(b for b in at.button if b.label == "Run Grid Search").click()
    at.run()
    return at


@pytest.fixture
def csv(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(80, 3))
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["y1"] = 2 * df.a - df.b + 0.1 * rng.normal(size=80)
    df["y2"] = df.c + 0.5 * df.a + 0.1 * rng.normal(size=80)
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


def _errors(at):
    return [e.message for e in at.exception]


def test_page_without_upload_is_idle():
    at = AppTest.from_function(_script, args=(PAGE, None, KEEP), default_timeout=60)
    at.run()
    assert not _errors(at)
    assert not [b for b in at.button if b.label == "Run Grid Search"]


def test_default_run_completes_and_reports_metrics(csv):
    at = run_page(csv, ["a", "b", "c"], ["y1", "y2"])
    assert not _errors(at), _errors(at)
    assert len(at.dataframe) + len(at.table) > 0


def test_run_with_conformal_and_pca(csv):
    at = run_page(
        csv, ["a", "b", "c"], ["y1", "y2"],
        checkboxes={"use_conformal": True, "use_pca": True},
    )
    assert not _errors(at), _errors(at)


def test_single_output_run(csv):
    at = run_page(csv, ["a", "b"], ["y1"])
    assert not _errors(at), _errors(at)


def test_missing_values_are_handled(tmp_path, csv):
    df = pd.read_csv(csv)
    df.loc[::7, "a"] = np.nan
    df.loc[::11, "y1"] = np.nan
    path = tmp_path / "gaps.csv"
    df.to_csv(path, index=False)
    at = run_page(str(path), ["a", "b", "c"], ["y1", "y2"])
    assert not _errors(at), _errors(at)
    # The page lists the columns with gaps before the run.
    assert any("missing" in m.value.lower() for m in list(at.markdown) + list(at.warning) + list(at.info))


def test_submit_without_outputs_does_not_run(csv):
    at = AppTest.from_function(_script, args=(PAGE, csv, KEEP), default_timeout=60)
    at.run()
    by_label = {m.label: m for m in at.multiselect}
    by_label["Select input features"].set_value(["a"])
    next(b for b in at.button if b.label == "Run Grid Search").click()
    at.run()
    assert not _errors(at)
    assert len(at.dataframe) == 0 or all("metric" not in str(d.value).lower() for d in at.dataframe)
