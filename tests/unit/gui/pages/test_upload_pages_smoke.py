# Copyright (c) 2026 takotime808
"""Render each upload-driven Streamlit page with a CSV and check it does not error.

``AppTest`` cannot simulate an upload, so ``st.file_uploader`` is replaced by a
stub returning a CSV path (every page passes the upload straight to
``pd.read_csv``, which accepts a path). Each page is also rendered once with no
upload, and with a degenerate CSV, to cover its early-exit paths.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
PAGES = os.path.join(ROOT, "multioutreg", "gui", "pages")
LHS_CSV = os.path.join(ROOT, "docs", "_static", "example_datasets", "lhs_sampled_dataset.csv")

UPLOAD_PAGES = [
    "sample_infer_dashboard.py",
    "04_Sampling_and_Bias_Dashboard.py",
    "lhs_detector_app.py",
    "sample_infer.py",
    "PCA_Sample_detection.py",
    "Detect_Sampling.py",
]


def _script(page_path, csv_path):
    # Runs inside AppTest; source is copied verbatim, so imports live here.
    import runpy

    import streamlit as st

    real_uploader = st.file_uploader
    st.file_uploader = lambda *args, **kwargs: csv_path
    try:
        runpy.run_path(page_path, run_name="__main__")
    finally:
        st.file_uploader = real_uploader


def run_page(page: str, csv_path: str | None) -> AppTest:
    at = AppTest.from_function(
        _script, args=(os.path.join(PAGES, page), csv_path), default_timeout=180)
    at.run()
    return at


def _errors(at: AppTest) -> list[str]:
    return [e.message for e in at.exception]


@pytest.mark.parametrize("page", UPLOAD_PAGES)
def test_page_without_upload_renders_prompt_only(page):
    at = run_page(page, None)
    assert not _errors(at)


@pytest.mark.parametrize("page", UPLOAD_PAGES)
def test_page_with_lhs_csv_renders_without_exception(page):
    at = run_page(page, LHS_CSV)
    assert not _errors(at), _errors(at)
    # Something beyond the upload prompt was produced.
    assert len(at.markdown) + len(at.dataframe) + len(at.json) + len(at.success) + len(at.info) > 0


@pytest.fixture
def tiny_csv(tmp_path):
    # Fewer rows than most detectors accept: an expected failure point that the
    # pages should report, not crash on.
    path = tmp_path / "tiny.csv"
    pd.DataFrame({"x1": [0.1, 0.5, 0.9], "x2": [0.3, 0.7, 0.2], "y": [1.0, 2.0, 3.0]}).to_csv(path, index=False)
    return str(path)


@pytest.mark.parametrize("page", ["lhs_detector_app.py", "Detect_Sampling.py", "PCA_Sample_detection.py"])
def test_detector_pages_handle_too_few_rows(page, tiny_csv):
    at = run_page(page, tiny_csv)
    shown = " ".join(e.value for e in list(at.error) + list(at.warning) + list(at.info))
    # Either the page shows a message, or it lets the exception surface in
    # at.exception -- but never a silent blank page.
    assert _errors(at) or shown or len(at.markdown) > 0


@pytest.fixture
def non_numeric_csv(tmp_path):
    path = tmp_path / "text.csv"
    pd.DataFrame({"name": list("abcdefghij"), "label": ["u", "v"] * 5}).to_csv(path, index=False)
    return str(path)


@pytest.mark.parametrize("page", ["sample_infer_dashboard.py", "sample_infer.py"])
def test_infer_pages_with_no_numeric_columns_do_not_crash(page, non_numeric_csv):
    at = run_page(page, non_numeric_csv)
    assert not _errors(at), _errors(at)


@pytest.fixture
def grid_csv(tmp_path):
    g = np.array(np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))).reshape(2, -1).T
    path = tmp_path / "grid.csv"
    pd.DataFrame({"x1": g[:, 0], "x2": g[:, 1], "y": g.sum(axis=1)}).to_csv(path, index=False)
    return str(path)


def test_lhs_detector_page_reports_grid(grid_csv):
    at = run_page("lhs_detector_app.py", grid_csv)
    assert not _errors(at)
    text = " ".join(m.value for m in list(at.markdown) + list(at.success) + list(at.info) + list(at.warning))
    assert "grid" in text.lower()


@pytest.mark.parametrize("page", ["05_Sampling_Datasets_Examples.py", "07_Other_Dataset_Examples.py"])
def test_dataset_example_pages_render(page):
    # These pages generate their own datasets (no upload) and take ~30 s each.
    at = AppTest.from_file(os.path.join(PAGES, page), default_timeout=300)
    at.run()
    assert not _errors(at), _errors(at)
    assert len(at.dataframe) > 0
