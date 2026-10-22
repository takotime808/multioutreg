# Copyright (c) 2026 takotime808

"""Drive the grid-search Streamlit app end to end with ``AppTest``.

``AppTest`` cannot simulate a file upload, so :func:`run_grid_search_app`
runs the real script with ``st.file_uploader`` replaced by a stub that returns
a CSV path (``pd.read_csv`` accepts either). The candidate list can be
shortened so a full run takes seconds instead of minutes.
"""

from __future__ import annotations

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "multioutreg", "gui", "Grid_Search_Surrogate_Models.py"))


def _script(app_path, csv_path, surrogate_defs):
    # Runs inside AppTest; source is copied verbatim, so imports live here.
    import runpy

    import streamlit as st

    import multioutreg.gui.grid_search_core as core

    real_uploader, real_defs = st.file_uploader, core.default_surrogate_defs
    st.file_uploader = lambda *args, **kwargs: csv_path
    if surrogate_defs is not None:
        core.default_surrogate_defs = lambda: list(surrogate_defs)
    try:
        runpy.run_path(app_path, run_name="__main__")
    finally:
        st.file_uploader, core.default_surrogate_defs = real_uploader, real_defs


def run_grid_search_app(
    csv_path: str,
    input_cols: list[str],
    output_cols: list[str],
    *,
    checkboxes: dict[str, bool] | None = None,
    surrogate_defs: list | None = None,
    timeout: float = 120,
) -> AppTest:
    """Load *csv_path*, pick the columns, tick *checkboxes*, and submit the form.

    *checkboxes* maps a checkbox key (e.g. ``"use_conformal"``) or label to its
    value. Returns the ``AppTest`` after the grid search has run, for asserting
    on ``at.exception``, ``at.json``, ``at.dataframe`` and so on.
    """
    at = AppTest.from_function(
        _script, args=(APP_PATH, csv_path, surrogate_defs), default_timeout=timeout)
    at.run()
    for name, value in (checkboxes or {}).items():
        box = next((c for c in at.checkbox if c.key == name or c.label == name), None)
        if box is None:
            raise KeyError(f"No checkbox with key or label {name!r}")
        box.set_value(value)
    at.run()
    by_label = {m.label: m for m in at.multiselect}
    by_label["Select input features"].set_value(input_cols)
    by_label["Select output targets"].set_value(output_cols)
    next(b for b in at.button if b.label == "Run Grid Search").click()
    at.run()
    return at
