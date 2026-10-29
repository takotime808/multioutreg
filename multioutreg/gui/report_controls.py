# Copyright (c) 2026 takotime808

"""Streamlit controls shared by the pages that build an HTML report."""

from typing import Any, Dict, Tuple

import streamlit as st

from multioutreg.report.render import report_json_from_html

REPORT_FORMATS = {
    "Redesigned": ("v2",),
    "Classic": ("classic",),
    "Both": ("v2", "classic"),
}
STYLE_LABELS = {"v2": "Redesigned", "classic": "Classic"}


def report_format_control(key: str = "report_format") -> Tuple[str, ...]:
    """Radio for the report style; returns the styles to build, e.g. ``("v2",)``."""
    label = st.radio(
        "Report format",
        list(REPORT_FORMATS),
        horizontal=True,
        key=key,
        help=(
            "Redesigned: verdict and scorecard first, one section per question, "
            "interactive charts, prints cleanly. Classic: the previous layout. "
            "Both: build both from the same run, to compare them side by side."
        ),
    )
    return REPORT_FORMATS[label]


def report_downloads(html: str, file_name: str, key: str, container: Any = None,
                     label: str = "Download HTML Report") -> None:
    """The HTML report download, plus the report data (.json) for a redesigned report."""
    box = container if container is not None else st
    box.download_button(
        label, html, file_name=file_name, mime="text/html",
        key=f"{key}_html")
    data = report_json_from_html(html)
    if data is not None:
        box.download_button(
            "Download report data (.json)", data,
            file_name=file_name.rsplit(".", 1)[0] + "_data.json",
            mime="application/json", key=f"{key}_json",
            help="The numbers behind the report's charts and verdict, for archiving "
                 "or rebuilding the report without retraining.")


def report_downloads_all(reports: Dict[str, str], file_name: str, key: str,
                         container: Any = None) -> None:
    """:func:`report_downloads` for each ``{style: html}``.

    One report keeps the plain button and file name; with several, each gets
    its style in the label and file name (``model_report_classic.html``).
    """
    if len(reports) == 1:
        report_downloads(next(iter(reports.values())), file_name, key, container)
        return
    stem, ext = file_name.rsplit(".", 1)
    for style, html in reports.items():
        name = STYLE_LABELS.get(style, style)
        report_downloads(html, f"{stem}_{name.lower()}.{ext}", f"{key}_{style}", container,
                         label=f"Download HTML Report ({name})")
