# Copyright (c) 2026 takotime808

"""Streamlit controls shared by the pages that build an HTML report."""

from typing import Any

import streamlit as st

from multioutreg.report.render import report_json_from_html

REPORT_FORMATS = {"Redesigned": "v2", "Classic": "classic"}


def report_format_control(key: str = "report_format") -> str:
    """Radio for the report style; returns ``"v2"`` or ``"classic"``."""
    label = st.radio(
        "Report format",
        list(REPORT_FORMATS),
        horizontal=True,
        key=key,
        help=(
            "Redesigned: verdict and scorecard first, one section per question, "
            "interactive charts, prints cleanly. Classic: the previous layout."
        ),
    )
    return REPORT_FORMATS[label]


def report_downloads(html: str, file_name: str, key: str, container: Any = None) -> None:
    """The HTML report download, plus the report data (.json) for a redesigned report."""
    box = container if container is not None else st
    box.download_button(
        "Download HTML Report", html, file_name=file_name, mime="text/html",
        key=f"{key}_html")
    data = report_json_from_html(html)
    if data is not None:
        box.download_button(
            "Download report data (.json)", data,
            file_name=file_name.rsplit(".", 1)[0] + "_data.json",
            mime="application/json", key=f"{key}_json",
            help="The numbers behind the report's charts and verdict, for archiving "
                 "or rebuilding the report without retraining.")
