# Copyright (c) 2026 takotime808

"""Render :class:`~multioutreg.report.data.ReportData` as a single HTML file.

The v2 report is one self-contained file: the stylesheet and chart script
(``static/report_v2.css`` and ``static/report_v2.js``) are inlined, and the
chart data is embedded as JSON in ``<script type="application/json"
id="report-data">``. The verdict, scorecard, findings, next steps and every
table are rendered server-side, so they read without JavaScript.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from multioutreg.report.data import ReportData
from multioutreg.report.verdict import READY_LABEL, coverage_band

STYLES = ("v2", "classic")
_HERE = Path(__file__).resolve().parent
_JSON_ESCAPES = {"<": "\\u003c", ">": "\\u003e", "&": "\\u0026",
                 " ": "\\u2028", " ": "\\u2029"}


def resolve_style(style: Optional[str] = None, template_path: Any = None) -> str:
    """Pick the report style.

    A custom template (``template_path`` or the ``MOR_TEMPLATE_PATH``
    environment variable) is a classic Jinja template, so it always means
    ``"classic"``. Otherwise ``style`` wins, then ``MOR_REPORT_STYLE``, then
    ``"v2"``.
    """
    if template_path or os.environ.get("MOR_TEMPLATE_PATH"):
        return "classic"
    chosen = (style or os.environ.get("MOR_REPORT_STYLE") or "v2").strip().lower()
    if chosen in ("redesigned", "new"):
        chosen = "v2"
    if chosen not in STYLES:
        raise ValueError(f"Unknown report style {chosen!r}; use one of {STYLES}")
    return chosen


def json_for_script(text: str) -> str:
    """Escape JSON so it can sit inside a ``<script>`` element.

    ``<``, ``>`` and ``&`` become ``\\u003c`` etc. (still valid JSON), so a
    value containing ``</script>`` cannot end the element early.
    """
    return re.sub("[<>&  ]", lambda m: _JSON_ESCAPES[m.group(0)], text)


_BLOB_RE = re.compile(
    r'<script type="application/json" id="report-data">(.*?)</script>', re.S)


def report_json_from_html(html: str) -> Optional[str]:
    """The embedded report data of a v2 report (``None`` for a classic one).

    Used for the "Download report data (.json)" button, so a caller that only
    has the HTML (e.g. from a background job) can offer the data too.
    """
    m = _BLOB_RE.search(html or "")
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# formatting filters
# ---------------------------------------------------------------------------

def _num(v: Any, sig: int = 4) -> str:
    if v is None:
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(x):
        return "—"
    if x == 0:
        return "0"
    if abs(x) >= 10 ** sig or abs(x) < 10 ** -3:
        return f"{x:.{max(sig - 1, 1)}e}"
    return f"{x:.{sig}g}"


def _pct(v: Any, d: int = 0) -> str:
    if v is None:
        return "—"
    return f"{100 * float(v):.{d}f}%"


def _spct(v: Any, d: int = 0) -> str:
    """A change already in percent, with a sign (− for negative)."""
    if v is None:
        return "—"
    s = f"{abs(float(v)):.{d}f}%"
    return ("+" if v > 0 else "−" if v < 0 else "") + s


def _names(items: List[str]) -> str:
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# ---------------------------------------------------------------------------
# section headlines (the finding, not the chart type)
# ---------------------------------------------------------------------------

def _headlines(d: Dict[str, Any]) -> Dict[str, str]:
    outs = d["outputs"]
    th = d["thresholds"]
    h: Dict[str, str] = {}

    low = [o["name"] for o in outs if o["status"]["accuracy"] in ("warn", "crit")]
    h["sec-2"] = (f"{_names(low)} {'is' if len(low) == 1 else 'are'} below the "
                  f"R² {th['r2_good']:.2f} target" if low
                  else ("Accurate on the test set" if len(outs) == 1
                        else "Every output reaches the accuracy target"))

    ok = [o["name"] for o in outs if o["status"]["gaussian"] == "good"]
    bad = [o for o in outs if o["status"]["gaussian"] in ("warn", "crit")]
    if not bad:
        h["sec-3"] = ("The Gaussian error bars hold" if len(outs) == 1
                      else "Gaussian error bars hold for every output")
    else:
        conf_ok = [o["name"] for o in bad if o["status"]["conformal"] == "good"]
        rest = [o["name"] for o in bad if o["name"] not in conf_ok]
        parts = []
        if ok:
            parts.append(f"Gaussian error bars hold for {_names(ok)}")
        if conf_ok:
            parts.append(f"for {_names(conf_ok)} use the conformal intervals")
        if rest:
            parts.append(f"the error bars for {_names(rest)} are too narrow")
        txt = "; ".join(parts)
        h["sec-3"] = txt[0].upper() + txt[1:]

    with_mf = [o for o in outs if o["status"]["mf"] is not None]
    if with_mf:
        helped = [o["name"] for o in with_mf if o["status"]["mf"] == "good"]
        none = [o["name"] for o in with_mf if o["status"]["mf"] == "neutral"]
        hurt = [o["name"] for o in with_mf if o["status"]["mf"] == "warn"]
        parts = []
        if helped:
            parts.append(f"Yes for {_names(helped)}")
        if none:
            parts.append(f"no clear gain for {_names(none)}")
        if hurt:
            parts.append(f"it hurts {_names(hurt)}")
        txt = "; ".join(parts)
        h["sec-4"] = txt[0].upper() + txt[1:]
    elif d.get("levels"):
        h["sec-4"] = "How informative each fidelity level is"

    shap = d.get("shap")
    if shap:
        top: Dict[str, List[str]] = {}
        for o in outs:
            row = shap["shares"].get(o["name"])
            if row:
                f = max(row, key=row.get)
                top.setdefault(f, []).append(o["name"])
        parts = [f"{f} matters most for {_names(v)}" for f, v in top.items()]
        if len(outs) == 1 and parts:
            parts = [f"{next(iter(top))} matters most"]
        txt = "; ".join(parts)
        h["sec-5"] = txt[0].upper() + txt[1:] if txt else "What drives the outputs"

    cv = d.get("cv")
    if cv and cv.get("rows"):
        okrows = [r for r in cv["rows"] if r["mean"] is not None]
        best = next((r for r in okrows if r["name"] == cv["best"]), okrows[0] if okrows else None)
        if best is not None:
            lim = best["mean"] + (best.get("sd") or 0)
            tied = [r for r in okrows if r["mean"] - (r.get("sd") or 0) <= lim]
            h["sec-7"] = (f"{best['name']} scored best, but the top {len(tied)} are within fold noise"
                          if len(tied) > 1 else f"{best['name']} scored best by a clear margin")
            cv["tied"] = [r["name"] for r in tied]
    return h


def _heat_cells(d: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    shap = d.get("shap")
    if not shap:
        return None
    outs = [o["name"] for o in d["outputs"] if o["name"] in shap["shares"]]
    hi = max((shap["shares"][o][f] for o in outs for f in shap["features"]), default=0) or 1
    rows = []
    for f in shap["features"]:
        cells = []
        for o in outs:
            v = shap["shares"][o].get(f, 0.0)
            k = min(6, int(v / hi * 6.999))
            cells.append({"output": o, "value": v, "step": k})
        rows.append({"feature": f, "cells": cells})
    return rows


def render_report(data: ReportData) -> str:
    """The v2 report for ``data`` as one HTML string."""
    env = Environment(
        loader=FileSystemLoader(str(_HERE / "templates")),
        autoescape=select_autoescape(default=True, default_for_string=True),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters.update(num=_num, pct=_pct, spct=_spct, names=_names)
    d = data.to_dict(include_images=True)
    headlines = _headlines(d)
    n = d["run"]["n_test_points"]
    th = d["thresholds"]
    band = coverage_band(th["coverage_target"], n, th["z"])
    blob_d = data.to_dict(include_images=False)
    blob_d["cv"] = d.get("cv")  # carries the "tied" list computed above
    blob = json.dumps(blob_d, allow_nan=False, ensure_ascii=False)
    has_mf = any(o["status"]["mf"] is not None for o in d["outputs"])
    ctx = dict(
        d=d,
        run=d["run"],
        outputs=d["outputs"],
        th=th,
        band=band,
        headlines=headlines,
        heat=_heat_cells(d),
        has_mf=has_mf,
        has_conformal=any(o.get("conformal") for o in d["outputs"]),
        ready=READY_LABEL,
        skipped_by=_group(d["skipped"], "section"),
        figures_by=_group(d["figures"], "section"),
        css=Markup((_HERE / "static" / "report_v2.css").read_text(encoding="utf-8")),
        js=Markup((_HERE / "static" / "report_v2.js").read_text(encoding="utf-8")),
        blob=Markup(json_for_script(blob)),
    )
    return env.get_template("report_v2.html.j2").render(**ctx)


def _group(items: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        out.setdefault(it.get(key) or "appendix", []).append(it)
    return out
