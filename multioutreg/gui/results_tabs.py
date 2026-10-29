# Copyright (c) 2026 takotime808

"""Results of a grid-search run, shown in tabs on the main page.

Everything here draws from the ``gs_results`` dict the page keeps in
``st.session_state``: the arrays saved after the run, the report data derived
from them (:func:`multioutreg.report.build.build_report_data`, the same numbers
the HTML report shows) and the SHAP / PDP data computed once in a background
job (:func:`compute_explanations`). Switching tabs or rerunning the page never
refits a model.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

TABS = ["Summary", "Predictions", "Uncertainty", "Explainability", "Report"]
# The report's output palette, so an output has the same color in the app and the report.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#d1477a", "#1a97ad", "#a3742c", "#6b7a1f"]
STATUS_ICON = {"good": "✅", "neutral": "✅", "warn": "⚠️", "crit": "❌", None: "–"}
MAX_COLS = 3


def _color(i: int) -> str:
    return COLORS[i % len(COLORS)]


def _grid(n: int, titles: List[str], **kw) -> go.Figure:
    cols = min(n, MAX_COLS)
    rows = -(-n // cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        horizontal_spacing=0.08, vertical_spacing=0.14, **kw)
    fig.update_layout(height=340 * rows, showlegend=False, margin=dict(t=50, b=40, l=50, r=20))
    return fig


def _rc(i: int, n: int):
    cols = min(n, MAX_COLS)
    return i // cols + 1, i % cols + 1


def compute_explanations(model: Any, X: np.ndarray, output_names: List[str],
                         feature_names: List[str]) -> Dict[str, Any]:
    """SHAP shares and partial dependence of the top three inputs, for the
    Explainability tab. Runs once per run (in a background worker); each part
    that fails is replaced by its error message."""
    from multioutreg.figures.pdp_plots import pdp_grid_data
    from multioutreg.figures.shap_multioutput import shap_importance_table

    out: Dict[str, Any] = {"shap": None, "pdp": None, "errors": []}
    X = np.asarray(X, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            table = shap_importance_table(model, X, output_names, feature_names)
            out["shap"] = {o: {f: v / (sum(row.values()) or 1.0) for f, v in row.items()}
                           for o, row in table.items()}
        except Exception as exc:
            out["errors"].append(f"SHAP: {type(exc).__name__}: {exc}")
        try:
            if out["shap"]:
                mean = {f: np.mean([out["shap"][o][f] for o in out["shap"]]) for f in feature_names}
                top = sorted(feature_names, key=lambda f: -mean[f])[:3]
            else:
                top = list(feature_names[:3])
            out["pdp"] = pdp_grid_data(model, X, output_names, feature_names,
                                       [feature_names.index(f) for f in top])
        except Exception as exc:
            out["errors"].append(f"Partial dependence: {type(exc).__name__}: {exc}")
    return out


# ---------------------------------------------------------------- Summary

def model_table(best_combo: List[dict], output_cols: List[str]) -> pd.DataFrame:
    """The chosen model per output as a readable table (one row per output)."""
    if len(best_combo) == 1 and len(output_cols) > 1:  # one joint model (e.g. MoE)
        best_combo = best_combo * len(output_cols)
    rows = []
    for name, combo in zip(output_cols, best_combo):
        params = {k: v for k, v in combo.items() if k != "model"}
        rows.append({
            "Output": name,
            "Model": combo.get("model", "?"),
            "Settings": ", ".join(f"{k}={v}" for k, v in params.items()) or "defaults",
        })
    return pd.DataFrame(rows).set_index("Output")


def _summary(res: Dict[str, Any]) -> None:
    data = res.get("report_data")
    if data:
        v = data["verdict"]
        getattr(st, {"good": "success", "warn": "warning", "crit": "error"}.get(v["status"], "info"))(
            f"**{v['headline']}** {v.get('why', '')}")
    outs = data["outputs"] if data else []
    if outs:
        cols = st.columns(min(len(outs), 4))
        for i, o in enumerate(outs):
            with cols[i % len(cols)]:
                st.markdown(f"**{STATUS_ICON[o['status']['overall']]} {o['name']}**")
                st.metric("Test R²", f"{o['r2']:.3f}" if o["r2"] is not None else "—",
                          help="Target ≥ 0.90 for a pass")
                st.metric("RMSE", f"{o['rmse']:.4g}" if o["rmse"] is not None else "—",
                          help=f"{100 * o['rmse'] / o['range']:.1f}% of the test range"
                          if o["rmse"] is not None else None)

    st.write("### Chosen model per output")
    st.dataframe(model_table(res["best_combo"], res["output_cols"]), width="stretch")

    st.write("### Metrics Table")
    st.dataframe(res["metrics"])

    cv = res.get("cv_table") or []
    ok = [r for r in cv if r.get("CV RMSE") is not None]
    if ok:
        st.write("### Model selection")
        st.caption("Cross-validated RMSE of every candidate (averaged over outputs), with ±1 "
                   "std across folds. Candidates whose bars overlap the best are statistically tied.")
        ok = sorted(ok, key=lambda r: r["CV RMSE"])[:20]
        names = [r["Model"] for r in ok][::-1]
        fig = go.Figure(go.Scatter(
            x=[r["CV RMSE"] for r in ok][::-1], y=names, mode="markers",
            error_x=dict(type="data", array=[r.get("CV RMSE std") or 0 for r in ok][::-1]),
            marker=dict(size=9, color=["#2b5d8c" if r.get("Selected for") else "#7d8692"
                                       for r in ok][::-1]),
            hovertemplate="%{y}<br>CV RMSE %{x:.4g}<extra></extra>"))
        fig.update_layout(height=60 + 26 * len(ok), xaxis_title="CV RMSE (lower is better)",
                          margin=dict(t=10, b=40, l=10, r=10))
        st.plotly_chart(fig, width="stretch")

    if res.get("pca_variance_plot") is not None:
        st.write("### PCA on the inputs")
        st.markdown(res["kaiser_rule_suggestion"])
        st.image(f"data:image/png;base64,{res['pca_variance_plot']}", caption="Scree Plot",
                 width="stretch")


# ---------------------------------------------------------------- Predictions

def _intervals(o: Dict[str, Any]):
    P = o["points"]
    if P.get("std") is not None:
        s = 1.959964 * np.asarray(P["std"])
        return np.asarray(P["pred"]) - s, np.asarray(P["pred"]) + s, "95% Gaussian interval"
    if P.get("lower") is not None:
        return np.asarray(P["lower"]), np.asarray(P["upper"]), "conformal interval"
    return None


def parity_figure(outs: List[Dict[str, Any]]) -> go.Figure:
    fig = _grid(len(outs), [f"{o['name']}  (R² {o['r2']:.3f})" for o in outs])
    for i, o in enumerate(outs):
        r, c = _rc(i, len(outs))
        y, p = np.asarray(o["points"]["y"]), np.asarray(o["points"]["pred"])
        iv = _intervals(o)
        err = dict(type="data", symmetric=False, array=iv[1] - p, arrayminus=p - iv[0],
                   thickness=1, width=0, color="rgba(125,134,146,0.5)") if iv else None
        fig.add_trace(go.Scatter(x=y, y=p, mode="markers", error_y=err,
                                 marker=dict(color=_color(o["idx"]), size=6),
                                 hovertemplate="observed %{x:.4g}<br>predicted %{y:.4g}<extra></extra>"),
                      row=r, col=c)
        lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 line=dict(dash="dash", color="#7d8692"), hoverinfo="skip"),
                      row=r, col=c)
        fig.update_xaxes(title_text="observed", row=r, col=c)
        fig.update_yaxes(title_text="predicted", row=r, col=c)
    return fig


def residuals_figure(outs: List[Dict[str, Any]]) -> go.Figure:
    fig = _grid(len(outs), [o["name"] for o in outs])
    for i, o in enumerate(outs):
        r, c = _rc(i, len(outs))
        y, p = np.asarray(o["points"]["y"]), np.asarray(o["points"]["pred"])
        fig.add_trace(go.Scatter(x=p, y=y - p, mode="markers",
                                 marker=dict(color=_color(o["idx"]), size=6),
                                 hovertemplate="predicted %{x:.4g}<br>residual %{y:.4g}<extra></extra>"),
                      row=r, col=c)
        fig.add_hline(y=0, line=dict(color="#7d8692"), row=r, col=c)
        t = o.get("trend")
        if t:
            xs = np.array([p.min(), p.max()])
            fig.add_trace(go.Scatter(
                x=xs, y=t["intercept"] + t["slope"] * xs, mode="lines",
                line=dict(color="#d03b3b" if t["significant"] else "#7d8692",
                          dash=None if t["significant"] else "dash"),
                hovertemplate=f"slope {t['slope']:.3g}, p = {t['p']:.2g}<extra></extra>"),
                row=r, col=c)
        fig.update_xaxes(title_text="predicted", row=r, col=c)
        fig.update_yaxes(title_text="observed − predicted", row=r, col=c)
    return fig


def histogram_figure(outs: List[Dict[str, Any]]) -> go.Figure:
    fig = _grid(len(outs), [o["name"] for o in outs])
    for i, o in enumerate(outs):
        r, c = _rc(i, len(outs))
        y, p = np.asarray(o["points"]["y"]), np.asarray(o["points"]["pred"])
        fig.add_trace(go.Histogram(x=y - p, marker_color=_color(o["idx"])), row=r, col=c)
        fig.add_vline(x=0, line=dict(dash="dot", color="#7d8692"), row=r, col=c)
        fig.update_xaxes(title_text="residual", row=r, col=c)
    return fig


def _predictions(res: Dict[str, Any]) -> None:
    outs = (res.get("report_data") or {}).get("outputs") or []
    if not outs:
        st.info("No predictions saved for this run.")
        return
    st.write("### Predicted vs observed")
    st.caption("Test-set predictions with their intervals. Points on the dashed diagonal "
               "are exact.")
    st.plotly_chart(parity_figure(outs), width="stretch")
    st.write("### Residuals vs predicted")
    st.caption("A flat cloud around zero is good. The line is a linear fit: red when its "
               "slope is significant (p < 0.05), meaning a systematic bias.")
    st.plotly_chart(residuals_figure(outs), width="stretch")
    st.write("### Residual distribution")
    st.plotly_chart(histogram_figure(outs), width="stretch")


# ---------------------------------------------------------------- Uncertainty

def ordered_intervals_figure(outs: List[Dict[str, Any]]) -> go.Figure:
    fig = _grid(len(outs), [o["name"] for o in outs])
    for i, o in enumerate(outs):
        r, c = _rc(i, len(outs))
        iv = _intervals(o)
        if iv is None:
            continue
        y = np.asarray(o["points"]["y"])
        order = np.argsort(y)
        lo, hi, y = iv[0][order], iv[1][order], y[order]
        k = np.arange(len(y))
        inside = (y >= lo) & (y <= hi)
        fig.add_trace(go.Scatter(
            x=np.r_[k, k[::-1]], y=np.r_[hi, lo[::-1]], fill="toself", mode="lines",
            line=dict(width=0), fillcolor="rgba(43,93,140,0.18)", hoverinfo="skip"),
            row=r, col=c)
        fig.add_trace(go.Scatter(
            x=k, y=y, mode="markers",
            marker=dict(size=6, color=np.where(inside, "#4c5561", "#d03b3b")),
            hovertemplate="observed %{y:.4g}<extra></extra>"), row=r, col=c)
        fig.layout.annotations[i].text += f"  ({int((~inside).sum())} of {len(y)} outside)"
        fig.update_xaxes(title_text="test points, sorted by observed value", row=r, col=c)
    return fig


def calibration_figure(outs: List[Dict[str, Any]], n_test: int, z: float = 1.96) -> go.Figure:
    fig = _grid(len(outs), [o["name"] for o in outs])
    ps = np.linspace(0, 1, 41)
    band = z * np.sqrt(ps * (1 - ps) / max(n_test, 1))
    for i, o in enumerate(outs):
        r, c = _rc(i, len(outs))
        fig.add_trace(go.Scatter(x=np.r_[ps, ps[::-1]],
                                 y=np.r_[np.minimum(1, ps + band), np.maximum(0, ps - band)[::-1]],
                                 fill="toself", mode="lines", line=dict(width=0),
                                 fillcolor="rgba(43,93,140,0.12)", hoverinfo="skip"), row=r, col=c)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#c3c8cf"),
                                 hoverinfo="skip"), row=r, col=c)
        if o.get("gauss_curve"):
            fig.add_trace(go.Scatter(x=o["levels"], y=o["gauss_curve"], mode="lines+markers",
                                     name="Gaussian", line=dict(dash="dash", color="#4c5561"),
                                     hovertemplate="nominal %{x:.0%}<br>Gaussian %{y:.0%}<extra></extra>"),
                          row=r, col=c)
        if o.get("conf_curve"):
            fig.add_trace(go.Scatter(x=o["levels"], y=o["conf_curve"], mode="lines+markers",
                                     name="Conformal", line=dict(color=_color(o["idx"])),
                                     hovertemplate="nominal %{x:.0%}<br>conformal %{y:.0%}<extra></extra>"),
                          row=r, col=c)
        fig.update_xaxes(title_text="nominal coverage", tickformat=".0%", range=[0, 1], row=r, col=c)
        fig.update_yaxes(title_text="observed coverage", tickformat=".0%", range=[0, 1.02], row=r, col=c)
    return fig


def _uncertainty(res: Dict[str, Any]) -> None:
    data = res.get("report_data") or {}
    outs = data.get("outputs") or []
    if outs:
        st.write("### Can the error bars be trusted?")
        st.caption("Observed against nominal coverage. Inside the shaded band is consistent with "
                   "good calibration for this many test points; below it the bars are too narrow. "
                   "Dashed: Gaussian (prediction ± z·std); solid: conformal, when computed.")
        st.plotly_chart(calibration_figure(outs, data["run"]["n_test_points"]), width="stretch")
        st.write("### Intervals, ordered by observed value")
        st.caption("Red points fall outside their interval; about 1 in 20 is expected at 95%.")
        st.plotly_chart(ordered_intervals_figure(outs), width="stretch")

    st.write("### Uncertainty Calibration (Gaussian std, test set)")
    st.dataframe(res["uncertainty_metrics"])
    if res.get("conformal_summary") is not None:
        st.write(f"### Conformal Prediction Summary (alpha={res['conformal_alpha']})")
        st.dataframe(res["conformal_summary"])


# ---------------------------------------------------------------- Explainability

def shap_figure(shares: Dict[str, Dict[str, float]]) -> go.Figure:
    outs = list(shares)
    feats = sorted(shares[outs[0]], key=lambda f: -np.mean([shares[o][f] for o in outs]))[:12]
    z = np.array([[shares[o][f] for o in outs] for f in feats])
    fig = go.Figure(go.Heatmap(
        z=z, x=outs, y=feats, colorscale="Blues", zmin=0, text=np.round(100 * z).astype(int),
        texttemplate="%{text}%", hovertemplate="%{y} → %{x}: %{z:.1%} of mean |SHAP|<extra></extra>",
        showscale=False))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=80 + 34 * len(feats), margin=dict(t=10, b=40, l=10, r=10))
    return fig


def pdp_figure(pdp: Dict[str, Any], output_names: List[str]) -> go.Figure:
    feats = pdp["features"]
    outs = [o for o in output_names if o in pdp["curves"]]
    fig = make_subplots(rows=len(outs), cols=len(feats), shared_xaxes="columns",
                        shared_yaxes="rows", horizontal_spacing=0.04, vertical_spacing=0.06,
                        column_titles=feats, row_titles=outs)
    for ri, o in enumerate(outs):
        for ci, f in enumerate(feats):
            xs, cur = pdp["grid"][f], pdp["curves"][o][f]
            fig.add_trace(go.Scatter(x=list(xs) + list(xs)[::-1], y=list(cur["p90"]) + list(cur["p10"])[::-1],
                                     fill="toself", mode="lines", line=dict(width=0),
                                     fillcolor="rgba(43,93,140,0.15)", hoverinfo="skip"),
                          row=ri + 1, col=ci + 1)
            fig.add_trace(go.Scatter(x=xs, y=cur["mean"], mode="lines",
                                     line=dict(color=_color(ri), width=2),
                                     hovertemplate=f"{f}=%{{x:.4g}}<br>{o}=%{{y:.4g}}<extra></extra>"),
                          row=ri + 1, col=ci + 1)
    fig.update_layout(height=60 + 220 * len(outs), showlegend=False,
                      margin=dict(t=40, b=30, l=40, r=40))
    return fig


def _explainability(res: Dict[str, Any]) -> None:
    exp = res.get("explain")
    if exp is None:
        st.info("SHAP values and partial dependence are being computed in the background; "
                "they appear here when ready (no model is refitted).")
        return
    if exp.get("shap"):
        st.write("### What drives each output")
        st.caption("Share of mean |SHAP| per output: each column sums to 100%.")
        st.plotly_chart(shap_figure(exp["shap"]), width="stretch")
    if exp.get("pdp"):
        st.write("### How the top inputs move each output")
        st.caption("Partial dependence (line) with the 10th–90th percentile of individual "
                   "curves (band). Columns share an x axis, rows a y axis.")
        st.plotly_chart(pdp_figure(exp["pdp"], res["output_cols"]), width="stretch")
    for err in exp.get("errors", []):
        st.caption(f"Skipped: {err}")


# ---------------------------------------------------------------- Report

def _report(res: Dict[str, Any], slot) -> None:
    """Fill the Report tab once the background report job has finished."""
    from multioutreg.gui.report_controls import report_downloads_all

    reports = res.get("html")
    if reports:
        box = slot.container()
        report_downloads_all(reports, "model_report.html", "gs_report", container=box)
        html = reports.get("v2") or next(iter(reports.values()))
        with box:
            st.caption("Preview of the report. The downloaded file is the same page and opens "
                       "offline in any browser.")
            components.html(html, height=900, scrolling=True)
    elif res.get("report_error"):
        slot.error(res["report_error"])


def render_results(res: Dict[str, Any]):
    """Draw the result tabs; returns the Report tab's slot, filled later by
    :func:`fill_report_tab` when the background report job finishes."""
    tabs = st.tabs(TABS)
    with tabs[0]:
        _summary(res)
    with tabs[1]:
        _predictions(res)
    with tabs[2]:
        _uncertainty(res)
    with tabs[3]:
        explain_slot = st.empty()
        with explain_slot.container():
            _explainability(res)
    with tabs[4]:
        report_slot = st.empty()
    return {"report": report_slot, "explain": explain_slot, "tabs": tabs}


def _report_export(res: Dict[str, Any], slot) -> None:
    """The report downloads again in the export section, where people look for them."""
    from multioutreg.gui.report_controls import report_downloads_all

    reports = res.get("html")
    if reports:
        report_downloads_all(reports, "model_report.html", "gs_export_report",
                             container=slot.container())
    elif res.get("report_error"):
        slot.error(res["report_error"])


def show_report_pending(slots: Dict[str, Any], secs: int) -> None:
    """Say the report is still building, in the Report tab and the export section."""
    slots["report"].info(
        f"Building the HTML report in the background ({secs}s)… "
        "The results above are final; the download appears here when it's ready.")
    if "export_report" in slots:
        slots["export_report"].info(
            f"Building the HTML report in the background ({secs}s)… Its download "
            "buttons appear here and in the **Report** tab when it's ready.")


def fill_report_tab(res: Dict[str, Any], slots: Dict[str, Any]) -> None:
    _report(res, slots["report"])
    if "export_report" in slots:
        _report_export(res, slots["export_report"])


def refresh_explain_tab(res: Dict[str, Any], slots: Dict[str, Any]) -> None:
    with slots["explain"].container():
        _explainability(res)
