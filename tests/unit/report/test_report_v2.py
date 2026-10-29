# Copyright (c) 2026 takotime808

"""Tests for the redesigned (v2) report: verdict rules, data, rendering."""

import json
import re

import numpy as np
import pytest

from multioutreg.report.build import build_report_data, data_hash
from multioutreg.report.data import ReportData
from multioutreg.report.render import (
    json_for_script,
    render_report,
    render_styles,
    report_json_from_html,
    resolve_style,
)
from multioutreg.report.verdict import (
    Thresholds,
    coverage_band,
    coverage_status,
    judge_output,
    worst,
)


def _fixture(n_out=3, n=60, seed=0, mf=True, names=None):
    rng = np.random.default_rng(seed)
    names = names or [f"y{j}" for j in range(n_out)]
    y = rng.normal(size=(n, n_out))
    pred = y + 0.2 * rng.normal(size=(n, n_out))
    std = np.full((n, n_out), 0.2)
    metrics = {
        nm: {"r2": 1 - np.mean((y[:, j] - pred[:, j]) ** 2) / np.var(y[:, j]),
             "rmse": float(np.sqrt(np.mean((y[:, j] - pred[:, j]) ** 2))),
             "mae": float(np.mean(np.abs(y[:, j] - pred[:, j])))}
        for j, nm in enumerate(names)}
    kw = dict(output_names=names, y_test=y, best_pred=pred, best_std=std, metrics=metrics,
              model_type="TestModel", n_train=100, n_test=n, explain=False,
              calibration_residuals=np.abs(0.2 * rng.normal(size=(40, n_out))),
              conformal_intervals=(pred - 0.4, pred + 0.4), conformal_alpha=0.1)
    if mf:
        kw.update(
            fidelity_levels=["lo", "hi"],
            level_summary=[{"level": lv, "output": nm, "n_train": 50, "n_test": 20, "mean": 0.0,
                            "std": 1.0, "r": 0.3 if lv == "lo" else None, "r2": 0.1, "error": None}
                           for lv in ("lo", "hi") for nm in names],
            baseline_comparison=[{"output": nm, "mf_rmse": metrics[nm]["rmse"],
                                  "hf_rmse": metrics[nm]["rmse"] * 1.3, "rmse_change_pct": -23.0,
                                  "mf_r2": metrics[nm]["r2"], "hf_r2": 0.8} for nm in names],
            baseline_label="GP",
            grid_search_results=[
                {"Model": "a", "CV MSE": 0.04, "CV RMSE": 0.2, "CV RMSE std": 0.05, "CV R²": 0.9, "Status": "OK"},
                {"Model": "b", "CV MSE": 0.0484, "CV RMSE": 0.22, "CV RMSE std": 0.05, "CV R²": 0.88, "Status": "OK"},
                {"Model": "c", "CV MSE": None, "CV RMSE": None, "CV RMSE std": None, "CV R²": None, "Status": "Error: x"},
            ],
            best_candidate_name="a",
        )
    return kw


# ---------------------------------------------------------------- verdict

def test_accuracy_threshold_boundaries():
    th = Thresholds()
    base = {"has_std": True, "gauss95": 0.95}
    assert judge_output({**base, "r2": 0.90}, 100, th)["accuracy"] == "good"
    assert judge_output({**base, "r2": 0.8999}, 100, th)["accuracy"] == "warn"
    assert judge_output({**base, "r2": 0.80}, 100, th)["accuracy"] == "warn"
    assert judge_output({**base, "r2": 0.7999}, 100, th)["accuracy"] == "crit"


def test_coverage_band_shrinks_with_n():
    assert coverage_band(0.95, 400) < coverage_band(0.95, 100) < coverage_band(0.95, 25)
    b = coverage_band(0.95, 100)
    assert coverage_status(0.95 - b, 0.95, 100) == "good"
    assert coverage_status(0.95 - b - 1e-6, 0.95, 100) == "warn"
    assert coverage_status(0.95 - 2 * b, 0.95, 100) == "warn"
    assert coverage_status(0.95 - 2 * b - 1e-6, 0.95, 100) == "crit"
    # The same coverage passes with fewer test points (wider noise band).
    assert coverage_status(0.86, 0.95, 20) == "good"
    assert coverage_status(0.86, 0.95, 500) == "crit"


def test_mf_thresholds_and_overall_is_worst():
    base = {"r2": 0.95, "has_std": True, "gauss95": 0.95}
    assert judge_output({**base, "mf_change_pct": -5.0}, 100)["mf"] == "good"
    assert judge_output({**base, "mf_change_pct": -4.9}, 100)["mf"] == "neutral"
    assert judge_output({**base, "mf_change_pct": 5.0}, 100)["mf"] == "warn"
    # neutral (no gain) does not lower the overall status
    assert judge_output({**base, "mf_change_pct": 0.0}, 100)["overall"] == "good"
    j = judge_output({"r2": 0.5, "has_std": True, "gauss95": 0.95, "mf_change_pct": -20}, 100)
    assert j["overall"] == "crit"
    assert worst("good", "neutral") == "good" and worst("good", "warn", None) == "warn"


def test_conformal_rescues_error_bars_as_caution():
    o = {"r2": 0.95, "has_std": True, "gauss95": 0.60,
         "conformal": {"nominal": 0.95, "coverage": 0.95}}
    j = judge_output(o, 50)
    assert j["gaussian"] == "crit" and j["conformal"] == "good" and j["bars"] == "warn"
    assert judge_output({"r2": 0.95, "has_std": False}, 50)["gaussian"] == "crit"


def test_verdict_findings_and_next_steps():
    kw = _fixture()
    kw["best_std"] = np.full_like(kw["best_std"], 0.02)  # far too narrow
    d = build_report_data(**kw)
    assert d.verdict["status"] in ("warn", "crit")
    assert "Not ready" in d.verdict["headline"] or "with care" in d.verdict["headline"]
    claims = " ".join(f["claim"] for f in d.findings)
    assert "too narrow" in claims
    assert any("conformal" in s["html"].lower() for s in d.next_steps)
    assert any("Consider dropping level 'lo'" in s["html"] for s in d.next_steps)


# ---------------------------------------------------------------- data

def test_report_data_json_round_trip_and_nan():
    kw = _fixture()
    kw["metrics"]["y0"]["extra"] = float("nan")
    d = build_report_data(**kw)
    text = d.to_json()
    assert "NaN" not in text
    back = ReportData.from_json(text)
    assert back.to_json() == text
    assert back.outputs[0]["metrics"]["extra"] is None


def test_downsampling_caps_points():
    d = build_report_data(**_fixture(n_out=1, n=3000, mf=False), max_points=500)
    pts = d.outputs[0]["points"]
    assert len(pts["y"]) == 500 and pts["downsampled"]
    assert d.outputs[0]["n"] == 3000  # statistics use every point


def test_data_hash_is_stable():
    a = np.arange(6.0).reshape(3, 2)
    assert data_hash(a) == data_hash(a.copy()) != data_hash(a + 1)
    assert len(data_hash(a)) == 7


def test_cv_rows_and_tied_candidates():
    d = build_report_data(**_fixture())
    assert d.cv["best"] == "a"
    assert [r["name"] for r in d.cv["rows"]] == ["a", "b", "c"]
    html = render_report(d)
    assert "Tied" in html and "Error: x" in html


# ---------------------------------------------------------------- render

def test_render_is_self_contained():
    html = render_report(build_report_data(**_fixture()))
    for attr in re.findall(r'(?:src|href)\s*=\s*"([^"]*)"', html):
        assert not attr.startswith(("http://", "https://", "//")), attr
    assert "@import" not in html
    assert "<link" not in html


def test_embedded_json_parses_and_matches():
    d = build_report_data(**_fixture())
    blob = report_json_from_html(render_report(d))
    parsed = json.loads(blob)
    assert parsed["run"]["model"] == "TestModel"
    assert len(parsed["outputs"]) == 3
    assert ReportData.from_dict(parsed).outputs[1]["name"] == "y1"


def test_script_breakout_is_escaped():
    evil = "</script><script>alert(1)</script>"
    kw = _fixture(n_out=1, names=[evil], mf=False)
    kw["description"] = "<b>bold</b>"
    html = render_report(build_report_data(**kw))
    assert "<script>alert(1)</script>" not in html
    assert "&lt;/script&gt;" in html and "&lt;b&gt;bold&lt;/b&gt;" in html
    assert json.loads(report_json_from_html(html))["outputs"][0]["name"] == evil
    assert json_for_script('{"a": "</script>&"}') == '{"a": "\\u003c/script\\u003e\\u0026"}'


def test_size_budget_six_outputs_500_points():
    html = render_report(build_report_data(**_fixture(n_out=6, n=500)))
    assert len(html.encode()) < 2 * 1024 * 1024


def test_single_fidelity_sections():
    html = render_report(build_report_data(**_fixture(mf=False)))
    assert "Single-fidelity run" in html
    assert "Per-candidate CV scores were not recorded" in html


def test_regret_rows_render_pass_fail():
    kw = _fixture(mf=False)
    kw.update(regret_metrics={"hv": 0.1, "scalar": 0.5, "eps": 0.2},
              regret_bounds={"hv": 0.2, "scalar": 0.4, "eps": None})
    html = render_report(build_report_data(**kw))
    assert "Multi-Objective Regret" in html
    assert '<span class="chip good">PASS</span>' in html
    assert '<span class="chip crit">FAIL</span>' in html


def test_resolve_style(monkeypatch):
    monkeypatch.delenv("MOR_TEMPLATE_PATH", raising=False)
    monkeypatch.delenv("MOR_REPORT_STYLE", raising=False)
    assert resolve_style() == "v2"
    assert resolve_style("classic") == "classic"
    assert resolve_style("v2", template_path="x.html") == "classic"
    monkeypatch.setenv("MOR_REPORT_STYLE", "classic")
    assert resolve_style() == "classic"
    monkeypatch.setenv("MOR_TEMPLATE_PATH", "t.html")
    assert resolve_style("v2") == "classic"
    with pytest.raises(ValueError):
        monkeypatch.delenv("MOR_TEMPLATE_PATH")
        resolve_style("fancy")


def test_render_styles(monkeypatch):
    monkeypatch.delenv("MOR_TEMPLATE_PATH", raising=False)
    calls = []

    def fake(report_style, **kw):
        calls.append((report_style, kw))
        return f"<{report_style}>"

    assert render_styles(fake, ("v2", "classic"), x=1) == {"v2": "<v2>", "classic": "<classic>"}
    assert calls == [("v2", {"x": 1}), ("classic", {"x": 1})]
    # A custom template forces classic, so "Both" renders it once.
    monkeypatch.setenv("MOR_TEMPLATE_PATH", "t.html")
    assert render_styles(fake, ("v2", "classic")) == {"classic": "<classic>"}


def test_explain_with_real_model():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor

    rng = np.random.default_rng(0)
    X = rng.uniform(size=(80, 4))
    Y = np.c_[3 * X[:, 0], X[:, 1] ** 2]
    model = MultiOutputRegressor(RandomForestRegressor(20, random_state=0)).fit(X[:60], Y[:60])
    pred = model.predict(X[60:])
    kw = _fixture(n_out=2, n=20, mf=False)
    kw.update(y_test=Y[60:], best_pred=pred, best_std=np.full_like(pred, 0.1),
              best_model=model, X_train=X[:60], input_names=["a", "b", "c", "d"], explain=True,
              conformal_intervals=None, calibration_residuals=None, conformal_alpha=None)
    d = build_report_data(**kw)
    assert d.shap is not None and d.shap["features"][0] in ("a", "b")
    assert abs(sum(d.shap["shares"]["y0"].values()) - 1) < 1e-9
    assert d.pdp is not None and len(d.pdp["features"]) == 3
    assert len(d.pdp["curves"]["y1"]["b"]["mean"]) == len(d.pdp["grid"]["b"])


def test_select_per_target_records_fold_spread():
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor

    from multioutreg.gui.grid_search_core import select_per_target

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = np.c_[X @ [1.0, 2.0, 3.0], np.sin(X[:, 0])]
    r = select_per_target([("linear", LinearRegression, {}),
                           ("dt", DecisionTreeRegressor, {"max_depth": [2, 4]})], X, y)
    assert len(r.cv_table) == 3
    assert all(row["CV RMSE std"] is not None and row["CV RMSE std"] >= 0 for row in r.cv_table)
    assert r.cv_table == sorted(r.cv_table, key=lambda row: row["CV MSE"])
    selected = sorted(j for row in r.cv_table for j in row["Selected for"])
    assert selected == [0, 1]
