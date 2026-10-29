# Copyright (c) 2026 takotime808

"""Turn model-evaluation results into :class:`~multioutreg.report.data.ReportData`.

:func:`build_report_data` takes the arguments the report callers already
have (test arrays, metrics, conformal intervals, level and baseline rows,
grid-search rows, ...), computes the derived series the v2 report shows and
runs the verdict rules in :mod:`multioutreg.report.verdict`.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import re
import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

from multioutreg.report.data import ReportData, downsample_index
from multioutreg.report.verdict import (
    Thresholds,
    build_findings,
    build_next_steps,
    build_verdict,
    judge_output,
)

#: Nominal levels for the calibration curves.
NOMINAL_LEVELS = [round(0.05 * k, 2) for k in range(1, 20)] + [0.99]


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("multioutreg")
    except Exception:
        return "unknown"


def data_hash(*arrays: Optional[np.ndarray]) -> str:
    """First 7 hex characters of a SHA-1 over the arrays' bytes."""
    h = hashlib.sha1()
    for a in arrays:
        if a is None:
            continue
        a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:7]


def _quantile(scores: np.ndarray, p: float) -> float:
    """Split-conformal quantile at nominal coverage ``p`` (inf when n is too small)."""
    n = len(scores)
    k = int(np.ceil((n + 1) * p))
    if k > n or n == 0:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def _as_2d(a: Optional[np.ndarray], n: int) -> Optional[np.ndarray]:
    if a is None:
        return None
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(n, -1)


def _trend(pred: np.ndarray, res: np.ndarray) -> Optional[Dict[str, Any]]:
    """OLS of residual on prediction: slope, intercept, t, two-sided p."""
    if len(pred) < 4 or np.ptp(pred) == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = stats.linregress(pred, res)
    if not np.isfinite(fit.slope):
        return None
    t = fit.slope / fit.stderr if fit.stderr and fit.stderr > 0 else 0.0
    p = float(fit.pvalue) if np.isfinite(fit.pvalue) else 1.0
    return {"slope": float(fit.slope), "intercept": float(fit.intercept),
            "t": float(t), "p": p, "significant": bool(p < 0.05)}


def _output_block(
    j: int, name: str, y: np.ndarray, pred: np.ndarray, std: Optional[np.ndarray],
    metrics: Dict[str, float], cal_res: Optional[np.ndarray],
    lower: Optional[np.ndarray], upper: Optional[np.ndarray],
    conformal_alpha: Optional[float], baseline: Optional[dict], cap: int,
) -> Dict[str, Any]:
    res = y - pred
    rng = float(np.ptp(y)) or 1.0
    has_std = std is not None and bool(np.any(std > 0))
    s = std if std is not None else np.zeros_like(y)
    z95 = float(stats.norm.ppf(0.975))

    gauss = [float(np.mean(np.abs(res) <= stats.norm.ppf((1 + p) / 2) * s)) for p in NOMINAL_LEVELS]
    i95 = NOMINAL_LEVELS.index(0.95)
    o: Dict[str, Any] = {
        "idx": j,
        "name": name,
        "n": int(len(y)),
        "r2": _f(metrics.get("r2")),
        "rmse": _f(metrics.get("rmse")),
        "mae": _f(metrics.get("mae")),
        "metrics": {k: _f(v) for k, v in metrics.items()},
        "range": rng,
        "has_std": has_std,
        "gauss95": gauss[i95] if has_std else 0.0,
        "outside95": int(np.sum(np.abs(res) > z95 * s)),
        "width_gauss_pct": 100.0 * float(np.mean(2 * z95 * s)) / rng if has_std else None,
        "levels": NOMINAL_LEVELS,
        "gauss_curve": gauss if has_std else None,
        "conf_curve": None,
        "conformal": None,
        "width_conf_pct": None,
        "trend": _trend(pred, res),
    }
    if cal_res is not None:
        conf = []
        for p in NOMINAL_LEVELS:
            q = _quantile(cal_res, p)
            conf.append(float(np.mean(np.abs(res) <= q)))
        o["conf_curve"] = conf
        q95 = _quantile(cal_res, 0.95)
        o["conformal"] = {"nominal": 0.95, "coverage": conf[i95], "n_cal": int(len(cal_res))}
        o["width_conf_pct"] = 100.0 * 2 * q95 / rng if np.isfinite(q95) else None
    elif lower is not None and upper is not None and conformal_alpha is not None:
        o["conformal"] = {"nominal": 1 - conformal_alpha,
                          "coverage": float(np.mean((y >= lower) & (y <= upper)))}
        w = float(np.mean(upper - lower))
        o["width_conf_pct"] = 100.0 * w / rng if np.isfinite(w) else None
    if baseline is not None:
        o["hf_rmse"] = _f(baseline.get("hf_rmse"))
        o["hf_r2"] = _f(baseline.get("hf_r2"))
        o["mf_change_pct"] = _f(baseline.get("rmse_change_pct"))
    else:
        o["hf_rmse"] = o["hf_r2"] = o["mf_change_pct"] = None

    idx = downsample_index(len(y), cap, seed=j)
    o["points"] = {
        "y": y[idx], "pred": pred[idx], "std": s[idx] if has_std else None,
        "lower": lower[idx] if lower is not None else None,
        "upper": upper[idx] if upper is not None else None,
        "downsampled": bool(len(idx) < len(y)),
    }
    return o


def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _levels(level_summary: Optional[List[dict]], fidelity_levels: Sequence[str]) -> List[dict]:
    if not level_summary:
        return []
    order: List[str] = []
    by: Dict[str, dict] = {}
    for r in level_summary:
        lv = str(r["level"])
        if lv not in by:
            order.append(lv)
            by[lv] = {"level": lv, "n_train": r.get("n_train"), "n_test": r.get("n_test"),
                      "outputs": {}}
        by[lv]["outputs"][r["output"]] = {
            "r": _f(r.get("r")), "r2": _f(r.get("r2")), "mean": _f(r.get("mean")),
            "std": _f(r.get("std")), "error": r.get("error")}
    highest = str(fidelity_levels[-1]) if fidelity_levels else order[-1]
    for lv in order:
        by[lv]["is_highest"] = lv == highest
    return [by[lv] for lv in order]


def _cv_from_rows(rows: List[dict], best_name: Optional[str],
                  output_names: Sequence[str]) -> Dict[str, Any]:
    out = []
    for r in rows:
        mean = _f(r.get("CV RMSE"))
        sel = [output_names[k] for k in r.get("Selected for", []) if k < len(output_names)]
        out.append({"name": str(r["Model"]), "mean": mean, "sd": _f(r.get("CV RMSE std")),
                    "r2": _f(r.get("CV R²")), "status": str(r.get("Status", "OK")),
                    "selected_for": sel})
    ok = sorted((r for r in out if r["mean"] is not None), key=lambda r: r["mean"])
    failed = [r for r in out if r["mean"] is None]
    if best_name is None:
        chosen = [r["name"] for r in ok if r["selected_for"]]
        best_name = chosen[0] if chosen else (ok[0]["name"] if ok else None)
    return {"metric": "CV RMSE", "spread": "±1 std across folds", "rows": ok + failed,
            "best": best_name,
            "per_output": any(r["selected_for"] for r in out)}


def _cv_from_selection(scores: List[Dict[str, Dict[str, float]]],
                       output_names: Sequence[str]) -> Dict[str, Any]:
    """Rows from ``AutoDetectMultiOutputRegressor.selection_scores_`` (neg MSE)."""
    names: List[str] = []
    for per_out in scores:
        for n in per_out:
            if n not in names:
                names.append(n)
    rows = []
    for n in names:
        mses = [-per_out[n]["mean"] for per_out in scores if n in per_out]
        ses = [per_out[n]["se"] for per_out in scores if n in per_out]
        if not mses or any(m < 0 for m in mses):
            continue
        mse = float(np.mean(mses))
        rmse = float(np.sqrt(mse))
        # Delta method: se(sqrt(m)) ≈ se(m) / (2 sqrt(m)).
        sd = float(np.mean(ses)) / (2 * rmse) if rmse > 0 else 0.0
        rows.append({"name": n, "mean": rmse, "sd": sd, "r2": None, "status": "OK",
                     "selected_for": []})
    rows.sort(key=lambda r: r["mean"])
    return {"metric": "CV RMSE", "spread": "±1 corrected standard error",
            "rows": rows, "best": rows[0]["name"] if rows else None, "per_output": True}


def _regret(metrics: Optional[dict], bounds: Optional[dict], settings: Optional[dict]):
    if not metrics:
        return None
    bounds = bounds or {}
    rows = []
    for key, label in [("hv", "Hypervolume regret"), ("scalar", "Scalarized regret (mean)"),
                       ("eps", "ε-regret")]:
        v, b = _f(metrics.get(key)), _f(bounds.get(key))
        status = None if v is None or b is None else ("good" if v <= b else "crit")
        rows.append({"key": key, "label": label, "value": v, "bound": b, "status": status})
    return {"rows": rows, "settings": settings or {}}


def _top_features(shap_shares: Optional[Dict[str, Dict[str, float]]],
                  feature_names: Sequence[str], k: int = 3) -> List[str]:
    if shap_shares:
        mean_share = {f: float(np.mean([shap_shares[o].get(f, 0.0) for o in shap_shares]))
                      for f in feature_names}
        return sorted(feature_names, key=lambda f: -mean_share[f])[:k]
    return list(feature_names[:k])


def _explain(model: Any, X: np.ndarray, output_names: List[str], feature_names: List[str],
             skipped: List[dict], max_heat: int = 12):
    """SHAP shares and PDP grids; failures become ``skipped`` lines."""
    from multioutreg.figures.pdp_plots import pdp_grid_data
    from multioutreg.figures.shap_multioutput import shap_importance_table

    shap_block = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            table = shap_importance_table(model, X, output_names, feature_names)
        shares = {}
        for o, row in table.items():
            tot = sum(row.values())
            shares[o] = {f: (v / tot if tot > 0 else 0.0) for f, v in row.items()}
        mean_share = {f: float(np.mean([shares[o][f] for o in shares])) for f in feature_names}
        order = sorted(feature_names, key=lambda f: -mean_share[f])
        shap_block = {"features": order[:max_heat], "n_features": len(order),
                      "shares": shares, "raw": table,
                      "missing": [o for o in output_names if o not in table]}
    except Exception as exc:
        skipped.append({"section": "sec-5", "reason": f"SHAP values could not be computed "
                        f"for this model ({type(exc).__name__})."})

    pdp_block = None
    if len(X) >= 2:
        try:
            if shap_block:
                top = _top_features(shap_block["shares"], feature_names)
            else:
                # Without SHAP, rank by how far each input moves the predictions.
                cand = list(range(min(len(feature_names), 10)))
                probe = pdp_grid_data(model, X, output_names, feature_names, cand,
                                      grid_resolution=8, max_rows=60)
                spread = {f: np.mean([np.ptp(probe["curves"][o][f]["mean"]) /
                                      (np.ptp(np.concatenate([probe["curves"][o][g]["mean"]
                                                              for g in probe["features"]])) or 1)
                                      for o in output_names])
                          for f in probe["features"]}
                top = sorted(spread, key=lambda f: -spread[f])[:3]
            idx = [feature_names.index(f) for f in top]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pdp_block = pdp_grid_data(model, X, output_names, feature_names, idx)
        except Exception as exc:
            skipped.append({"section": "sec-5", "reason": f"Partial dependence could not be "
                            f"computed for this model ({type(exc).__name__})."})
    return shap_block, pdp_block


def _strip_md(text: Optional[str]) -> Optional[str]:
    return re.sub(r"\*\*(.+?)\*\*", r"\1", text) if text else text


def build_report_data(
    *,
    output_names: List[str],
    y_test: np.ndarray,
    best_pred: np.ndarray,
    best_std: Optional[np.ndarray],
    metrics: Dict[str, Dict[str, float]],
    title: str = "Surrogate Model Report",
    model_type: str = "",
    description: str = "",
    notes: str = "",
    uncertainty_metrics: Optional[Dict[str, float]] = None,
    n_train: Any = None,
    n_test: Any = None,
    cross_validation: str = "",
    seed: Any = None,
    fidelity_levels: Optional[Sequence[str]] = None,
    input_names: Optional[List[str]] = None,
    X_train: Optional[np.ndarray] = None,
    best_model: Any = None,
    conformal_intervals: Optional[tuple] = None,
    conformal_alpha: Optional[float] = None,
    calibration_residuals: Optional[np.ndarray] = None,
    imputation_summary: Optional[dict] = None,
    level_summary: Optional[List[dict]] = None,
    baseline_comparison: Optional[List[dict]] = None,
    baseline_label: Optional[str] = None,
    grid_search_results: Optional[List[dict]] = None,
    best_candidate_name: Optional[str] = None,
    selection_scores: Optional[List[dict]] = None,
    regret_metrics: Optional[dict] = None,
    regret_bounds: Optional[dict] = None,
    regret_settings: Optional[dict] = None,
    pca_explained_variance: Optional[List[float]] = None,
    pca_variance_plot: Optional[str] = None,
    pca_method: Optional[str] = None,
    pca_threshold: Optional[float] = None,
    pca_n_components: Optional[int] = None,
    kaiser_rule_suggestion: Optional[str] = None,
    sampling_plot: Optional[str] = None,
    sampling_explanation: Optional[str] = None,
    figures: Optional[List[dict]] = None,
    summary_points: Optional[List[str]] = None,
    skipped: Optional[List[dict]] = None,
    dataset_name: Optional[str] = None,
    explain: bool = True,
    thresholds: Optional[Thresholds] = None,
    max_points: int = 2000,
) -> ReportData:
    """Collect and derive everything the v2 report shows.

    The arrays are ``(n_test, n_outputs)``. ``explain=False`` skips SHAP and
    partial dependence (they need ``best_model`` and ``X_train``). ``figures``
    are extra PNGs as ``{key, section, title, caption, img_b64}`` dicts, placed
    in their section (``"sec-6"``) or the appendix (``"appendix"``).
    """
    th = thresholds or Thresholds()
    y = np.asarray(y_test, dtype=np.float64).reshape(len(y_test), -1)
    n = len(y)
    pred = _as_2d(best_pred, n)
    std = _as_2d(best_std, n) if best_std is not None else None
    cal = (_as_2d(calibration_residuals, len(calibration_residuals))
           if calibration_residuals is not None else None)
    lower = upper = None
    if conformal_intervals is not None:
        lower, upper = (_as_2d(conformal_intervals[0], n), _as_2d(conformal_intervals[1], n))
    skipped = list(skipped or [])
    base_by = {r["output"]: r for r in baseline_comparison or []}

    outputs = []
    for j, name in enumerate(output_names):
        o = _output_block(
            j, name, y[:, j], pred[:, j], std[:, j] if std is not None else None,
            metrics.get(name, {}), cal[:, j] if cal is not None else None,
            lower[:, j] if lower is not None else None,
            upper[:, j] if upper is not None else None,
            conformal_alpha, base_by.get(name), max_points)
        o["status"] = judge_output(o, n, th)
        outputs.append(o)

    levels = list(fidelity_levels or [])
    if input_names is None and X_train is not None:
        input_names = [f"x{k}" for k in range(np.asarray(X_train).shape[1])]
    run = {
        "title": title,
        "model": model_type,
        "description": description or "",
        "notes": notes or "",
        "dataset": dataset_name,
        "data_hash": data_hash(X_train, y),
        "seed": seed,
        "version": _package_version(),
        "date": _dt.date.today().isoformat(),
        "cross_validation": cross_validation or "",
        "n_train": n_train,
        "n_test": n_test,
        "n_test_points": n,
        "fidelity_levels": [str(v) for v in levels],
        "input_names": list(input_names or []),
        "output_names": list(output_names),
        "highest_level": str(levels[-1]) if levels else None,
    }

    cv = None
    if grid_search_results:
        cv = _cv_from_rows(grid_search_results, best_candidate_name, output_names)
    elif selection_scores:
        cv = _cv_from_selection(selection_scores, output_names)
    else:
        skipped.append({"section": "sec-7", "reason": "Per-candidate CV scores were not "
                        "recorded for this run, so only the selection method is shown."})

    shap_block = pdp_block = None
    if explain and best_model is not None and X_train is not None:
        shap_block, pdp_block = _explain(best_model, np.asarray(X_train, dtype=np.float64),
                                         list(output_names), list(input_names), skipped)
    elif explain:
        skipped.append({"section": "sec-5", "reason": "Input attributions need the fitted "
                        "model and training inputs, which were not passed to the report."})

    pca = None
    if pca_explained_variance:
        pca = {"explained_variance": [float(v) for v in pca_explained_variance],
               "method": pca_method, "threshold": _f(pca_threshold),
               "n_components": pca_n_components,
               "kaiser": _strip_md(kaiser_rule_suggestion), "plot_b64": pca_variance_plot}

    sampling = None
    if sampling_plot or sampling_explanation:
        sampling = {"explanation": sampling_explanation,
                    "plots": [sampling_plot] if sampling_plot else []}

    if not imputation_summary:
        skipped.append({"section": "sec-6", "reason": "No imputation ran (no missing values "
                        "were handled for this run)."})

    data = ReportData(
        run=run,
        outputs=outputs,
        thresholds=th.as_dict(),
        uncertainty_metrics={k: _f(v) for k, v in (uncertainty_metrics or {}).items()
                             if k != "dummy_metric"},
        conformal_alpha=conformal_alpha,
        levels=_levels(level_summary, levels),
        baseline_label=baseline_label,
        cv=cv,
        shap=shap_block,
        pdp=pdp_block,
        imputation=imputation_summary,
        regret=_regret(regret_metrics, regret_bounds, regret_settings),
        pca=pca,
        sampling=sampling,
        figures=list(figures or []),
        skipped=skipped,
        summary_points=list(summary_points or []),
    )
    d = {"outputs": outputs, "run": run, "levels": data.levels, "cv": cv,
         "regret": data.regret}
    data.verdict = build_verdict(outputs, th)
    data.findings = build_findings(d, th)
    data.next_steps = build_next_steps(d, th)
    return data
