# Copyright (c) 2026 takotime808

"""Regenerate the example reports in ``docs/_extra/example_reports/``.

Writes each report in both styles (``*_v2.html``, the redesigned report, and
``*_classic.html``):

* ``grid_search``: the ``multioutreg grid_search`` CLI on the bundled
  3-output example dataset.
* ``auto_detect``: the ``multioutreg grid_search_auto_detect`` CLI on the same
  data, with conformal intervals.
* ``multi_fidelity``: the Multi-Fidelity page's own pipeline (grid search over
  multi-fidelity models, HF-only baseline, per-level fits, conformal) on two
  fidelity levels built from the Park and Currin benchmark pairs, plus one
  output whose low level carries no information.

Run from the repo root (takes a few minutes)::

    OMP_NUM_THREADS=1 python examples/make_example_reports.py

On macOS with both torch and lightgbm installed, import lightgbm first
(``python -c "import lightgbm, runpy; runpy.run_path('examples/make_example_reports.py', run_name='__main__')"``).
"""

import importlib.util
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "_extra", "example_reports")
DATA = os.path.join(ROOT, "multioutreg", "gui", "data", "example_dataset_3_outputs.csv")
STYLES = ("v2", "classic")


def cli_reports() -> None:
    from mor_cli.grid_search import grid_search
    from mor_cli.grid_search_auto_detect import grid_search_auto_detect

    inputs = ",".join(f"feature_{i}" for i in range(5))
    outputs = "output_pressure,output_temperature,output_velocity"
    for style in STYLES:
        grid_search(
            DATA, inputs, outputs, use_pca=False, pca_method=None, n_components=None,
            pca_threshold=None, use_moe=False, moe_n_experts=4, moe_gating_type="linear",
            use_imputation=False, skip_expensive=True,
            description="Example: the bundled 3-output dataset (300 rows, 5 inputs).",
            out_html=os.path.join(OUT, f"grid_search_{style}.html"), report_style=style)
        grid_search_auto_detect(
            DATA, inputs, outputs, use_pca=False, pca_method=None, n_components=None,
            pca_threshold=None, conformal=True, conformal_alpha=0.1, use_moe=False,
            moe_n_experts=4, moe_gating_type="linear", use_imputation=False,
            skip_expensive=True,
            description="Example: the bundled 3-output dataset with conformal intervals.",
            out_html=os.path.join(OUT, f"auto_detect_{style}.html"), report_style=style)


def _mf_page():
    path = os.path.join(ROOT, "multioutreg", "gui", "pages",
                        "02_Multi_Fidelity_Surrogate_Models.py")
    spec = importlib.util.spec_from_file_location("mf_page", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mf_dataframe(seed: int = 0) -> pd.DataFrame:
    """Two fidelity levels, 4 inputs on [0, 1], 3 outputs."""
    from multioutreg.active.benchmarks import currin, currin_lo, park, park_lo

    rng = np.random.default_rng(seed)
    frames = []
    for level, n in (("lo", 300), ("hi", 120)):
        X = rng.uniform(size=(n, 4))
        hi = level == "hi"
        y_park = (park if hi else park_lo)(X)[:, 0]
        y_currin = (currin if hi else currin_lo)(X[:, :2])[:, 0]
        # "drag": the low level is noise around the mean, so it carries no information.
        drag_hi = 0.02 + 0.03 * X[:, 1] ** 2 + 0.01 * np.sin(6 * X[:, 3])
        y_drag = drag_hi + 0.002 * rng.normal(size=n) if hi else 0.03 + 0.01 * rng.normal(size=n)
        frames.append(pd.DataFrame({
            "fidelity": level, "x1": X[:, 0], "x2": X[:, 1], "x3": X[:, 2], "x4": X[:, 3],
            "park": y_park, "currin": y_currin, "drag": y_drag}))
    return pd.concat(frames, ignore_index=True)


def mf_reports(seed: int = 0) -> None:
    """The Multi-Fidelity page's run, step for step, without Streamlit."""
    page = _mf_page()
    df = mf_dataframe(seed)
    inputs, outputs, levels = ["x1", "x2", "x3", "x4"], ["park", "currin", "drag"], ["lo", "hi"]
    level_data = page._split_by_fidelity(df, "fidelity", inputs, outputs, levels)
    train, test = {}, {}
    for lv, (X, Y) in level_data.items():
        X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.25, random_state=seed)
        train[lv], test[lv] = (X_tr, Y_tr), (X_te, Y_te)
    X_test, y_test = test["hi"]
    X_fit, X_cal, Y_fit, y_cal = train_test_split(*train["hi"], test_size=0.2, random_state=seed)
    train["hi"] = (X_fit, Y_fit)

    cands = page._build_candidates(levels, skip_expensive=True, seed=seed)
    gs, best, cv = page._select_best_candidate(cands, train, levels, random_state=seed)
    model = page._fit_candidate(best, train, levels)
    pred, std = page._predict_highest(model, best, X_test, levels)
    metrics = page._compute_metrics(y_test, pred, std, outputs)
    base_label = best["name"].split("+")[1]
    level_preds = page._single_level_predictions(
        best["surrogate_cls"], train, X_test,
        fitted_models=model.models if best["mf_type"] == "independent" else None)
    level_rows = page._level_summary(train, test, level_preds, y_test, outputs)
    hf = level_preds["hi"]
    baseline = (page._baseline_comparison(metrics, page._compute_metrics(y_test, hf[0], hf[1], outputs))
                if isinstance(hf, tuple) else None)
    unc = page._compute_uncertainty_metrics(y_test, pred, std)

    from multioutreg.conformal.base import BaseConformalPredictor

    alpha = 0.1
    cal_pred, _ = page._predict_highest(model, best, X_cal, levels)
    residuals = np.abs(y_cal - cal_pred)
    q = np.array([BaseConformalPredictor._conformal_quantile(residuals[:, j], alpha)
                  for j in range(len(outputs))])
    lower, upper = pred - q, pred + q
    summary = page._summary_points(
        best, gs, metrics, baseline, base_label, level_rows, levels, unc, std, len(X_test),
        conformal_coverage=float(np.mean((y_test >= lower) & (y_test <= upper))),
        conformal_alpha=alpha)
    gs_sorted = sorted(gs, key=lambda r: (r["CV MSE"] is None, r["CV MSE"] or 0.0))
    for style in STYLES:
        html = page.generate_html_report(
            model_type=f"{best['name']} ({type(model).__name__})", fidelity_levels=levels,
            output_names=outputs,
            description=("Example: Park and Currin benchmark pairs plus a 'drag' output whose "
                         "low-fidelity level is pure noise. 300 low-fidelity and 120 "
                         "high-fidelity runs."),
            metrics=metrics, uncertainty_metrics=unc, y_test=y_test, best_pred=pred,
            best_std=std,
            best_model=page._HighestFidelityView(model, best, levels, len(outputs)),
            X_train=X_fit,
            n_train=f"{sum(len(v[0]) for v in train.values())} across all levels "
                    f"(lo: {len(train['lo'][0])}, hi: {len(X_fit)})",
            n_test=f"{len(X_test)} (highest fidelity: hi)",
            cross_validation=(f"Model selected by {page.describe_cv(cv)} on the highest-fidelity "
                              "training data, then refit on all training data. Conformal "
                              f"intervals calibrated on {len(X_cal)} held-out points."),
            seed=seed, notes="Generated by examples/make_example_reports.py.",
            conformal_intervals=(lower, upper), conformal_alpha=alpha, input_names=inputs,
            calibration_residuals=residuals,
            X_design={lv: level_data[lv][0] for lv in levels}, summary_points=summary,
            level_summary=level_rows, baseline_comparison=baseline, baseline_label=base_label,
            grid_search_results=gs_sorted, best_candidate_name=best["name"], report_style=style)
        with open(os.path.join(OUT, f"multi_fidelity_{style}.html"), "w", encoding="utf-8") as fh:
            fh.write(html)
        print(f"wrote multi_fidelity_{style}.html", flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.makedirs(OUT, exist_ok=True)
    os.environ.pop("MOR_TEMPLATE_PATH", None)
    which = sys.argv[1:] or ["cli", "mf"]
    if "mf" in which:
        mf_reports()
    if "cli" in which:
        cli_reports()
