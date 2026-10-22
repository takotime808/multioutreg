# Copyright (c) 2026 takotime808

"""Streamlit page for multi-fidelity surrogate modeling.

Accepts a single CSV with a fidelity level column, splits data per level,
and grid-searches over combinations of multi-fidelity surrogate types and
base surrogate classes:

* StackedVFMSurrogate  — nonlinear recursive feature augmentation (N levels)
* AdditiveCorrectionVFM — two-level additive correction (Kennedy-O'Hagan)
* MultiFidelitySurrogate — independent surrogate per level (no coupling)

Example data can be downloaded from the "Other Dataset Examples" page.
"""

import matplotlib
matplotlib.use("Agg")  # plotting helpers call plt.show(); never open desktop windows

import functools
import html
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from jinja2 import Template
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from multioutreg.figures.conformal_plots import (
    plot_conformal_intervals_ordered,
    plot_conformal_vs_gaussian,
)
from multioutreg.figures.coverage_plots import plot_coverage
from multioutreg.figures.error_histograms import generate_error_histogram
from multioutreg.figures.pdp_plots import generate_pdp_plot
from multioutreg.figures.prediction_plots import plot_predictions_with_error_bars
from multioutreg.figures.residuals import plot_residuals_multioutput_with_regplot
from multioutreg.figures.shap_multioutput import generate_shap_plot
from multioutreg.figures.confidence_intervals import plot_intervals_ordered_multi
from multioutreg.surrogates import (
    AdditiveCorrectionVFM,
    GaussianProcessSurrogate,
    HistGradientBoostingSurrogate,
    LinearRegressionSurrogate,
    MultiFidelitySurrogate,
    RandomForestSurrogate,
    StackedVFMSurrogate,
    ElasticNetSurrogate,
    LassoSurrogate,
)
from multioutreg.surrogates.lightgbm_sklearn import _LIGHTGBM_AVAILABLE, LightGBMSurrogate
from multioutreg.surrogates.xgboost_sklearn import _XGBOOST_AVAILABLE, XGBoostSurrogate
from multioutreg.surrogates.catboost_sklearn import _CATBOOST_AVAILABLE, CatBoostSurrogate
from multioutreg.surrogates.sparse_gp_gpytorch import _GPYTORCH_AVAILABLE, SparseGPSurrogate
from multioutreg.sampling.detect import detect_sampling_method
from multioutreg.utils.figure_utils import plot_to_b64, safe_plot_b64
from multioutreg.utils.imputation import detect_missing, drop_unusable_rows, impute_train_test


# ---------------------------------------------------------------------------
# HTML report helper (mirrors Grid_Search_Surrogate_Models.py)
# ---------------------------------------------------------------------------

def generate_html_report(
    model_type: str,
    fidelity_levels: List[str],
    output_names: List[str],
    description: str,
    metrics: Dict[str, Dict[str, float]],
    uncertainty_metrics: Dict[str, float],
    y_test: np.ndarray,
    best_pred: np.ndarray,
    best_std: np.ndarray,
    best_model: Any,
    X_train: np.ndarray,
    n_train: int,
    n_test: int,
    cross_validation: str,
    seed: int,
    notes: str,
    conformal_intervals: tuple | None = None,
    conformal_alpha: float | None = None,
    imputation_summary: dict | None = None,
    input_names: List[str] | None = None,
    calibration_residuals: np.ndarray | None = None,
    X_design: np.ndarray | Dict[str, np.ndarray] | None = None,
    summary_points: List[str] | None = None,
    level_summary: List[dict] | None = None,
    baseline_comparison: List[dict] | None = None,
    baseline_label: str | None = None,
    grid_search_results: List[dict] | None = None,
    best_candidate_name: str | None = None,
) -> str:
    """Render a Jinja2 HTML report from model evaluation results.

    ``best_model`` is explained by SHAP / PDP, so pass a
    :class:`_HighestFidelityView` to explain the level that is evaluated.
    ``calibration_residuals`` are |y_cal - y_cal_pred| from the conformal
    calibration split; the Gaussian-vs-conformal plot needs them.
    ``X_design`` is the input design used for the sampling section: a dict of
    ``{level: X}`` shows every fidelity level, an array a single design
    (defaults to ``X_train``).
    The remaining optional arguments fill the Summary, Fidelity Levels,
    HF-only baseline and Model Selection sections; each is omitted when None.
    """
    prediction_plots: Dict[str, str] = {}
    prediction_plots["all_in_one"] = safe_plot_b64(
        plot_intervals_ordered_multi,
        best_pred,
        best_std,
        y_test,
        target_list=output_names,
    )
    for i, name in enumerate(output_names):
        prediction_plots[name] = safe_plot_b64(
            plot_predictions_with_error_bars,
            y_test[:, [i]],
            best_pred[:, [i]],
            best_std[:, [i]],
            output_names=[name],
            n_cols=3,
        )

    shap_plots = generate_shap_plot(
        best_model, X_train, output_names, feature_names=input_names
    )

    unc_img = safe_plot_b64(
        plot_coverage, y_test, best_pred, best_std, output_names=output_names
    )
    uncertainty_plots = [
        {
            "img_b64": unc_img,
            "title": "Coverage Plot",
            "caption": (
                "Fraction of highest-fidelity test points inside the model's Gaussian "
                "interval (prediction ± z·std) at each nominal level. Points below the "
                "diagonal mean the predicted std is too small (overconfident); points "
                "above it mean the std is too large (underconfident)."
            ),
        }
    ]

    if input_names is None:
        input_names = [f"x{j}" for j in range(X_train.shape[1])]
    pdp_plots = generate_pdp_plot(
        best_model, X_train, output_names, feature_names=list(input_names)
    )

    if X_design is None:
        X_design = X_train
    if not isinstance(X_design, dict):
        X_design = {(fidelity_levels or ["design"])[-1]: X_design}
    sampling_umap_plot, sampling_method_explanation = _sampling_plot(
        X_design, input_names
    )

    other_img = safe_plot_b64(
        plot_residuals_multioutput_with_regplot,
        best_pred,
        y_test,
        target_list=output_names,
        seed=seed if isinstance(seed, int) else None,
    )
    other_plots = [
        {
            "img_b64": other_img,
            "title": "Residuals",
            "caption": (
                "Residual (observed − predicted) vs predicted value on the "
                "highest-fidelity test set, with a linear trend and its 95% band. "
                "A trend or an offset from zero indicates systematic bias."
            ),
        }
    ]
    other_plots += generate_error_histogram(y_test, best_pred, output_names)

    conformal_plots_list = []
    if conformal_intervals is not None and conformal_alpha is not None:
        y_lower, y_upper = conformal_intervals
        alpha = conformal_alpha

        conformal_plots_list.append({
            "img_b64": safe_plot_b64(
                plot_conformal_intervals_ordered,
                y_test, y_lower, y_upper,
                y_pred=best_pred,
                output_names=output_names,
                alpha=alpha,
            ),
            "title": f"Conformal Prediction Intervals ({int((1 - alpha) * 100)}%)",
            "caption": (
                "Split-conformal prediction intervals on the highest-fidelity test set, "
                "ordered by observed value. The half-width is calibrated on a held-out "
                "part of the highest-fidelity training data, so coverage is at least "
                f"{1 - alpha:.0%} on average when calibration and test points are "
                "exchangeable."
            ),
        })
        if calibration_residuals is not None:
            conformal_plots_list.append({
                "img_b64": safe_plot_b64(
                    plot_conformal_vs_gaussian,
                    y_test, best_pred, best_std,
                    calibration_residuals,
                    output_names=output_names,
                ),
                "title": "Gaussian CI vs Conformal PI Coverage",
                "caption": (
                    "Empirical test-set coverage at each nominal level for Gaussian "
                    "intervals (prediction ± z·std) and conformal intervals "
                    "(prediction ± a quantile of the calibration residuals). Curves "
                    "near the diagonal are well calibrated."
                ),
            })

    template_path = os.path.join(
        os.path.dirname(__file__), "../../report/template.html"
    )
    env_override = os.environ.get("MOR_TEMPLATE_PATH")
    if env_override:
        template_path = env_override
    with open(template_path, "r", encoding="utf-8") as f:
        template_text = f.read()

    rendered = Template(template_text).render(
        project_title="Multi-Fidelity Surrogate Modeling Report",
        model_type=model_type,
        fidelity_levels=fidelity_levels,
        output_names=output_names,
        description=description,
        metrics=metrics,
        uncertainty_metrics=uncertainty_metrics,
        uncertainty_plots=uncertainty_plots,
        prediction_plots=prediction_plots,
        shap_plots=shap_plots,
        pdp_plots=pdp_plots,
        sampling_umap_plot=sampling_umap_plot,
        sampling_method_explanation=sampling_method_explanation,
        sampling_other_plots=[],
        other_plots=other_plots,
        n_train=n_train,
        n_test=n_test,
        cross_validation=cross_validation,
        seed=seed,
        notes=notes,
        feature_names_pca=None,
        pca_explained_variance=None,
        pca_variance_plot=None,
        pca_method=None,
        pca_threshold=None,
        pca_n_components=None,
        kaiser_rule_suggestion=None,
        conformal_plots=conformal_plots_list,
        imputation_summary=imputation_summary,
        summary_points=summary_points,
        level_summary=level_summary,
        baseline_comparison=baseline_comparison,
        baseline_label=baseline_label,
        grid_search_results=grid_search_results,
        best_candidate_name=best_candidate_name,
    )
    return rendered


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _split_by_fidelity(
    df: pd.DataFrame,
    fidelity_col: str,
    input_cols: List[str],
    output_cols: List[str],
    ordered_levels: List,
) -> Dict[str, tuple]:
    """Return ``{level_name: (X, Y)}`` for each fidelity level."""
    level_data: Dict[str, tuple] = {}
    for raw in ordered_levels:
        mask = df[fidelity_col] == raw
        sub = df[mask]
        X_k = sub[input_cols].values.astype(np.float64)
        Y_k = sub[output_cols].values.astype(np.float64)
        level_data[str(raw)] = (X_k, Y_k)
    return level_data


def _compute_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    output_names: List[str],
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(output_names):
        metrics[name] = {
            "r2": r2_score(y_test[:, i], y_pred[:, i]),
            "rmse": root_mean_squared_error(y_test[:, i], y_pred[:, i]),
            "mae": mean_absolute_error(y_test[:, i], y_pred[:, i]),
            "mean_predicted_std": float(np.mean(y_std[:, i])),
        }
    return metrics


def _compute_uncertainty_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    nominal: float = 0.95,
) -> Dict[str, float]:
    """Average (over outputs) calibration metrics of the Gaussian ``mean ± std``."""
    err = np.abs(y_test - y_pred)
    z = norm.ppf(0.5 + nominal / 2)
    coverage = float(np.mean(err <= z * y_std))
    pct = f"{nominal:.0%}"
    out = {
        "Sharpness (mean predicted std)": float(np.mean(y_std)),
        f"Empirical coverage of {pct} Gaussian interval": coverage,
        f"Miscoverage at {pct} (|nominal − empirical|)": abs(nominal - coverage),
    }
    # NLL is undefined for zero std (models without native uncertainty).
    if np.all(y_std > 0):
        out["Gaussian NLL (mean per point)"] = float(
            np.mean(-norm.logpdf(y_test, loc=y_pred, scale=y_std))
        )
    return out


def _single_level_predictions(
    surrogate_cls: Any,
    train_data: Dict[str, tuple],
    X_test: np.ndarray,
    fitted_models: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Fit ``surrogate_cls`` on each level alone and predict the hi-fi test inputs.

    Returns ``{level: (y_pred, y_std)}``, or ``{level: "error message"}`` when
    that level's fit fails. The highest level's entry is the HF-only baseline.
    ``fitted_models`` (e.g. ``MultiFidelitySurrogate.models``) are reused
    instead of refitting, so an independent winner is compared with exactly
    its own per-level models rather than a re-randomised refit.
    """
    out: Dict[str, Any] = {}
    for lv, (X_k, Y_k) in train_data.items():
        try:
            if fitted_models is not None and lv in fitted_models:
                model = fitted_models[lv]
            else:
                model = surrogate_cls().fit(X_k, Y_k)
            pred, std = model.predict(X_test, return_std=True)
            pred = np.asarray(pred, dtype=np.float64).reshape(len(X_test), -1)
            std = np.asarray(std, dtype=np.float64).reshape(len(X_test), -1)
            out[lv] = (pred, std)
        except Exception as exc:
            out[lv] = f"{type(exc).__name__}: {exc}"
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _level_summary(
    train_data: Dict[str, tuple],
    test_data: Dict[str, tuple],
    level_preds: Dict[str, Any],
    y_test: np.ndarray,
    output_names: List[str],
) -> List[dict]:
    """One row per (level, output): sizes, output spread, and how well a model
    trained on that level alone tracks the highest-fidelity test outputs."""
    rows: List[dict] = []
    for lv, (_, Y_k) in train_data.items():
        res = level_preds.get(lv)
        for j, name in enumerate(output_names):
            row = {
                "level": lv,
                "output": name,
                "n_train": int(Y_k.shape[0]),
                "n_test": int(test_data[lv][1].shape[0]),
                "mean": float(np.mean(Y_k[:, j])),
                "std": float(np.std(Y_k[:, j])),
                "r": None,
                "r2": None,
                "error": None,
            }
            if isinstance(res, tuple):
                row["r"] = _pearson(res[0][:, j], y_test[:, j])
                row["r2"] = float(r2_score(y_test[:, j], res[0][:, j]))
            else:
                row["error"] = res
            rows.append(row)
    return rows


def _baseline_comparison(
    metrics: Dict[str, Dict[str, float]],
    baseline_metrics: Dict[str, Dict[str, float]],
) -> List[dict]:
    """Per-output multi-fidelity vs HF-only test metrics."""
    rows = []
    for name, m in metrics.items():
        b = baseline_metrics[name]
        rows.append({
            "output": name,
            "mf_rmse": m["rmse"],
            "hf_rmse": b["rmse"],
            "rmse_change_pct": (
                100.0 * (m["rmse"] - b["rmse"]) / b["rmse"] if b["rmse"] > 0 else None
            ),
            "mf_r2": m["r2"],
            "hf_r2": b["r2"],
        })
    return rows


def _summary_points(
    best_cand: dict,
    gs_rows: List[dict],
    metrics: Dict[str, Dict[str, float]],
    baseline_rows: List[dict] | None,
    baseline_label: str,
    level_rows: List[dict],
    level_names: List[str],
    uncertainty_metrics: Dict[str, float],
    best_std: np.ndarray,
    n_test: int,
    conformal_coverage: float | None = None,
    conformal_alpha: float | None = None,
) -> List[str]:
    """Plain-language findings derived from the computed metrics."""
    pts: List[str] = []

    ok = sorted(
        (r for r in gs_rows if r["Val MSE"] is not None), key=lambda r: r["Val MSE"]
    )
    msg = (
        f"Best configuration: {best_cand['name']} "
        f"(validation MSE {ok[0]['Val MSE']:.4g}, chosen from {len(ok)} working "
        f"candidates"
    )
    if len(ok) > 1 and ok[0]["Val MSE"] > 0:
        gap = 100.0 * (ok[1]["Val MSE"] - ok[0]["Val MSE"]) / ok[0]["Val MSE"]
        msg += f"; runner-up {ok[1]['Model']} is {gap:.0f}% worse"
    n_failed = len(gs_rows) - len(ok)
    if n_failed:
        msg += f"; {n_failed} failed to fit"
    pts.append(msg + ").")

    r2s = [m["r2"] for m in metrics.values()]
    pts.append(
        f"Highest-fidelity test R² is {min(r2s):.3f}"
        + (f" to {max(r2s):.3f} across outputs." if len(r2s) > 1 else ".")
    )

    if baseline_rows:
        mf = float(np.mean([r["mf_rmse"] for r in baseline_rows]))
        hf = float(np.mean([r["hf_rmse"] for r in baseline_rows]))
        change = 100.0 * (mf - hf) / hf if hf > 0 else 0.0
        if best_cand["mf_type"] == "independent":
            pts.append(
                "The winner fits each level independently, so its highest-fidelity "
                "predictions do not use the lower-fidelity data: it is the HF-only "
                f"{baseline_label} model (test RMSE {mf:.4g}). No coupled "
                "multi-fidelity model beat high-fidelity data alone on validation "
                "for this dataset."
            )
        elif change <= -5:
            pts.append(
                f"Using the lower-fidelity data reduced mean test RMSE by {-change:.0f}% "
                f"compared with an HF-only {baseline_label} ({mf:.4g} vs {hf:.4g})."
            )
        elif change >= 5:
            pts.append(
                f"The multi-fidelity model's mean test RMSE is {change:.0f}% worse than "
                f"an HF-only {baseline_label} ({mf:.4g} vs {hf:.4g}); the lower-fidelity "
                "data is not helping and an HF-only model may be preferable."
            )
        else:
            pts.append(
                f"Mean test RMSE is within 5% of an HF-only {baseline_label} "
                f"({mf:.4g} vs {hf:.4g}): no clear gain from the lower-fidelity data."
            )
    else:
        pts.append(f"The HF-only {baseline_label} baseline could not be fitted.")

    lower = [r for r in level_rows if r["level"] != level_names[-1]]
    weak = sorted({
        r["level"] for r in lower if r["r"] is None or abs(r["r"]) < 0.5
    })
    rs = [abs(r["r"]) for r in lower if r["r"] is not None]
    if weak:
        pts.append(
            "Weakly related lower level(s): "
            + ", ".join(f"'{lv}'" for lv in weak)
            + " (|r| < 0.5 with the highest-fidelity test outputs for at least one "
            "output), so they carry little usable information."
        )
    elif rs:
        pts.append(
            f"All lower levels track the highest fidelity closely (|r| ≥ {min(rs):.2f}), "
            "so they are informative even where their R² is lower because of bias."
        )

    cov_key = next(k for k in uncertainty_metrics if k.startswith("Empirical coverage"))
    cov = uncertainty_metrics[cov_key]
    if not np.any(best_std > 0):
        pts.append(
            "The model reports no predictive uncertainty (std = 0); "
            "use conformal intervals for error bars."
        )
    elif cov < 0.85:
        pts.append(
            f"The model's Gaussian uncertainty is overconfident: the 95% interval "
            f"covers only {cov:.0%} of test points"
            + (", so use the conformal intervals instead." if conformal_coverage is not None
               else "; enable conformal intervals for calibrated error bars.")
        )
    elif cov > 0.99:
        pts.append(
            f"The 95% Gaussian interval covers {cov:.0%} of test points; the predicted "
            "std may be conservative (intervals wider than needed)."
        )
    else:
        pts.append(f"The 95% Gaussian interval covers {cov:.0%} of test points, close to nominal.")

    if conformal_coverage is not None and conformal_alpha is not None:
        pts.append(
            f"Conformal {1 - conformal_alpha:.0%} intervals cover {conformal_coverage:.0%} "
            "of test points."
        )

    if n_test < 30:
        pts.append(
            f"The test set has only {n_test} highest-fidelity points, so these metrics "
            "(and the comparison above) are noisy."
        )
    return pts


_LEVEL_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _sampling_plot(designs: Dict[str, np.ndarray], input_names: List[str]) -> tuple:
    """Return ``(b64_png, explanation)`` for the sampling section of the report.

    ``designs`` maps each fidelity level (lowest to highest) to its input
    design. All levels are min-max scaled to [0, 1] with the bounds of their
    union and drawn on one plot, one colour + marker per level. The sampling
    method is inferred separately for each level.
    """
    designs = {
        str(lv): np.asarray(X, dtype=np.float64) for lv, X in designs.items()
    }
    designs = {lv: X[np.all(np.isfinite(X), axis=1)] for lv, X in designs.items()}
    designs = {lv: X for lv, X in designs.items() if len(X)}

    lines = []
    for lv, X in designs.items():
        try:
            det = detect_sampling_method(X)
            txt = f"{det.method}. {det.explanation}"
        except ValueError as exc:
            txt = f"not inferred: {exc}"
        lines.append(f"<b>{html.escape(lv)}</b> (n={len(X)}): {html.escape(txt)}")
    explanation = (
        "Inferred sampling method per fidelity level:<br>" + "<br>".join(lines)
        + "<br><i>The plot scales every input to [0, 1] using the bounds of all "
        "levels combined, so the levels can be compared directly.</i>"
    )

    if not designs:
        return None, explanation

    X_all = np.vstack(list(designs.values()))
    lo = X_all.min(axis=0)
    span = np.where(X_all.max(axis=0) > lo, X_all.max(axis=0) - lo, 1.0)
    scaled = {lv: (X - lo) / span for lv, X in designs.items()}
    n_dim = X_all.shape[1]
    colors = plt.get_cmap("tab10").colors

    def plot_fn():
        plt.figure()
        if n_dim > 2 and len(X_all) >= 5:
            import umap

            emb_all = umap.UMAP(
                n_neighbors=min(15, len(X_all) - 1), n_components=2, random_state=0
            ).fit_transform(np.vstack(list(scaled.values())))
            offsets = np.cumsum([0] + [len(X) for X in scaled.values()])
            coords = {
                lv: emb_all[offsets[k]:offsets[k + 1]]
                for k, lv in enumerate(scaled)
            }
            plt.xlabel("UMAP-1")
            plt.ylabel("UMAP-2")
            plt.title("UMAP projection of input design, all fidelity levels")
        elif n_dim == 1:
            coords = {
                lv: np.column_stack([X[:, 0], np.full(len(X), k)])
                for k, (lv, X) in enumerate(scaled.items())
            }
            plt.xlabel(f"{input_names[0]} (scaled)")
            plt.yticks(range(len(scaled)), list(scaled))
            plt.ylim(-0.5, len(scaled) - 0.5)
            plt.title("Input design by fidelity level")
        else:
            coords = {lv: X[:, :2] for lv, X in scaled.items()}
            plt.xlabel(f"{input_names[0]} (scaled)")
            plt.ylabel(f"{input_names[1]} (scaled)")
            plt.title("Input design, all fidelity levels")
        for k, (lv, C) in enumerate(coords.items()):
            plt.scatter(
                C[:, 0], C[:, 1],
                color=colors[k % len(colors)],
                marker=_LEVEL_MARKERS[k % len(_LEVEL_MARKERS)],
                alpha=0.7, label=f"{lv} (n={len(C)})", zorder=2 + k,
            )
        plt.legend(title="Fidelity level", fontsize="small")

    return plot_to_b64(plot_fn), explanation


# ---------------------------------------------------------------------------
# Grid search helpers
# ---------------------------------------------------------------------------

# Constructor argument each base surrogate uses for its seed; None means the
# model is deterministic. Anything not listed takes ``random_state``.
_SEED_KWARG: Dict[Any, str | None] = {
    CatBoostSurrogate: "random_seed",
    LinearRegressionSurrogate: None,
}


def _seeded(surrogate_cls: Any, seed: int | None) -> Any:
    """Return a zero-argument factory for ``surrogate_cls`` seeded with ``seed``.

    With ``seed=None`` the class is returned unchanged (each model uses its own
    default, which is unseeded for e.g. SparseGP / RF).
    """
    kw = _SEED_KWARG.get(surrogate_cls, "random_state")
    if seed is None or kw is None:
        return surrogate_cls
    return functools.partial(surrogate_cls, **{kw: seed})


def _build_candidates(
    level_names: List[str], skip_expensive: bool, seed: int | None = None
) -> List[dict]:
    """Return a list of candidate configs for the grid search.

    Each config is a dict with keys:
        name            display name for the results table
        mf_type         "stacked" | "additive" | "independent"
        surrogate_cls   base surrogate factory (class, or class seeded via
                        :func:`_seeded` when ``seed`` is given)
        augment_with_std  bool (StackedVFM only)
    """
    surrogates: List[tuple] = [
        ("RF", RandomForestSurrogate),
        ("HGB", HistGradientBoostingSurrogate),
        ("Linear", LinearRegressionSurrogate),
        ("ElasticNet", ElasticNetSurrogate),
        ("Lasso", LassoSurrogate),
    ]
    if not skip_expensive:
        surrogates.append(("GP", GaussianProcessSurrogate))
    if _LIGHTGBM_AVAILABLE:
        surrogates.append(("LightGBM", LightGBMSurrogate))
    if _XGBOOST_AVAILABLE:
        surrogates.append(("XGBoost", XGBoostSurrogate))
    if _CATBOOST_AVAILABLE:
        surrogates.append(("CatBoost", CatBoostSurrogate))
    if _GPYTORCH_AVAILABLE:
        surrogates.append(("SparseGP", SparseGPSurrogate))

    candidates: List[dict] = []
    for sur_name, sur_cls in surrogates:
        sur_cls = _seeded(sur_cls, seed)
        # StackedVFM — without and with std augmentation
        candidates.append({
            "name": f"StackedVFM+{sur_name}",
            "mf_type": "stacked",
            "surrogate_cls": sur_cls,
            "augment_with_std": False,
        })
        candidates.append({
            "name": f"StackedVFM+{sur_name}+augStd",
            "mf_type": "stacked",
            "surrogate_cls": sur_cls,
            "augment_with_std": True,
        })
        # MultiFidelitySurrogate (independent per level, no coupling)
        candidates.append({
            "name": f"MultiFidelity+{sur_name}",
            "mf_type": "independent",
            "surrogate_cls": sur_cls,
            "augment_with_std": False,
        })
        # AdditiveCorrectionVFM — 2-level only
        if len(level_names) == 2:
            candidates.append({
                "name": f"AdditiveVFM+{sur_name}",
                "mf_type": "additive",
                "surrogate_cls": sur_cls,
                "augment_with_std": False,
            })

    return candidates


def _fit_candidate(
    cand: dict,
    train_data: Dict[str, tuple],
    level_names: List[str],
) -> Any:
    """Fit and return the model described by *cand*."""
    sur_cls = cand["surrogate_cls"]

    if cand["mf_type"] == "stacked":
        model = StackedVFMSurrogate(
            fidelity_levels=level_names,
            surrogate_cls=sur_cls,
            augment_with_std=cand["augment_with_std"],
        )
        model.fit(train_data)

    elif cand["mf_type"] == "additive":
        lo_name, hi_name = level_names[0], level_names[1]
        model = AdditiveCorrectionVFM(
            lo_surrogate_cls=sur_cls,
            hi_surrogate_cls=sur_cls,
        )
        model.fit({"lo": train_data[lo_name], "hi": train_data[hi_name]})

    else:  # independent
        model = MultiFidelitySurrogate(sur_cls, level_names)
        model.fit(train_data)

    return model


def _predict_highest(
    model: Any,
    cand: dict,
    X: np.ndarray,
    level_names: List[str],
) -> tuple:
    """Return highest-fidelity ``(mean, std)`` predictions as 2-D float arrays."""
    if cand["mf_type"] == "independent":
        # MultiFidelitySurrogate.predict defaults to the *lowest* level.
        y_pred, y_std = model.predict(X, level=level_names[-1], return_std=True)
    else:
        y_pred, y_std = model.predict(X, return_std=True)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_std = np.asarray(y_std, dtype=np.float64)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_std.ndim == 1:
        y_std = y_std.reshape(-1, 1)
    return y_pred, y_std


class _OutputColumn(RegressorMixin, BaseEstimator):
    """Single-output, already-fitted regressor for SHAP and sklearn PDP."""

    def __init__(self, predict_fn=None, index: int = 0):
        self.predict_fn = predict_fn
        self.index = index

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return self.predict_fn(np.asarray(X, dtype=np.float64))[:, self.index]

    def __sklearn_is_fitted__(self):
        return True


class _HighestFidelityView:
    """Expose a fitted multi-fidelity model's highest-fidelity predictions.

    ``MultiFidelitySurrogate.predict`` defaults to the lowest level and none of
    the multi-fidelity models have ``estimators_``, so SHAP / PDP would
    otherwise explain the wrong level or be skipped.
    """

    def __init__(self, model: Any, cand: dict, level_names: List[str], n_outputs: int):
        self.model = model
        self._cand = cand
        self._level_names = level_names
        self.estimators_ = [_OutputColumn(self.predict, i) for i in range(n_outputs)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _predict_highest(self.model, self._cand, X, self._level_names)[0]


def _select_best_candidate(
    candidates: List[dict],
    train_data: Dict[str, tuple],
    level_names: List[str],
    val_size: float = 0.2,
    random_state: int = 0,
    on_progress: Any = None,
) -> tuple:
    """Grid-search *candidates* on a validation split of the highest-fidelity
    training data and return ``(results, best_cand)``.

    The test set is never seen here, so the winner's test-set metrics are an
    honest estimate rather than the best of N scores on the same data.
    """
    hi_name = level_names[-1]
    X_hi, Y_hi = train_data[hi_name]
    if len(X_hi) < 5:
        raise ValueError(
            f"Highest-fidelity level '{hi_name}' has only {len(X_hi)} training "
            "samples; at least 5 are needed to hold out a validation split."
        )
    X_fit, X_val, Y_fit, Y_val = train_test_split(
        X_hi, Y_hi, test_size=val_size, random_state=random_state
    )
    fit_data = {**train_data, hi_name: (X_fit, Y_fit)}

    results: List[dict] = []
    best_cand = None
    best_mse = np.inf
    for i, cand in enumerate(candidates):
        if on_progress is not None:
            on_progress(i, cand)
        try:
            model = _fit_candidate(cand, fit_data, level_names)
            y_pred, _ = _predict_highest(model, cand, X_val, level_names)
            mse = float(np.mean((Y_val - y_pred) ** 2))
            results.append({
                "Model": cand["name"],
                "Val MSE": round(mse, 6),
                "Val RMSE": round(float(np.sqrt(mse)), 6),
                "Val R²": round(float(r2_score(Y_val.ravel(), y_pred.ravel())), 4),
                "Status": "OK",
            })
            if mse < best_mse:
                best_mse = mse
                best_cand = cand
        except Exception as exc:
            results.append({
                "Model": cand["name"],
                "Val MSE": None,
                "Val RMSE": None,
                "Val R²": None,
                "Status": f"Error: {exc}",
            })
    return results, best_cand


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------

st.title("Multi-Fidelity Surrogate Model")
st.info(
    "Upload a single CSV that contains a **fidelity level column** alongside your input "
    "features and output targets.  Each unique value in that column defines one fidelity "
    "level (e.g. 0 = low, 1 = medium, 2 = high).  \n\n"
    "The page will **grid-search** over all combinations of multi-fidelity model type "
    "and base surrogate class, scoring each on a validation split of the highest-fidelity "
    "training data. The best configuration is refit on all training data, evaluated once "
    "on the held-out test set, and reported.\n\n"
    "Example data can be downloaded from the **Other Dataset Examples** page."
)

uploaded_file = st.file_uploader(
    "Upload CSV file with a fidelity level column.",
    type=["csv"],
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("## Preview", df.head())

    # ------------------------------------------------------------------
    # Missing value handling
    # ------------------------------------------------------------------
    _missing_summary = detect_missing(df)
    _impute_choices: Dict[str, str] = {}
    if not _missing_summary.empty:
        st.warning(
            f"{int(_missing_summary['missing_count'].sum())} missing value(s) detected "
            f"across {len(_missing_summary)} column(s)."
        )
        st.dataframe(_missing_summary)
        st.write("**Choose how to handle missing values per column:**")
        st.caption(
            "Missing target values are never imputed: those rows are always dropped. "
            "Imputation of inputs is fitted on the training split only."
        )
        for _col in _missing_summary.index:
            _impute_choices[_col] = st.selectbox(
                f"`{_col}`",
                ["Impute (KNN)", "Drop rows"],
                key=f"imp_{_col}",
            )

    # ------------------------------------------------------------------
    # Fidelity column selector — outside the form so changing it
    # immediately recomputes `remaining` and the level options.
    # ------------------------------------------------------------------
    st.subheader("Column Selection")
    fidelity_col = st.selectbox(
        "Fidelity level column",
        options=df.columns,
        help="Column whose unique values identify fidelity levels (e.g. 0, 1, 2 or 'lo', 'hi').",
        key="fidelity_col_select",
    )
    remaining = [c for c in df.columns if c != fidelity_col]
    _detected = sorted(df[fidelity_col].dropna().unique().tolist(), key=str)

    # ------------------------------------------------------------------
    # Column selection + grid search options form
    # ------------------------------------------------------------------
    with st.form("mf_column_selection"):
        input_cols = st.multiselect("Input feature columns", options=remaining)
        output_cols = st.multiselect("Output target columns", options=remaining)

        st.subheader("Fidelity Level Ordering")
        ordered_levels = st.multiselect(
            "Fidelity levels — ordered lowest → highest",
            options=_detected,
            default=_detected,
            help=(
                "Re-order by removing and re-adding levels in the correct order. "
                "The last entry is treated as the highest fidelity."
            ),
        )

        st.subheader("Grid Search Options")
        skip_expensive = st.checkbox(
            "Skip computationally expensive models (Gaussian Process)",
            value=True,
            help=(
                "Gaussian Process has O(n³) training cost and becomes very slow for "
                "n > ~300.  Uncheck to include GP-based candidates in the grid search."
            ),
        )
        fix_seed = st.checkbox(
            "Fix random seed (reproducible results)",
            value=True,
            help=(
                "When checked, the data splits and every model with randomness "
                "(RF, SparseGP, boosting, GP optimizer restarts) use the seed below, "
                "so re-running gives identical results. Uncheck to see run-to-run "
                "variation."
            ),
        )
        seed_value = st.number_input(
            "Random seed", min_value=0, value=0, step=1,
            help="Ignored when 'Fix random seed' is unchecked.",
        )
        use_conformal = st.checkbox(
            "Compute conformal prediction intervals",
            value=False,
            help="Calibrate split-conformal intervals on a held-out part of the highest-fidelity training data.",
        )
        conformal_alpha_sel = st.slider(
            "Conformal alpha (miscoverage level)",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Target miscoverage rate α; intervals achieve ≥ 1−α marginal coverage.",
        )

        description = st.text_area("Optional: project description")
        submitted = st.form_submit_button("Run Grid Search")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if submitted:
        if not input_cols:
            st.error("Please select at least one input feature column.")
            st.stop()
        if not output_cols:
            st.error("Please select at least one output target column.")
            st.stop()
        if len(ordered_levels) < 2:
            st.error("Please select at least 2 fidelity levels.")
            st.stop()

        # Apply imputation
        _all_selected = list(input_cols) + list(output_cols)
        _cols_to_impute = [
            c for c in _all_selected
            if st.session_state.get(f"imp_{c}") == "Impute (KNN)"
        ]
        _cols_to_drop = [
            c for c in _all_selected
            if st.session_state.get(f"imp_{c}") == "Drop rows"
        ]
        _rows_before = len(df)
        # Targets are never imputed: rows missing a target are dropped. Inputs marked
        # "Impute (KNN)" are imputed per level after the train/test split (below).
        df = drop_unusable_rows(df, list(output_cols), _cols_to_drop)
        _rows_after = len(df)

        _imputation_summary = None
        if (_cols_to_impute or _cols_to_drop) and not _missing_summary.empty:
            _summary_cols = []
            for _c in _cols_to_impute:
                if _c in _missing_summary.index:
                    _summary_cols.append({
                        "name": _c,
                        "action": (
                            "Rows dropped (missing target)" if _c in output_cols
                            else "Imputed (KNN, fit on training split)"
                        ),
                        "missing_count": int(_missing_summary.loc[_c, "missing_count"]),
                        "missing_pct": float(_missing_summary.loc[_c, "missing_pct"]),
                    })
            for _c in _cols_to_drop:
                if _c in _missing_summary.index:
                    _summary_cols.append({
                        "name": _c,
                        "action": "Rows dropped",
                        "missing_count": int(_missing_summary.loc[_c, "missing_count"]),
                        "missing_pct": float(_missing_summary.loc[_c, "missing_pct"]),
                    })
            _imputation_summary = {
                "rows_before": _rows_before,
                "rows_after": _rows_after,
                "columns": _summary_cols,
            }

        seed = int(seed_value) if fix_seed else None

        # Split by fidelity level
        level_names = [str(lv) for lv in ordered_levels]
        level_data = _split_by_fidelity(
            df, fidelity_col, list(input_cols), list(output_cols), ordered_levels
        )

        # Warn on small levels
        for lv_name, (X_k, _) in level_data.items():
            if X_k.shape[0] < 10:
                st.warning(
                    f"Level '{lv_name}' has only {X_k.shape[0]} samples — "
                    "results may be unreliable."
                )

        # Train/test split per level (75/25); evaluate on highest-fidelity test set
        train_data: Dict[str, tuple] = {}
        test_data: Dict[str, tuple] = {}
        for lv_name, (X_k, Y_k) in level_data.items():
            X_tr, X_te, Y_tr, Y_te = train_test_split(
                X_k, Y_k, test_size=0.25, random_state=seed
            )
            X_tr, X_te = impute_train_test(X_tr, X_te)
            train_data[lv_name] = (X_tr, Y_tr)
            test_data[lv_name] = (X_te, Y_te)

        X_test, y_test = test_data[level_names[-1]]

        # Conformal calibration must use data disjoint from both training and the
        # test set used to report coverage, so hold out part of the hi-fi training split.
        if use_conformal:
            _X_hi, _Y_hi = train_data[level_names[-1]]
            _X_fit, X_cal, _Y_fit, y_cal = train_test_split(
                _X_hi, _Y_hi, test_size=0.2, random_state=seed
            )
            train_data[level_names[-1]] = (_X_fit, _Y_fit)

        X_train_hi, _ = train_data[level_names[-1]]  # used for SHAP / plots
        output_names = list(output_cols)

        # ------------------------------------------------------------------
        # Grid search
        # ------------------------------------------------------------------
        candidates = _build_candidates(level_names, skip_expensive, seed=seed)

        prog = st.progress(0, text="Running grid search…")

        def _on_progress(_i: int, _cand: dict) -> None:
            prog.progress(
                (_i + 1) / len(candidates),
                text=f"Fitting {_cand['name']} ({_i + 1}/{len(candidates)})…",
            )

        try:
            gs_results, best_cand = _select_best_candidate(
                candidates, train_data, level_names, random_state=seed,
                on_progress=_on_progress,
            )
        except ValueError as exc:
            prog.empty()
            st.error(str(exc))
            st.stop()

        prog.empty()

        if best_cand is None:
            st.error(
                "All model candidates failed. "
                "Check the error messages in the results table below."
            )
            st.dataframe(pd.DataFrame(gs_results))
            st.stop()

        # Refit the winner on all training data; the test set is used only here.
        best_name = best_cand["name"]
        best_model = _fit_candidate(best_cand, train_data, level_names)
        best_pred, best_std = _predict_highest(best_model, best_cand, X_test, level_names)

        _summary_slot = st.container()  # filled once all metrics are computed

        st.write(f"### Grid Search Results (validation split) — best: **{best_name}**")
        gs_sorted = sorted(
            gs_results,
            key=lambda r: (r["Val MSE"] is None, r["Val MSE"] or 0.0),
        )
        st.dataframe(pd.DataFrame(gs_sorted), width="stretch")

        # Per-output metrics for the best model
        metrics = _compute_metrics(y_test, best_pred, best_std, output_names)
        st.write("### Best Model Metrics (highest-fidelity test set)")
        st.dataframe(pd.DataFrame(metrics).T)

        # Each level fitted alone with the winner's base surrogate: shows how
        # informative every level is, and the HF-only baseline (highest level).
        _base_cls = best_cand["surrogate_cls"]
        _base_label = best_name.split("+")[1]
        with st.spinner("Fitting single-level comparison models…"):
            level_preds = _single_level_predictions(
                _base_cls, train_data, X_test,
                fitted_models=(
                    best_model.models if best_cand["mf_type"] == "independent" else None
                ),
            )
        level_rows = _level_summary(
            train_data, test_data, level_preds, y_test, output_names
        )
        _hf_res = level_preds[level_names[-1]]
        baseline_rows = None
        if isinstance(_hf_res, tuple):
            baseline_rows = _baseline_comparison(
                metrics, _compute_metrics(y_test, _hf_res[0], _hf_res[1], output_names)
            )
            st.write(f"### Multi-Fidelity vs HF-only {_base_label} (highest-fidelity test set)")
            st.dataframe(pd.DataFrame(baseline_rows).set_index("output"))
        else:
            st.warning(f"HF-only baseline could not be fitted: {_hf_res}")
        st.write("### Fidelity Levels")
        st.caption(
            "r and R²: a model trained on that level alone, scored against the "
            "highest-fidelity test outputs."
        )
        st.dataframe(pd.DataFrame(level_rows))

        uncertainty_metrics = _compute_uncertainty_metrics(y_test, best_pred, best_std)
        st.write("### Uncertainty Calibration (Gaussian std, highest-fidelity test set)")
        st.dataframe(pd.Series(uncertainty_metrics, name="value").to_frame())

        # Conformal prediction
        conformal_intervals = None
        residuals = None
        if use_conformal:
            from multioutreg.conformal.base import BaseConformalPredictor
            from multioutreg.conformal.metrics import conformal_summary as _conf_summary

            _cal_pred, _ = _predict_highest(best_model, best_cand, X_cal, level_names)
            residuals = np.abs(y_cal - _cal_pred)
            n_outputs = y_test.shape[1]
            q = np.array([
                BaseConformalPredictor._conformal_quantile(
                    residuals[:, j], conformal_alpha_sel
                )
                for j in range(n_outputs)
            ])
            y_lower = best_pred - q[np.newaxis, :]
            y_upper = best_pred + q[np.newaxis, :]
            conformal_intervals = (y_lower, y_upper)

            conf_df = _conf_summary(
                y_test, y_lower, y_upper,
                alpha=conformal_alpha_sel,
                output_names=output_names,
            )
            st.write(f"### Conformal Prediction Summary (alpha={conformal_alpha_sel})")
            st.dataframe(conf_df)

        summary_points = _summary_points(
            best_cand, gs_results, metrics, baseline_rows, _base_label,
            level_rows, level_names, uncertainty_metrics, best_std, X_test.shape[0],
            conformal_coverage=(
                float(np.mean((y_test >= y_lower) & (y_test <= y_upper)))
                if use_conformal else None
            ),
            conformal_alpha=conformal_alpha_sel if use_conformal else None,
        )
        with _summary_slot:
            st.write("### Summary")
            st.markdown("\n".join(f"- {p}" for p in summary_points))

        _n_per_level = {lv: train_data[lv][0].shape[0] for lv in level_names}
        _n_train_txt = (
            f"{sum(_n_per_level.values())} across all levels ("
            + ", ".join(f"{lv}: {n}" for lv, n in _n_per_level.items())
            + ")"
        )
        _cv_txt = (
            "Model selected on a hold-out validation split (20% of highest-fidelity "
            "training data), then refit on all training data. Metrics are on a 25% "
            "highest-fidelity test split never used for selection."
        )
        if use_conformal:
            _cv_txt += (
                f" Conformal intervals calibrated on {X_cal.shape[0]} highest-fidelity "
                "training points held out from fitting."
            )

        # Report
        try:
            with st.spinner("Generating HTML report…"):
                html = generate_html_report(
                    model_type=f"{best_name} ({best_model.__class__.__name__})",
                    fidelity_levels=level_names,
                    output_names=output_names,
                    description=description,
                    metrics=metrics,
                    uncertainty_metrics=uncertainty_metrics,
                    y_test=y_test,
                    best_pred=best_pred,
                    best_std=best_std,
                    best_model=_HighestFidelityView(
                        best_model, best_cand, level_names, len(output_names)
                    ),
                    X_train=X_train_hi,
                    n_train=_n_train_txt,
                    n_test=f"{X_test.shape[0]} (highest fidelity: {level_names[-1]})",
                    cross_validation=_cv_txt,
                    seed=seed if seed is not None else "not fixed (results vary between runs)",
                    notes=description or "Generated by Multi-Fidelity Surrogate Model page.",
                    conformal_intervals=conformal_intervals if use_conformal else None,
                    conformal_alpha=conformal_alpha_sel if use_conformal else None,
                    imputation_summary=_imputation_summary,
                    input_names=list(input_cols),
                    calibration_residuals=residuals,
                    X_design={lv: level_data[lv][0] for lv in level_names},
                    summary_points=summary_points,
                    level_summary=level_rows,
                    baseline_comparison=baseline_rows,
                    baseline_label=_base_label,
                    grid_search_results=gs_sorted,
                    best_candidate_name=best_name,
                )
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")
            st.stop()

        st.download_button(
            "Download HTML Report",
            html,
            file_name="mf_model_report.html",
            mime="text/html",
        )
