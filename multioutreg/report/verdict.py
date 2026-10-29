# Copyright (c) 2026 takotime808

"""Rules that turn report metrics into a verdict, findings and next steps.

Every status is one of ``"good"``, ``"warn"``, ``"crit"`` or ``"neutral"``
(no gain, which does not lower the overall status), or ``None`` when the
criterion does not apply. The thresholds are printed in the report's appendix
so a reader can check how each verdict was reached.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

_ORDER = {None: -1, "good": 0, "neutral": 0, "warn": 1, "crit": 2}


@dataclass(frozen=True)
class Thresholds:
    """Pass / caution limits used by :func:`judge_output`.

    ``mf_gain`` and ``mf_hurt`` are RMSE changes in percent against the
    high-fidelity-only baseline (negative is better). Coverage passes when it is
    no further below its nominal level than ``z`` binomial standard errors for
    the test-set size, and gets a caution within twice that.
    """

    r2_good: float = 0.90
    r2_warn: float = 0.80
    mf_gain: float = -5.0
    mf_hurt: float = 5.0
    coverage_target: float = 0.95
    z: float = 1.96
    weak_level_r: float = 0.5
    small_test: int = 30

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def coverage_band(p: float, n: int, z: float = 1.96) -> float:
    """Half-width of the range of coverage expected from sampling noise alone."""
    if n <= 0:
        return float("nan")
    return z * math.sqrt(p * (1 - p) / n)


def worst(*statuses: Optional[str]) -> Optional[str]:
    """The most severe status; ``neutral`` counts as ``good``."""
    present = [s for s in statuses if s is not None]
    if not present:
        return None
    w = max(present, key=lambda s: _ORDER[s])
    return "good" if w == "neutral" else w


def coverage_status(coverage: Optional[float], nominal: float, n: int,
                    th: Thresholds = Thresholds()) -> Optional[str]:
    """Pass at or above ``nominal - band``, caution within ``2 * band``.

    Coverage above nominal passes: wide intervals are safe, and their cost is
    shown separately as the interval width.
    """
    if coverage is None or not np.isfinite(coverage):
        return None
    b = coverage_band(nominal, n, th.z)
    if coverage >= nominal - b:
        return "good"
    if coverage >= nominal - 2 * b:
        return "warn"
    return "crit"


def judge_output(o: Dict[str, Any], n_test: int, th: Thresholds = Thresholds()) -> Dict[str, Any]:
    """Status per criterion for one output (a dict from ``build_report_data``).

    Keys: ``accuracy`` (test R²), ``gaussian`` and ``conformal`` (coverage of
    each kind of interval), ``bars`` (the error bars a user should report:
    Gaussian if it passes, else conformal as a caution), ``mf`` (RMSE vs the
    HF-only baseline) and ``overall`` (the worst of accuracy, bars and mf).
    """
    r2 = o.get("r2")
    acc = None
    if r2 is not None and np.isfinite(r2):
        acc = "good" if r2 >= th.r2_good else "warn" if r2 >= th.r2_warn else "crit"

    if not o.get("has_std", True):
        g = "crit"
    else:
        g = coverage_status(o.get("gauss95"), th.coverage_target, n_test, th)
    conf = o.get("conformal") or {}
    c = coverage_status(conf.get("coverage"), conf.get("nominal", th.coverage_target), n_test, th)
    if g == "good" or c is None:
        bars = g
    elif c == "good":
        bars = "warn"
    else:
        bars = worst(g, c)

    change = o.get("mf_change_pct")
    mf = None
    if change is not None and np.isfinite(change):
        mf = "good" if change <= th.mf_gain else "warn" if change >= th.mf_hurt else "neutral"

    return {"accuracy": acc, "gaussian": g, "conformal": c, "bars": bars, "mf": mf,
            "overall": worst(acc, bars, mf) or "neutral"}


READY_LABEL = {"good": "Ready", "neutral": "Ready", "warn": "Use with care", "crit": "Not ready"}


def _names(items: List[str]) -> str:
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _pct(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f}%"


def _reasons(o: Dict[str, Any], th: Thresholds) -> List[str]:
    s = o["status"]
    out = []
    if s["accuracy"] in ("warn", "crit"):
        out.append(f"its test R² is {o['r2']:.2f} (target ≥ {th.r2_good:.2f})")
    if not o.get("has_std", True):
        out.append("the model gives no predictive uncertainty")
    elif s["gaussian"] in ("warn", "crit"):
        txt = f"its Gaussian 95% error bars cover only {_pct(o['gauss95'])} of test points"
        conf = o.get("conformal")
        if conf and conf.get("coverage") is not None:
            txt += (f", while conformal {_pct(conf['nominal'])} intervals cover "
                    f"{_pct(conf['coverage'])}")
        out.append(txt)
    if s["mf"] == "warn":
        out.append(f"the low-fidelity data makes it {o['mf_change_pct']:+.0f}% worse than an "
                   "HF-only model")
    return out


def build_verdict(outputs: List[Dict[str, Any]], th: Thresholds = Thresholds()) -> Dict[str, Any]:
    """Headline, explanation and overall status from each output's ``status``."""
    groups: Dict[str, List[str]] = {"good": [], "warn": [], "crit": []}
    for o in outputs:
        overall = o["status"]["overall"]
        groups["good" if overall == "neutral" else overall].append(o["name"])
    if groups["good"] and not groups["warn"] and not groups["crit"]:
        head = ("Ready to use" if len(outputs) == 1
                else f"Ready for all {len(outputs)} outputs" if len(outputs) > 2
                else f"Ready for {_names(groups['good'])}") + "."
    else:
        head = f"Ready for {_names(groups['good'])}." if groups["good"] else "Not ready."
        if groups["warn"]:
            head += f" Use {_names(groups['warn'])} with care."
        if groups["crit"]:
            head += f" Do not use {_names(groups['crit'])} yet."
    why = ""
    flagged = [o for o in outputs if o["status"]["overall"] in ("crit", "warn")]
    flagged.sort(key=lambda o: -_ORDER[o["status"]["overall"]])
    if flagged:
        o = flagged[0]
        rs = _reasons(o, th)
        why = f"{o['name']}: " + (rs[0] if len(rs) == 1 else ", ".join(rs[:-1]) + ", and " + rs[-1]) + "."
        if len(flagged) > 1:
            why += f" {len(flagged) - 1} more output(s) are flagged; see the scorecard."
    status = worst(*[o["status"]["overall"] for o in outputs]) or "good"
    return {"headline": head, "why": why, "status": status,
            "groups": groups}


def _tied(cv: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not cv or not cv.get("rows"):
        return []
    ok = [r for r in cv["rows"] if r.get("mean") is not None]
    best = cv.get("best") or (ok[0]["name"] if ok else None)
    b = next((r for r in ok if r["name"] == best), None)
    if b is None:
        return []
    lim = b["mean"] + (b.get("sd") or 0.0)
    return [r for r in ok if r["mean"] - (r.get("sd") or 0.0) <= lim]


def build_findings(data: Dict[str, Any], th: Thresholds = Thresholds(),
                   limit: int = 4) -> List[Dict[str, Any]]:
    """Claim-and-evidence cards for the front page, most severe first.

    Each card is ``{claim, detail, status, section, plot}`` where ``plot`` names
    the mini chart (``"dumbbell"``, ``"calib:<i>"``, ``"parity:<i>"``, ...).
    """
    outputs = data["outputs"]
    n_test = data["run"]["n_test_points"]
    cards: List[Dict[str, Any]] = []

    with_mf = [o for o in outputs if o["status"]["mf"] is not None]
    if with_mf:
        best = min(with_mf, key=lambda o: o["mf_change_pct"])
        helped = [o["name"] for o in with_mf if o["status"]["mf"] == "good"]
        hurt = [o["name"] for o in with_mf if o["status"]["mf"] == "warn"]
        if helped:
            claim = f"Low-fidelity data cut {best['name']}'s error by {-best['mf_change_pct']:.0f}%"
            rest = [o["name"] for o in with_mf if o["name"] not in helped]
            detail = f"RMSE falls for {_names(helped)} compared with an HF-only model."
            if rest:
                detail += f" {_names(rest)}: within ±{th.mf_hurt:.0f}% or worse."
            cards.append({"claim": claim, "detail": detail, "status": "good",
                          "section": "sec-4", "plot": "dumbbell"})
        else:
            claim = ("The low-fidelity data makes the model worse" if hurt
                     else "The low-fidelity data gives no clear gain")
            detail = (f"Test RMSE is within ±{th.mf_hurt:.0f}% of an HF-only model"
                      + (f", and worse for {_names(hurt)}." if hurt else " for every output.")
                      + " The simpler HF-only model is a reasonable choice.")
            cards.append({"claim": claim, "detail": detail, "status": "warn",
                          "section": "sec-4", "plot": "dumbbell"})

    bad_bars = [o for o in outputs if o["status"]["gaussian"] in ("warn", "crit")]
    if bad_bars:
        o = min(bad_bars, key=lambda o: o.get("gauss95") or 0.0)
        if not o.get("has_std", True):
            claim, detail = (f"{o['name']} has no error bars",
                             "The model reports zero predictive std. Use conformal intervals.")
        else:
            claim = f"{o['name']}'s Gaussian error bars are too narrow"
            detail = f"They cover {_pct(o['gauss95'])} of test points at 95%."
            conf = o.get("conformal")
            if conf and conf.get("coverage") is not None:
                detail += (f" Conformal {_pct(conf['nominal'])} intervals cover "
                           f"{_pct(conf['coverage'])}"
                           + (", inside the expected band." if o["status"]["conformal"] == "good"
                              else "."))
            else:
                detail += " Enable conformal intervals for calibrated error bars."
        cards.append({"claim": claim, "detail": detail, "status": o["status"]["gaussian"],
                      "section": "sec-3", "plot": f"calib:{o['idx']}"})
    elif any(o.get("has_std", True) for o in outputs):
        b = coverage_band(th.coverage_target, n_test, th.z)
        cards.append({
            "claim": "The error bars hold for every output",
            "detail": (f"Gaussian 95% intervals cover at least {_pct(th.coverage_target - b)} "
                       f"of test points, the lower end of the range expected for n = {n_test}."),
            "status": "good", "section": "sec-3", "plot": "cov95"})

    lo = min(outputs, key=lambda o: o["r2"] if o["r2"] is not None else -np.inf)
    if lo["status"]["accuracy"] in ("warn", "crit"):
        cards.append({
            "claim": f"{lo['name']} is the least accurate output",
            "detail": (f"Test R² {lo['r2']:.3f}, below the {th.r2_good:.2f} target. More "
                       "training runs where its predicted std is largest would help most."),
            "status": lo["status"]["accuracy"], "section": f"out-{lo['idx']}",
            "plot": f"parity:{lo['idx']}"})
    elif lo["r2"] is not None:
        cards.append({
            "claim": (f"{lo['name']} is accurate (R² {lo['r2']:.3f})" if len(outputs) == 1
                      else f"Every output reaches R² ≥ {th.r2_good:.2f}"),
            "detail": (f"The lowest test R² is {lo['r2']:.3f} ({lo['name']})."
                       if len(outputs) > 1 else "Predictions follow the observed values closely."),
            "status": "good", "section": f"out-{lo['idx']}", "plot": f"parity:{lo['idx']}"})

    biased = [o for o in outputs if (o.get("trend") or {}).get("significant")]
    if biased:
        o = biased[0]
        t = o["trend"]
        cards.append({
            "claim": f"{o['name']} has a systematic bias",
            "detail": (f"Residuals change with the prediction (slope {t['slope']:.2f}, "
                       f"p = {t['p']:.3g}): the model {'over' if t['slope'] < 0 else 'under'}"
                       f"predicts high values of {o['name']}."),
            "status": "warn", "section": f"out-{o['idx']}", "plot": f"resid:{o['idx']}"})

    weak = _weak_levels(data, th)
    if weak:
        lv, outs = next(iter(weak.items()))
        cards.append({
            "claim": f"Level '{lv}' says little about {_names(outs)}",
            "detail": (f"A model trained on '{lv}' alone has |r| < {th.weak_level_r} with the "
                       f"high-fidelity truth for {_names(outs)}."),
            "status": "warn", "section": "sec-4", "plot": "quadrant"})

    tied = _tied(data.get("cv"))
    if len(tied) > 1:
        cards.append({
            "claim": f"The top {len(tied)} candidate models are statistically tied",
            "detail": (f"Their CV scores overlap within one fold standard deviation. "
                       f"{tied[0]['name']} scored best; a simpler tied model would also be "
                       "a reasonable choice."),
            "status": "neutral", "section": "sec-7", "plot": "cv"})

    reg = data.get("regret") or {}
    failed = [r["label"] for r in reg.get("rows", []) if r.get("status") == "crit"]
    if failed:
        cards.append({
            "claim": "The predicted Pareto front misses its regret bound",
            "detail": f"{_names(failed)} exceed the maximum you set.",
            "status": "crit", "section": "sec-2", "plot": None})

    cards.sort(key=lambda c: -_ORDER[c["status"]])
    return cards[:limit]


def _weak_levels(data: Dict[str, Any], th: Thresholds) -> Dict[str, List[str]]:
    weak: Dict[str, List[str]] = {}
    for lv in data.get("levels") or []:
        if lv.get("is_highest"):
            continue
        for name, v in lv["outputs"].items():
            if v.get("error") is None and (v.get("r") is None or abs(v["r"]) < th.weak_level_r):
                weak.setdefault(lv["level"], []).append(name)
    return weak


def build_next_steps(data: Dict[str, Any], th: Thresholds = Thresholds()) -> List[Dict[str, str]]:
    """Actions that follow from the statuses, each tagged with its section."""
    outputs = data["outputs"]
    n_test = data["run"]["n_test_points"]
    steps: List[Dict[str, str]] = []

    use_conf = [o["name"] for o in outputs
                if o["status"]["gaussian"] in ("warn", "crit") and o["status"]["conformal"] == "good"]
    need_conf = [o["name"] for o in outputs
                 if o["status"]["gaussian"] in ("warn", "crit") and o["status"]["conformal"] is None]
    if use_conf:
        steps.append({"html": f"<b>Report conformal intervals for {_names(use_conf)}.</b> "
                      "Their Gaussian error bars are too narrow; the conformal ones hold.",
                      "section": "Section 3"})
    if need_conf:
        steps.append({"html": f"<b>Turn on conformal intervals.</b> The Gaussian error bars for "
                      f"{_names(need_conf)} are too narrow, and conformal intervals give "
                      "distribution-free coverage.", "section": "Section 3"})
    weak_acc = [o["name"] for o in outputs if o["status"]["accuracy"] in ("warn", "crit")]
    if weak_acc:
        steps.append({"html": f"<b>Add training runs for {_names(weak_acc)}</b> where the "
                      "predicted std is largest. <code>multioutreg.active.ActiveLearner</code> "
                      "can pick the points.", "section": "Section 2"})
    biased = [o["name"] for o in outputs if (o.get("trend") or {}).get("significant")]
    if biased:
        steps.append({"html": f"<b>Check {_names(biased)} for systematic bias.</b> The residuals "
                      "trend with the prediction; a missing input or a transform of the output "
                      "may fix it.", "section": "Section 2"})
    for lv, outs in _weak_levels(data, th).items():
        steps.append({"html": f"<b>Consider dropping level '{lv}'</b> for {_names(outs)}: a "
                      "model trained on it alone barely tracks the high-fidelity truth.", "section": "Section 4"})
    no_gain = [o["name"] for o in outputs if o["status"]["mf"] in ("neutral", "warn")]
    if no_gain:
        steps.append({"html": f"<b>For {_names(no_gain)}, an HF-only model is as good or "
                      "better</b> and is simpler to maintain.", "section": "Section 4"})
    if n_test < th.small_test:
        b = coverage_band(th.coverage_target, n_test, th.z)
        steps.append({"html": f"<b>Hold out more test points.</b> With {n_test}, coverage is "
                      f"uncertain by about ±{100 * b:.0f} percentage points.",
                      "section": "Section 6"})
    reg = data.get("regret") or {}
    if any(r.get("status") == "crit" for r in reg.get("rows", [])):
        steps.append({"html": "<b>Improve the outputs that drive the Pareto front</b> before "
                      "using the model for trade-off studies; the regret bound is not met.",
                      "section": "Section 2"})
    if not steps:
        steps.append({"html": "<b>Use the model.</b> Re-run this report when new data "
                      "arrives to check that the verdict still holds.", "section": ""})
    return steps


def summary_points(
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
    """Plain-language findings derived from the computed metrics.

    Used by the multi-fidelity page (for its on-screen Summary and the classic
    report). ``best_cand`` needs ``name`` and ``mf_type``; ``gs_rows`` are the
    grid-search rows with ``Model`` and ``CV MSE``.
    """
    pts: List[str] = []

    ok = sorted(
        (r for r in gs_rows if r["CV MSE"] is not None), key=lambda r: r["CV MSE"]
    )
    msg = (
        f"Best configuration: {best_cand['name']} "
        f"(CV MSE {ok[0]['CV MSE']:.4g}, chosen from {len(ok)} working "
        f"candidates"
    )
    if len(ok) > 1 and ok[0]["CV MSE"] > 0:
        gap = 100.0 * (ok[1]["CV MSE"] - ok[0]["CV MSE"]) / ok[0]["CV MSE"]
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
                "multi-fidelity model beat high-fidelity data alone in cross-validation "
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
