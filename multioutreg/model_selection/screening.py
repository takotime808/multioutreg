# Copyright (c) 2026 takotime808

"""Statistical pre-screening tests for model selection.

Runs inexpensive statistical tests on (X, Y) *before* training any model so that
unsuitable or computationally expensive models can be skipped.

Six tests are provided:

1. **Sample size** — rule-based thresholds gate GP (O(N³)), SVR, MLP, and bootstrap LR.
2. **Normality** (Shapiro-Wilk / D'Agostino K² on OLS residuals) — non-normal
   residuals suggest tree-based or non-linear models.
3. **Linearity** (Ramsey RESET) — detected non-linearity activates RF, GB, MLP, DT.
4. **Heteroscedasticity** (Breusch-Pagan) — input-dependent variance activates quantile
   GB and heteroscedastic ensemble wrappers. Only run when RESET passes; otherwise
   it is reported as not tested and those models are kept.
5. **Multicollinearity** (VIF) — flags unstable linear model fits; adds BayesianRidge.

Tests 2–4 are hypothesis tests run per output; their p-values are corrected
jointly (Holm by default, see ``ModelScreener(correction=...)``).
6. **Non-linear dependency** (cross-validated RF–LR R² gain) — confirms whether a
   non-linear model actually outperforms linear regression on this data.

Quick usage::

    from multioutreg.model_selection.screening import ModelScreener

    screener = ModelScreener().fit(X_train, Y_train)
    specs = screener.screen()                    # global candidate list
    matrix = screener.screen_per_output(names)   # per-output boolean schedule
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

warnings.filterwarnings("ignore")


# ── TestResult ─────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    """Result of a single statistical test."""
    name: str
    passed: bool        # True = null hypothesis *not* rejected (no problem detected)
    statistic: float
    p_value: Optional[float]
    message: str
    alpha: float = 0.05
    p_adjusted: Optional[float] = None  # set by ModelScreener's multiple-testing correction

    def __repr__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        p_str = f"p={self.p_value:.4f}" if self.p_value is not None else ""
        if self.p_adjusted is not None:
            p_str += f" p_adj={self.p_adjusted:.4f}"
        return f"[{status}] {self.name} | stat={self.statistic:.4f} {p_str} — {self.message}"


# ── Individual tests ────────────────────────────────────────────────────────────

def test_sample_size(X: np.ndarray) -> Dict[str, TestResult]:
    """Rule-based sample-size thresholds — no p-value, purely deterministic.

    Returns a dict with keys:
    - ``gp_feasible``       — N < 300  (GP cost O(N³))
    - ``svr_feasible``      — N < 2000 (SVR kernel matrix O(N²))
    - ``mlp_feasible``      — N > 100  (MLP needs sufficient data)
    - ``bootstrap_needed``  — N < 500  (bootstrap LR adds value on small N)
    - ``knn_ratio_ok``      — N/p > 10 (KNN degrades in high dimensions)
    - ``pbr_feasible``      — p < 20   (degree-2 expansion of 20+ features is expensive)
    """
    n, p = X.shape[0], X.shape[1]
    return {
        "gp_feasible": TestResult(
            "GP feasible (N<300)", n < 300, float(n), None,
            f"N={n}; GP cost O(N³), skip above 300"),
        "svr_feasible": TestResult(
            "SVR feasible (N<2000)", n < 2000, float(n), None,
            f"N={n}; SVR kernel matrix grows O(N²)"),
        "mlp_feasible": TestResult(
            "MLP feasible (N>100)", n > 100, float(n), None,
            f"N={n}; MLP needs enough samples to generalise"),
        "bootstrap_needed": TestResult(
            "Bootstrap useful (N<500)", n < 500, float(n), None,
            f"N={n}; bootstrap LR adds uncertainty value on small datasets"),
        "knn_ratio_ok": TestResult(
            "KNN ratio ok (N/p>10)", n / p > 10, float(n / p), None,
            f"N/p={n/p:.1f}; KNN degrades in high dimensions"),
        "pbr_feasible": TestResult(
            "PBR feasible (p<20)", p < 20, float(p), None,
            f"p={p}; degree-2 PBR creates C(p+2,2)={int((p+2)*(p+1)//2)} features"),
    }


def _ols_residuals(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    import statsmodels.api as sm

    return np.asarray(sm.OLS(y, sm.add_constant(X)).fit().resid)


def test_normality(X: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> TestResult:
    """Shapiro-Wilk (N ≤ 5000) or D'Agostino K² (N > 5000) on OLS residuals.

    The assumption that matters is normality of the errors, not of y: a skewed
    or multimodal y can still leave perfectly normal residuals after a linear fit.

    PASS = residuals are consistent with normality.
    FAIL = significant non-normality; tree/non-linear models are recommended.
    """
    from scipy import stats as _stats

    y = np.asarray(y).ravel()
    resid = _ols_residuals(X, y)
    if len(resid) <= 5000:
        stat, p = _stats.shapiro(resid)
        name = "Normality of residuals (Shapiro-Wilk)"
    else:
        stat, p = _stats.normaltest(resid)
        name = "Normality of residuals (D'Agostino K²)"
    passed = p >= alpha
    msg = ("residuals consistent with normality"
           if passed else "significant non-normality detected")
    return TestResult(name, passed, float(stat), float(p), msg, alpha)


def test_linearity(X: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> TestResult:
    """Ramsey RESET test for functional-form misspecification.

    Fits OLS then checks whether powers of the fitted values improve the fit.
    PASS = linear model is adequate.
    FAIL = non-linearity detected; tree/MLP models are likely beneficial.
    """
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import linear_reset
    from scipy import stats as _stats

    y = np.asarray(y).ravel()
    Xc = sm.add_constant(X)
    ols = sm.OLS(y, Xc).fit()
    try:
        result = linear_reset(ols, power=3, test_type="fitted", use_f=True)
        stat, p = float(result.statistic), float(result.pvalue)
    except Exception:
        # Fallback: Spearman–Pearson rank correlation gap
        pearson_r = float(np.corrcoef(ols.fittedvalues, y)[0, 1])
        spearman_r, _ = _stats.spearmanr(ols.fittedvalues, y)
        gap = abs(float(spearman_r) - pearson_r)
        passed = gap < 0.05
        return TestResult(
            "Linearity (Spearman–Pearson gap)", passed, gap, None,
            "linear" if passed else "non-linearity indicated by rank-correlation gap",
            alpha)
    passed = p >= alpha
    msg = ("linear model is adequate"
           if passed else "non-linearity detected — tree/MLP models recommended")
    return TestResult("Linearity (RESET)", passed, stat, p, msg, alpha)


def test_heteroscedasticity(
        X: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> TestResult:
    """Breusch-Pagan test: regresses squared OLS residuals on features.

    Detects heteroscedasticity where **variance** is a linear function of the
    regressors (e.g. ``var(e | x) = a + b·x``). Only valid when the linear mean
    is adequate -- see :func:`heteroscedasticity_not_tested`.

    PASS = no evidence of heteroscedasticity.
    FAIL = input-dependent variance; quantile GB or heteroscedastic ensembles are
           indicated.
    """
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan

    y = np.asarray(y).ravel()
    lm, lm_p, _, _ = het_breuschpagan(_ols_residuals(X, y), sm.add_constant(X))
    passed = lm_p >= alpha
    msg = ("homoscedastic — constant variance assumption holds"
           if passed else
           "heteroscedastic — noise variance depends on inputs; "
           "quantile GB or EnsembleHeteroscedastic recommended")
    return TestResult(
        "Heteroscedasticity (Breusch-Pagan)", passed,
        float(lm), float(lm_p), msg, alpha)


def heteroscedasticity_not_tested(alpha: float = 0.05) -> TestResult:
    """Placeholder result used instead of Breusch-Pagan when RESET fails.

    With a misspecified mean the residuals contain the missed signal, whose
    squares track x even under constant noise, so Breusch-Pagan rejects for the
    wrong reason. Residuals of a flexible fit do not fix this (their bias also
    varies with x; 15-95% false rejections in simulation). The result is
    reported as FAIL so heteroscedastic models are kept rather than ruled out.
    """
    return TestResult(
        "Heteroscedasticity (not tested)", False, float("nan"), None,
        "not tested: mean is non-linear (RESET failed), so Breusch-Pagan cannot "
        "separate variance from misfit; heteroscedastic models kept eligible",
        alpha)


def test_multicollinearity(
        X: np.ndarray, threshold: float = 10.0) -> TestResult:
    """Variance Inflation Factor (VIF) for each feature.

    PASS = max VIF < threshold.
    FAIL = severe collinearity; linear model coefficients are unstable.
    """
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    Xc = sm.add_constant(X)
    vifs = [variance_inflation_factor(Xc, i) for i in range(1, Xc.shape[1])]
    max_vif = float(np.max(vifs))
    passed = max_vif < threshold
    msg = (f"max VIF={max_vif:.1f} — no severe multicollinearity"
           if passed else
           f"max VIF={max_vif:.1f} — consider feature selection or PCA "
           "before linear models")
    return TestResult("Multicollinearity (VIF)", passed, max_vif, None, msg)


def test_nonlinear_dependency(
        X: np.ndarray, y: np.ndarray,
        gain_threshold: float = 0.05,
        cv: int = 3) -> TestResult:
    """Cross-validated RF vs LinearRegression R² gain.

    If a small RandomForest outperforms LinearRegression by more than
    *gain_threshold* in CV R², non-linear structure is confirmed.

    Preferred over the MI-Pearson gap because it uses the full feature set in
    both models and directly answers: *does a non-linear model predict better?*

    PASS = RF and LR perform similarly (linear structure sufficient).
    FAIL = RF outperforms LR; RF / GB / MLP are recommended.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    y = np.asarray(y).ravel()
    lr_r2 = float(np.mean(cross_val_score(
        LinearRegression(), X, y, cv=cv, scoring="r2")))
    rf_r2 = float(np.mean(cross_val_score(
        RandomForestRegressor(n_estimators=20, max_depth=5, random_state=0),
        X, y, cv=cv, scoring="r2")))
    gain = rf_r2 - lr_r2
    passed = gain < gain_threshold
    msg = (f"RF-LR R² gain={gain:.3f} (LR={lr_r2:.3f}, RF={rf_r2:.3f}) — "
           + ("linear model captures structure well"
              if passed else
              "non-linear model outperforms; RF / GB / MLP recommended"))
    return TestResult(
        "Non-linear dependency (RF–LR R² gain)", passed, float(gain), None, msg)


# ── ModelScreener ───────────────────────────────────────────────────────────────

class ModelScreener:
    """Run all pre-screening tests and produce a model candidate schedule.

    Parameters
    ----------
    alpha : float
        Significance level for all hypothesis tests (default 0.05).
    gain_threshold : float
        Minimum RF–LR R² gain to flag data as non-linear (default 0.05).
    vif_threshold : float
        VIF above which multicollinearity is flagged (default 10.0).
    correction : {"holm", "bh", None}
        Multiple-testing correction applied across the normality, linearity and
        heteroscedasticity tests of all outputs (3 × n_outputs hypotheses).
        ``"holm"`` controls the family-wise error rate, ``"bh"``
        (Benjamini–Hochberg) the false discovery rate, ``None`` disables it.

    Examples
    --------
    >>> screener = ModelScreener().fit(X_train, Y_train)
    >>> specs = screener.screen()          # global list of ModelSpec
    >>> matrix = screener.screen_per_output(output_names)  # per-output DataFrame
    """

    def __init__(self,
                 alpha: float = 0.05,
                 gain_threshold: float = 0.05,
                 vif_threshold: float = 10.0,
                 correction: Optional[str] = "holm") -> None:
        if correction not in _CORRECTION_METHODS:
            raise ValueError(
                f"correction must be one of {sorted(k for k in _CORRECTION_METHODS if k)}"
                f" or None, got {correction!r}")
        self.alpha = alpha
        self.gain_threshold = gain_threshold
        self.vif_threshold = vif_threshold
        self.correction = correction

        self._size: Dict[str, TestResult] = {}
        self._vif: Optional[TestResult] = None
        self._per_output: List[Dict[str, TestResult]] = []
        self.global_flags_: Dict[str, bool] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, Y: np.ndarray) -> "ModelScreener":
        """Run all tests.  Call before ``screen()`` or ``screen_per_output()``."""
        X = np.asarray(X)
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        self._size = test_sample_size(X)
        self._vif = test_multicollinearity(X, self.vif_threshold)

        self._per_output = []
        for j in range(Y.shape[1]):
            y_j = Y[:, j]
            linearity = test_linearity(X, y_j, self.alpha)
            # Breusch-Pagan is only valid when the linear mean is adequate. Uses
            # the uncorrected RESET decision, which errs towards not testing.
            if linearity.passed:
                hetero = test_heteroscedasticity(X, y_j, self.alpha)
            else:
                hetero = heteroscedasticity_not_tested(self.alpha)
            self._per_output.append({
                "normality":          test_normality(X, y_j, self.alpha),
                "linearity":          linearity,
                "heteroscedasticity": hetero,
                "nonlinear_dep":      test_nonlinear_dependency(
                    X, y_j, self.gain_threshold),
            })
        self._apply_correction()

        self.global_flags_ = {
            "any_nonlinear":       any(not r["linearity"].passed
                                       for r in self._per_output),
            "any_nonnormal":       any(not r["normality"].passed
                                       for r in self._per_output),
            "any_heteroscedastic": any(not r["heteroscedasticity"].passed
                                       for r in self._per_output),
            "any_nonlinear_dep":   any(not r["nonlinear_dep"].passed
                                       for r in self._per_output),
            "multicollinear":      not self._vif.passed,
            "outputs_heterogeneous": (
                len({r["linearity"].passed for r in self._per_output}) > 1
                or len({r["heteroscedasticity"].passed for r in self._per_output}) > 1
            ),
        }
        return self

    def _apply_correction(self) -> None:
        """Adjust the p-values of all hypothesis tests jointly and update ``passed``.

        Without this, 3 tests × n_outputs at alpha=0.05 OR'ed together flag
        "a problem" on pure-noise data far more often than 5% of the time.
        """
        method = _CORRECTION_METHODS[self.correction]
        if method is None:
            return
        from statsmodels.stats.multitest import multipletests

        results = [r[key] for r in self._per_output for key in _HYPOTHESIS_TESTS
                   if r[key].p_value is not None]
        if not results:
            return
        reject, p_adj, _, _ = multipletests(
            [r.p_value for r in results], alpha=self.alpha, method=method)
        for r, rej, pa in zip(results, reject, p_adj):
            r.p_adjusted = float(pa)
            # Adjusted p >= raw p, so the only possible change is FAIL -> PASS.
            if not r.passed and not rej:
                r.passed = True
                r.message = (f"not significant after {self.correction} correction "
                             f"(p={r.p_value:.4f}, p_adj={pa:.4f})")

    # --------------------------------------------------------------- screen
    def screen(self) -> List["ModelSpec"]:
        """Return global list of :class:`ModelSpec` objects.

        Models whose factory is ``None`` were skipped; the ``reason_excluded``
        field explains why.
        """
        s = self._size
        g = self.global_flags_
        specs: List[ModelSpec] = []

        def _add(name, factory, cost, reason):
            specs.append(ModelSpec(name, factory, cost, reason_included=reason))

        # always include cheap baselines
        lr_reason = "always included as cheap baseline"
        if g["multicollinear"]:
            lr_reason += (f"; max VIF={self._vif.statistic:.1f} — coefficients are "
                          "unstable, compare against BayesianRidge")
        _add("LinearRegression", lambda: _lr(), "low", lr_reason)
        if g["multicollinear"]:
            _add("BayesianRidge",
                 lambda: _br(), "low",
                 f"max VIF={self._vif.statistic:.1f} ≥ {self.vif_threshold:g} — "
                 "regularised linear model is stable under multicollinearity")
        else:
            specs.append(ModelSpec(
                "BayesianRidge", None, "low",
                reason_excluded=f"max VIF={self._vif.statistic:.1f} — no severe "
                                "multicollinearity, plain LR is stable"))
        _add("DecisionTree",
             lambda: _dt(), "low",
             "always included — cheap, interpretable non-linear baseline")

        # size-gated: GP
        if s["gp_feasible"].passed:
            _add("GaussianProcess",
                 lambda: _gp(), "high",
                 f"N={int(s['gp_feasible'].statistic)} < 300 — GP is feasible")
        else:
            specs.append(ModelSpec(
                "GaussianProcess", None, "high",
                reason_excluded=f"N={int(s['gp_feasible'].statistic)} ≥ 300 — O(N³) cost"))

        # size-gated: SVR
        if s["svr_feasible"].passed:
            _add("SVR",
                 lambda: _svr(), "medium",
                 f"N={int(s['svr_feasible'].statistic)} < 2000 — SVR is feasible")
        else:
            specs.append(ModelSpec(
                "SVR", None, "medium",
                reason_excluded=f"N={int(s['svr_feasible'].statistic)} ≥ 2000 — kernel matrix too large"))

        # size-gated: KNN
        if s["knn_ratio_ok"].passed:
            _add("KNN",
                 lambda: _knn(), "low",
                 f"N/p={s['knn_ratio_ok'].statistic:.1f} > 10")
        else:
            specs.append(ModelSpec(
                "KNN", None, "low",
                reason_excluded=f"N/p={s['knn_ratio_ok'].statistic:.1f} — too few samples per dimension"))

        # size-gated: bootstrap LR
        if s["bootstrap_needed"].passed:
            _add("BootstrapLinearRegression",
                 lambda: _blr(), "low",
                 f"N={int(s['bootstrap_needed'].statistic)} < 500 — bootstrap adds value on small N")
        else:
            specs.append(ModelSpec(
                "BootstrapLinearRegression", None, "low",
                reason_excluded=f"N={int(s['bootstrap_needed'].statistic)} ≥ 500 — plain LR sufficient"))

        # non-linearity gated: RF, GB, MLP
        needs_nonlin = (g["any_nonlinear"] or g["any_nonnormal"]
                        or g["any_nonlinear_dep"])
        if needs_nonlin:
            reasons = []
            if g["any_nonlinear"]: reasons.append("RESET detected non-linearity")
            if g["any_nonnormal"]: reasons.append("non-normal residuals")
            if g["any_nonlinear_dep"]: reasons.append("RF beats LR in CV R²")
            reason_nl = "; ".join(reasons)

            _add("RandomForest",      lambda: _rf(),  "medium", reason_nl)
            _add("GradientBoosting",  lambda: _gb(),  "medium", reason_nl)

            if s["mlp_feasible"].passed:
                _add("MLP",
                     lambda: _mlp(), "high",
                     reason_nl + f"; N={int(s['mlp_feasible'].statistic)} > 100")
            else:
                specs.append(ModelSpec(
                    "MLP", None, "high",
                    reason_excluded=f"N={int(s['mlp_feasible'].statistic)} ≤ 100 — too few samples for MLP"))
        else:
            for m in ("RandomForest", "GradientBoosting", "MLP"):
                specs.append(ModelSpec(
                    m, None, "medium" if m != "MLP" else "high",
                    reason_excluded="data is linear with normal residuals and RF does "
                                    "not beat LR — heavy non-linear models not needed"))

        # heteroscedasticity gated
        if g["any_heteroscedastic"]:
            detected = any(r["heteroscedasticity"].p_value is not None
                           and not r["heteroscedasticity"].passed
                           for r in self._per_output)
            why = ("Breusch-Pagan detected heteroscedasticity" if detected else
                   "heteroscedasticity not testable (non-linear mean)")
            _add("GradientBoosting_Quantile",
                 lambda: _gbq(), "high",
                 f"{why} — quantile regression appropriate")
            _add("EnsembleHeteroscedastic",
                 lambda: _ehet(), "high",
                 f"{why} — bootstrap ensemble captures input-dependent noise")
        else:
            for m in ("GradientBoosting_Quantile", "EnsembleHeteroscedastic"):
                specs.append(ModelSpec(
                    m, None, "high",
                    reason_excluded="homoscedastic data — heteroscedastic models not needed"))

        return specs

    # -------------------------------------------------- screen_per_output
    def screen_per_output(
            self, output_names: Optional[List[str]] = None) -> "pd.DataFrame":
        """Return a boolean DataFrame (index=model, columns=output).

        ``True`` = run this model on this output; ``False`` = skip.
        Requires :func:`fit` to have been called first.
        """
        import pandas as pd

        if not self._per_output:
            raise RuntimeError("Call fit() before screen_per_output().")

        n_out = len(self._per_output)
        names = output_names or [f"y{j}" for j in range(n_out)]

        records: Dict[str, Dict[str, bool]] = {}
        for j, oname in enumerate(names):
            records[oname] = _eligible_for_output_j(
                self._per_output[j], self._size)

        return pd.DataFrame(records)

    # -------------------------------------------------------- eligibility
    def eligible_indices_for_output(
            self, output_idx: int, model_names: List[str]) -> List[int]:
        """Return indices into *model_names* that should be tried for *output_idx*.

        Used by :class:`~multioutreg.model_selection.AutoDetectMultiOutputRegressor`
        to filter its estimator list per output when ``pre_screen=True``.
        """
        if not self._per_output:
            raise RuntimeError("Call fit() first.")
        rules = _eligible_for_output_j(
            self._per_output[output_idx], self._size)
        unknown = [name for name in model_names if name not in rules]
        if unknown:
            warnings.warn(
                f"No screening rule for model(s) {unknown}; they are always tried.",
                UserWarning, stacklevel=2)
        return [i for i, name in enumerate(model_names) if rules.get(name, True)]


# ── helpers ────────────────────────────────────────────────────────────────────

# Tests whose p-values are adjusted jointly by ModelScreener._apply_correction.
_HYPOTHESIS_TESTS = ("normality", "linearity", "heteroscedasticity")
_CORRECTION_METHODS = {"holm": "holm", "bh": "fdr_bh", None: None}

# Other spellings of the canonical rule keys used by _eligible_for_output_j.
# The CLI / AutoDetect use the canonical short names; the grid-search GUI and
# ModelScreener.screen() use some of the others.
_RULE_ALIASES = {
    "gpr": "gp",
    "ngb": "ngboost",
    "sgp": "nystroem_gp",   # GUI name for the Nystroem GP approximation
    "LinearRegression": "linear",
    "DecisionTree": "dt",
    "BootstrapLinearRegression": "blr",
    "KNN": "knn",
    "SVR": "svr",
    "GaussianProcess": "gp",
    "RandomForest": "rf",
    "GradientBoosting": "gb",
    "MLP": "mlp",
    "BayesianRidge": "bayesian_ridge",
    "GradientBoosting_Quantile": "gbq",
    "EnsembleHeteroscedastic": "ehet",
}


def _eligible_for_output_j(
        tests: Dict[str, TestResult],
        size: Dict[str, TestResult]) -> Dict[str, bool]:
    """Map test results to a {model_name: eligible} dict.

    Every model name used by the CLI, AutoDetect and the GUI has an explicit
    rule here (directly or via ``_RULE_ALIASES``).
    """
    nonlinear  = not tests["linearity"].passed
    nonnormal  = not tests["normality"].passed
    hetero     = not tests["heteroscedasticity"].passed
    nonlin_dep = not tests["nonlinear_dep"].passed
    needs_nonlin = nonlinear or nonnormal or nonlin_dep
    gp_ok = size["gp_feasible"].passed
    mlp_ok = needs_nonlin and size["mlp_feasible"].passed

    rules = {
        # Cheap linear baselines and regularised linear models -- always eligible
        "linear":         True,
        "dt":             True,
        "mfs_lr":         True,
        "bayesian_ridge": True,
        "elastic_net":    True,
        "lasso":          True,
        "blr":            size["bootstrap_needed"].passed,
        # Median (L1) regression: robust to heavy-tailed / skewed residuals
        "quantile":       nonnormal,
        "knn":            size["knn_ratio_ok"].passed,
        "svr":            size["svr_feasible"].passed,
        # Exact GPs are O(n³): GP, ARD GP, GPX (Rust) and KPLS (Kriging + PLS)
        "gp":             gp_ok,
        "ard_gp":         gp_ok,
        "gpx":            gp_ok,
        "kpls":           gp_ok,
        # Scalable GP approximations (random features, Nystroem, inducing points)
        "rfgp":           True,
        "nystroem_gp":    True,
        "sgpr":           True,
        # PBR: polynomial feature explosion becomes expensive for many inputs
        "pbr":            size["pbr_feasible"].passed,
        # Tree ensembles / boosters
        "rf":             needs_nonlin,
        "et":             needs_nonlin,
        "gb":             needs_nonlin,
        "hgb":            needs_nonlin,
        "lgbm":           needs_nonlin,
        "xgb":            needs_nonlin,
        "catboost":       needs_nonlin,
        # NGBoost models the noise distribution too, so heteroscedasticity alone justifies it
        "ngboost":        needs_nonlin or hetero,
        # Neural nets (cpn is the conformal MLP that AutoDetect deploys for "mlp")
        "mlp":            mlp_ok,
        "cpn":            mlp_ok,
        # Heteroscedastic models (ModelScreener.screen() only)
        "gbq":            hetero,
        "ehet":           hetero,
    }
    rules.update({alias: rules[key] for alias, key in _RULE_ALIASES.items()})
    return rules


# ── ModelSpec dataclass ─────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    """Descriptor for a candidate model returned by :meth:`ModelScreener.screen`."""
    name: str
    factory: object          # callable() -> sklearn estimator, or None if skipped
    cost: str                # 'low' | 'medium' | 'high'
    reason_included: str = ""
    reason_excluded: str = ""


# ── lazy model factories (avoid sklearn imports at module level) ────────────────

def _lr():
    from sklearn.linear_model import LinearRegression
    return LinearRegression()

def _br():
    from sklearn.linear_model import BayesianRidge
    return BayesianRidge()

def _scaled(est):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), est)

def _gp():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from multioutreg.surrogates.gp_sklearn import default_gp_kernel
    return _scaled(GaussianProcessRegressor(
        kernel=default_gp_kernel(), normalize_y=True, n_restarts_optimizer=2))

def _rf():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=100, random_state=0)

def _gb():
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(n_estimators=100, random_state=0)

def _svr():
    from sklearn.svm import SVR
    return _scaled(SVR(C=1.0))

def _knn():
    from sklearn.neighbors import KNeighborsRegressor
    return _scaled(KNeighborsRegressor(n_neighbors=5))

def _dt():
    from sklearn.tree import DecisionTreeRegressor
    return DecisionTreeRegressor(max_depth=5)

def _mlp():
    from sklearn.neural_network import MLPRegressor
    return _scaled(MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=0))

def _blr():
    from sklearn.linear_model import LinearRegression
    from sklearn.base import BaseEstimator, RegressorMixin, clone
    import numpy as _np

    class _BootstrapLR(BaseEstimator, RegressorMixin):
        def __init__(self, n_bootstraps=20, random_state=0):
            self.n_bootstraps = n_bootstraps
            self.random_state = random_state
        def fit(self, X, y):
            rng_ = _np.random.default_rng(self.random_state)
            n = X.shape[0]
            self.models_ = []
            for _ in range(self.n_bootstraps):
                idx = rng_.integers(0, n, n)
                self.models_.append(LinearRegression().fit(X[idx], y[idx]))
            return self
        def predict(self, X, return_std=False):
            preds = _np.stack([m.predict(X) for m in self.models_], axis=1)
            mean = preds.mean(axis=1)
            if return_std:
                return mean, preds.std(axis=1)
            return mean

    return _BootstrapLR()

def _gbq():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.base import BaseEstimator, RegressorMixin
    import numpy as _np

    class _GBQ(BaseEstimator, RegressorMixin):
        def __init__(self, alpha=0.9, n_estimators=100):
            self.alpha = alpha
            self.n_estimators = n_estimators
        def fit(self, X, y):
            a = (1 - self.alpha) / 2
            self.lower_ = GradientBoostingRegressor(loss="quantile", alpha=a,
                                                     n_estimators=self.n_estimators).fit(X, y)
            self.upper_ = GradientBoostingRegressor(loss="quantile", alpha=1-a,
                                                     n_estimators=self.n_estimators).fit(X, y)
            self.mid_   = GradientBoostingRegressor(loss="squared_error",
                                                     n_estimators=self.n_estimators).fit(X, y)
            return self
        def predict(self, X, return_std=False):
            mean = self.mid_.predict(X)
            if return_std:
                # The (lower, upper) quantiles span a central alpha-interval, whose
                # half-width is z * sigma for Gaussian noise; divide by z to get sigma.
                from scipy.stats import norm
                z = norm.ppf(0.5 + self.alpha / 2)
                half_width = (self.upper_.predict(X) - self.lower_.predict(X)) / 2
                std = _np.clip(half_width / z, 0.0, None)  # quantile models can cross
                return mean, std
            return mean

    return _GBQ()

def _ehet():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.base import BaseEstimator, RegressorMixin
    import numpy as _np

    class _EHet(BaseEstimator, RegressorMixin):
        """Bootstrap RF ensemble (epistemic) + RF noise-variance model (aleatoric).

        The noise model is fitted on ``r_i^2 - v_i`` at each training point, where
        ``r_i`` is the residual of the members that did not see point i
        (out-of-bag) and ``v_i`` their variance, so it estimates the noise
        variance as a function of x. std(x) = sqrt(spread(x)^2 + noise(x)).
        """
        def __init__(self, n_estimators=10, random_state=0):
            self.n_estimators = n_estimators
            self.random_state = random_state
        def fit(self, X, y):
            X, y = _np.asarray(X), _np.asarray(y).ravel()
            rng_ = _np.random.default_rng(self.random_state)
            n = X.shape[0]
            self.models_ = []
            oob = _np.ones((self.n_estimators, n), dtype=bool)
            for i in range(self.n_estimators):
                idx = rng_.integers(0, n, n)
                oob[i, idx] = False
                m = RandomForestRegressor(n_estimators=50, random_state=i)
                m.fit(X[idx], y[idx])
                self.models_.append(m)

            preds = _np.stack([m.predict(X) for m in self.models_])
            count = oob.sum(axis=0)
            keep = count >= 2
            self.noise_model_ = None
            if keep.sum() >= 10:
                preds, oob, count = preds[:, keep], oob[:, keep], count[keep]
                mean = (preds * oob).sum(axis=0) / count
                var = (((preds - mean) ** 2) * oob).sum(axis=0) / count
                target = (y[keep] - mean) ** 2 - var
                self.noise_model_ = RandomForestRegressor(
                    n_estimators=100, min_samples_leaf=max(5, n // 20),
                    random_state=self.random_state,
                ).fit(X[keep], target)
            return self
        def predict(self, X, return_std=False):
            preds = _np.stack([m.predict(X) for m in self.models_], axis=1)
            mean = preds.mean(axis=1)
            if return_std:
                noise = (_np.clip(self.noise_model_.predict(X), 0.0, None)
                         if self.noise_model_ is not None else 0.0)
                return mean, _np.sqrt(preds.var(axis=1) + noise)
            return mean

    return _EHet()
