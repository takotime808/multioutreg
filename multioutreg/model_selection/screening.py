# Copyright (c) 2026 takotime808

"""Statistical pre-screening tests for model selection.

Runs inexpensive statistical tests on (X, Y) *before* training any model so that
unsuitable or computationally expensive models can be skipped.

Six tests are provided:

1. **Sample size** — rule-based thresholds gate GP (O(N³)), SVR, MLP, and bootstrap LR.
2. **Normality** (Shapiro-Wilk / D'Agostino K²) — non-normal residuals suggest
   tree-based or non-linear models.
3. **Linearity** (Ramsey RESET) — detected non-linearity activates RF, GB, MLP, DT.
4. **Heteroscedasticity** (Breusch-Pagan) — input-dependent variance activates quantile
   GB and heteroscedastic ensemble wrappers.
5. **Multicollinearity** (VIF) — flags unstable linear model fits.
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

    def __repr__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        p_str = f"p={self.p_value:.4f}" if self.p_value is not None else ""
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
    }


def test_normality(y: np.ndarray, alpha: float = 0.05) -> TestResult:
    """Shapiro-Wilk (N ≤ 5000) or D'Agostino K² (N > 5000).

    PASS = residuals are consistent with normality.
    FAIL = significant non-normality; tree/non-linear models are recommended.
    """
    from scipy import stats as _stats

    y = np.asarray(y).ravel()
    if len(y) <= 5000:
        stat, p = _stats.shapiro(y)
        name = "Normality (Shapiro-Wilk)"
    else:
        stat, p = _stats.normaltest(y)
        name = "Normality (D'Agostino K²)"
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
    regressors (e.g. ``var(e | x) = a + b·x``).

    PASS = no evidence of heteroscedasticity.
    FAIL = input-dependent variance; quantile GB or heteroscedastic ensembles are
           indicated.
    """
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan

    y = np.asarray(y).ravel()
    Xc = sm.add_constant(X)
    ols = sm.OLS(y, Xc).fit()
    lm, lm_p, _, _ = het_breuschpagan(ols.resid, Xc)
    passed = lm_p >= alpha
    msg = ("homoscedastic — constant variance assumption holds"
           if passed else
           "heteroscedastic — noise variance depends on inputs; "
           "quantile GB or EnsembleHeteroscedastic recommended")
    return TestResult(
        "Heteroscedasticity (Breusch-Pagan)", passed,
        float(lm), float(lm_p), msg, alpha)


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

    Examples
    --------
    >>> screener = ModelScreener().fit(X_train, Y_train)
    >>> specs = screener.screen()          # global list of ModelSpec
    >>> matrix = screener.screen_per_output(output_names)  # per-output DataFrame
    """

    def __init__(self,
                 alpha: float = 0.05,
                 gain_threshold: float = 0.05,
                 vif_threshold: float = 10.0) -> None:
        self.alpha = alpha
        self.gain_threshold = gain_threshold
        self.vif_threshold = vif_threshold

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
            self._per_output.append({
                "normality":          test_normality(y_j, self.alpha),
                "linearity":          test_linearity(X, y_j, self.alpha),
                "heteroscedasticity": test_heteroscedasticity(X, y_j, self.alpha),
                "nonlinear_dep":      test_nonlinear_dependency(
                    X, y_j, self.gain_threshold),
            })

        self.global_flags_ = {
            "any_nonlinear":       any(not r["linearity"].passed
                                       for r in self._per_output),
            "any_nonnormal":       any(not r["normality"].passed
                                       for r in self._per_output),
            "any_heteroscedastic": any(not r["heteroscedasticity"].passed
                                       for r in self._per_output),
            "outputs_heterogeneous": (
                len({r["linearity"].passed for r in self._per_output}) > 1
                or len({r["heteroscedasticity"].passed for r in self._per_output}) > 1
            ),
        }
        return self

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
        _add("LinearRegression",
             lambda: _lr(), "low", "always included as cheap baseline")
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
        needs_nonlin = g["any_nonlinear"] or g["any_nonnormal"]
        if needs_nonlin:
            reasons = []
            if g["any_nonlinear"]: reasons.append("RESET detected non-linearity")
            if g["any_nonnormal"]: reasons.append("non-normal residuals")
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
                    reason_excluded="data is linear and normal — heavy non-linear models not needed"))

        # heteroscedasticity gated
        if g["any_heteroscedastic"]:
            _add("GradientBoosting_Quantile",
                 lambda: _gbq(), "high",
                 "Breusch-Pagan detected heteroscedasticity — quantile regression appropriate")
            _add("EnsembleHeteroscedastic",
                 lambda: _ehet(), "high",
                 "Breusch-Pagan detected heteroscedasticity — bootstrap ensemble captures input-dependent noise")
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
        return [i for i, name in enumerate(model_names) if rules.get(name, True)]


# ── helpers ────────────────────────────────────────────────────────────────────

def _eligible_for_output_j(
        tests: Dict[str, TestResult],
        size: Dict[str, TestResult]) -> Dict[str, bool]:
    """Map test results to a {model_name: eligible} dict."""
    nonlinear  = not tests["linearity"].passed
    nonnormal  = not tests["normality"].passed
    hetero     = not tests["heteroscedasticity"].passed
    nonlin_dep = not tests["nonlinear_dep"].passed
    needs_nonlin = nonlinear or nonnormal or nonlin_dep

    return {
        "LinearRegression":          True,
        "DecisionTree":              True,
        "BootstrapLinearRegression": size["bootstrap_needed"].passed,
        "KNN":                       size["knn_ratio_ok"].passed,
        "SVR":                       size["svr_feasible"].passed,
        "GaussianProcess":           size["gp_feasible"].passed,
        "RandomForest":              needs_nonlin,
        "GradientBoosting":          needs_nonlin,
        "MLP":                       needs_nonlin and size["mlp_feasible"].passed,
        "GradientBoosting_Quantile": hetero,
        "EnsembleHeteroscedastic":   hetero,
        # generic fallback for any unrecognised name
        "linear":    True,
        "gp":        size["gp_feasible"].passed,
        "rf":        needs_nonlin,
        "gb":        needs_nonlin,
        "svr":       size["svr_feasible"].passed,
        "knn":       size["knn_ratio_ok"].passed,
        "dt":        True,
        "mlp":       needs_nonlin and size["mlp_feasible"].passed,
        "blr":       size["bootstrap_needed"].passed,
        "cpn":       needs_nonlin and size["mlp_feasible"].passed,
        "mfs_lr":    True,
    }


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

def _gp():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
    return GaussianProcessRegressor(kernel=RBF(), n_restarts_optimizer=2)

def _rf():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=100, random_state=0)

def _gb():
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(n_estimators=100, random_state=0)

def _svr():
    from sklearn.svm import SVR
    return SVR(C=1.0)

def _knn():
    from sklearn.neighbors import KNeighborsRegressor
    return KNeighborsRegressor(n_neighbors=5)

def _dt():
    from sklearn.tree import DecisionTreeRegressor
    return DecisionTreeRegressor(max_depth=5)

def _mlp():
    from sklearn.neural_network import MLPRegressor
    return MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=0)

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
            self.models_ = [LinearRegression().fit(X[rng_.integers(0, n, n)], y[rng_.integers(0, n, n)])
                            for _ in range(self.n_bootstraps)]
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
                std = (self.upper_.predict(X) - self.lower_.predict(X)) / 2
                return mean, std
            return mean

    return _GBQ()

def _ehet():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.base import BaseEstimator, RegressorMixin, clone
    import numpy as _np

    class _EHet(BaseEstimator, RegressorMixin):
        def __init__(self, n_estimators=10, random_state=0):
            self.n_estimators = n_estimators
            self.random_state = random_state
        def fit(self, X, y):
            rng_ = _np.random.default_rng(self.random_state)
            n = X.shape[0]
            self.models_ = []
            for i in range(self.n_estimators):
                idx = rng_.integers(0, n, n)
                m = RandomForestRegressor(n_estimators=50, random_state=i)
                m.fit(X[idx], y[idx])
                self.models_.append(m)
            return self
        def predict(self, X, return_std=False):
            preds = _np.stack([m.predict(X) for m in self.models_], axis=1)
            mean = preds.mean(axis=1)
            if return_std:
                return mean, preds.std(axis=1)
            return mean

    return _EHet()
