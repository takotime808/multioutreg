# Copyright (c) 2026 takotime808

"""Sample-size-dependent evaluation plans for tabular (non-time-series) data.

:func:`plan_evaluation` decides, from the number of rows and the conformal
miscoverage level, how a model is evaluated and how its intervals are
calibrated (``docs/AUDIT_CHECKLIST.md``, items S1 and S2):

- **Performance estimate.** A 25% holdout is unbiased at any n but has few
  points at small n (Vabalas et al. 2019), so when it would hold fewer than
  :data:`MIN_TEST_POINTS` rows the estimate comes from nested CV instead:
  every row is predicted once by a model whose selection never saw it
  (Varma & Simon 2006).
- **Conformal calibration.** Split conformal needs at least
  ``ceil(1/alpha) - 1`` calibration points for a finite interval. With fewer,
  CV+ (Barber et al. 2021) is used: it calibrates on out-of-fold residuals of
  every training row.

:func:`fit_with_conformal` and :func:`nested_cv_predict` carry out a plan for
any estimator with ``fit`` / ``predict(X, return_std=True)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from sklearn.model_selection import KFold, train_test_split

#: Below this many holdout rows the performance estimate uses nested CV.
#: A convention (no literature gives a cutoff); see AUDIT_CHECKLIST S1.
MIN_TEST_POINTS = 30
#: Outer folds for nested CV and folds for CV+.
N_OUTER_FOLDS = 5


def min_calibration_points(alpha: float) -> int:
    """Smallest calibration set giving a finite split-conformal interval.

    The interval uses the k-th smallest of n scores with
    ``k = ceil((n + 1)(1 - alpha))``, which is at most n only when
    ``n >= ceil(1/alpha) - 1``.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return math.ceil(1 / alpha - 1e-12) - 1


@dataclass(frozen=True)
class EvaluationPlan:
    """How to estimate performance and calibrate intervals for one dataset."""
    n_samples: int
    scheme: str                       # "holdout" or "nested_cv"
    test_size: float
    outer_cv: Optional[KFold]         # set when scheme == "nested_cv"
    conformal: Optional[str]          # None, "split" or "cv_plus"
    calibration_size: float
    alpha: Optional[float]
    random_state: int = 0
    notes: tuple = field(default_factory=tuple)

    @property
    def n_conformal_folds(self) -> int:
        return N_OUTER_FOLDS

    def describe(self) -> str:
        """One-line summary for reports."""
        if self.scheme == "nested_cv":
            text = (f"Nested {self.outer_cv.n_splits}-fold CV: each of the "
                    f"{self.n_samples} rows is predicted by a model selected and "
                    "fitted without it")
        else:
            text = f"{self.test_size:.0%} held-out test set"
        if self.conformal == "split":
            text += (f"; split-conformal intervals calibrated on "
                     f"{self.calibration_size:.0%} of the training rows")
        elif self.conformal == "cv_plus":
            text += (f"; CV+ conformal intervals ({self.n_conformal_folds}-fold, "
                     "calibrated on every training row)")
        return text


def plan_evaluation(
    n_samples: int,
    alpha: Optional[float] = None,
    test_size: float = 0.25,
    calibration_size: float = 0.2,
    min_test: int = MIN_TEST_POINTS,
    random_state: int = 0,
) -> EvaluationPlan:
    """Choose holdout vs nested CV and split vs CV+ conformal for *n_samples* rows.

    ``alpha=None`` means no conformal intervals are wanted.
    """
    if n_samples < 2:
        raise ValueError(f"Need at least 2 rows to evaluate a model, got {n_samples}")
    notes = []
    n_test = math.ceil(test_size * n_samples)   # what train_test_split holds out
    if n_test < min_test:
        k = min(N_OUTER_FOLDS, n_samples)
        outer = KFold(n_splits=k, shuffle=True, random_state=random_state)
        n_fit = n_samples - math.ceil(n_samples / k)   # smallest outer training fold
        scheme = "nested_cv"
        notes.append(
            f"A {test_size:.0%} test split would hold only {n_test} rows (< {min_test}), "
            f"so performance is estimated by nested {k}-fold CV instead.")
    else:
        outer = None
        n_fit = n_samples - n_test
        scheme = "holdout"

    conformal = None
    if alpha is not None:
        need = min_calibration_points(alpha)
        n_cal = math.ceil(calibration_size * n_fit)
        if n_cal >= need:
            conformal = "split"
        else:
            conformal = "cv_plus"
            notes.append(
                f"A {calibration_size:.0%} calibration split would hold {n_cal} rows, "
                f"fewer than the {need} needed for finite intervals at alpha={alpha}, "
                "so CV+ is used instead.")
            if n_fit < need:
                notes.append(
                    f"Even CV+ needs at least {need} training rows at alpha={alpha} "
                    f"(have about {n_fit}); intervals will be unbounded. Use a larger "
                    "alpha or more data.")
    return EvaluationPlan(
        n_samples=n_samples, scheme=scheme, test_size=test_size, outer_cv=outer,
        conformal=conformal, calibration_size=calibration_size, alpha=alpha,
        random_state=random_state, notes=tuple(notes))


def _as_2d(a) -> np.ndarray:
    a = np.asarray(a)
    return a.reshape(-1, 1) if a.ndim == 1 else a


@dataclass
class ConformalFit:
    """A model fitted under a plan, plus what is needed for its intervals."""
    model: Any                                  # predicts with return_std=True
    method: Optional[str]                       # None, "split" or "cv_plus"
    residuals: Optional[np.ndarray] = None      # (n_cal, n_outputs) |y - y_hat|
    _cv_plus: Any = None

    def predict_interval(self, X, alpha: float):
        """Return ``(lower, upper)`` arrays of shape (n, n_outputs)."""
        if self.method == "cv_plus":
            lo, hi = self._cv_plus.predict_interval(X, alpha=alpha)
            return _as_2d(lo), _as_2d(hi)
        if self.method == "split":
            from multioutreg.conformal.base import BaseConformalPredictor
            pred = _as_2d(self.model.predict(X))
            q = np.array([BaseConformalPredictor._conformal_quantile(self.residuals[:, j], alpha)
                          for j in range(self.residuals.shape[1])])
            return pred - q, pred + q
        raise AttributeError("No conformal calibration was requested for this fit")


def fit_with_conformal(model, X, y, plan: EvaluationPlan) -> ConformalFit:
    """Fit *model* on (X, y) and calibrate conformal intervals as *plan* says.

    - no conformal: fit on all rows.
    - ``"split"``: fit on ``1 - calibration_size`` of the rows, keep absolute
      residuals on the rest.
    - ``"cv_plus"``: :class:`~multioutreg.conformal.cv_plus.CVPlusConformalPredictor`
      refits a clone of *model* (including any model selection it does) on
      each fold, and the final model is fitted on all rows.
    """
    y = _as_2d(y)
    if plan.conformal is None:
        return ConformalFit(model=model.fit(X, y), method=None)
    if plan.conformal == "split":
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X, y, test_size=plan.calibration_size, random_state=plan.random_state)
        model.fit(X_fit, y_fit)
        residuals = np.abs(y_cal - _as_2d(model.predict(X_cal)))
        return ConformalFit(model=model, method="split", residuals=residuals)
    from multioutreg.conformal.cv_plus import CVPlusConformalPredictor
    cvp = CVPlusConformalPredictor(
        model, n_folds=min(plan.n_conformal_folds, len(X)),
        random_state=plan.random_state).fit(X, y)
    return ConformalFit(model=cvp.model_, method="cv_plus",
                        residuals=_as_2d(cvp.residuals_), _cv_plus=cvp)


@dataclass
class OutOfFoldPredictions:
    """Nested-CV predictions: row i comes from the outer fold that left it out."""
    pred: np.ndarray
    std: np.ndarray
    lower: Optional[np.ndarray]
    upper: Optional[np.ndarray]
    fold: np.ndarray                 # outer fold index of each row


def nested_cv_predict(
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], dict],
    X: np.ndarray,
    y: np.ndarray,
    plan: EvaluationPlan,
    on_fold: Optional[Callable[[int, int], None]] = None,
) -> OutOfFoldPredictions:
    """Predict every row from a model that never saw it (outer loop of nested CV).

    ``fit_predict(X_train, y_train, X_test)`` must do *all* data-dependent
    steps (imputation, PCA, model selection, conformal calibration) on
    ``X_train`` only and return a dict with ``"pred"`` and ``"std"`` and,
    when conformal intervals are wanted, ``"lower"`` and ``"upper"``, each of
    shape (len(X_test), n_outputs). ``on_fold(k, n_folds)`` is called before
    each outer fold.
    """
    if plan.outer_cv is None:
        raise ValueError("plan.scheme is not 'nested_cv'")
    y = _as_2d(y)
    n, m = y.shape
    out = {key: np.full((n, m), np.nan) for key in ("pred", "std", "lower", "upper")}
    fold = np.full(n, -1, dtype=int)
    splits = list(plan.outer_cv.split(X))
    for k, (tr, te) in enumerate(splits):
        if on_fold is not None:
            on_fold(k, len(splits))
        res = fit_predict(X[tr], y[tr], X[te])
        for key in out:
            if res.get(key) is not None:
                out[key][te] = _as_2d(res[key])
        fold[te] = k
    has_interval = not np.all(np.isnan(out["lower"]))
    return OutOfFoldPredictions(
        pred=out["pred"], std=out["std"],
        lower=out["lower"] if has_interval else None,
        upper=out["upper"] if has_interval else None,
        fold=fold)


@dataclass
class EvaluationResult:
    """Everything a report needs from :func:`run_evaluation`.

    ``y_true`` / ``pred`` / ``std`` / ``lower`` / ``upper`` are the rows the
    metrics are computed on: the holdout rows, or every row (out-of-fold) under
    nested CV. ``X_eval`` holds those rows' inputs after preprocessing (under
    nested CV, with the preprocessing of the final fit). ``fit`` is the
    deployable model, fitted on the training split (holdout) or on all rows
    (nested CV); ``X_fit`` is its preprocessed training input and ``info`` is
    whatever ``preprocess`` returned for that fit.
    """
    plan: EvaluationPlan
    y_true: np.ndarray
    pred: np.ndarray
    std: np.ndarray
    lower: Optional[np.ndarray]
    upper: Optional[np.ndarray]
    fit: ConformalFit
    X_fit: np.ndarray
    X_eval: np.ndarray
    info: Any = None

    @property
    def model(self):
        return self.fit.model

    @property
    def n_train(self) -> int:
        return len(self.X_fit)

    @property
    def n_test(self) -> int:
        return len(self.y_true)


def run_evaluation(
    make_model: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    plan: EvaluationPlan,
    preprocess: Optional[Callable[[np.ndarray, np.ndarray], tuple]] = None,
    on_step: Optional[Callable[[str], None]] = None,
) -> EvaluationResult:
    """Carry out *plan*: holdout or nested CV, with split or CV+ conformal.

    ``make_model()`` returns a fresh unfitted model (it is called once per
    outer fold and once for the final fit). ``preprocess(X_train, X_test)``
    returns ``(X_train', X_test', info)`` and must fit only on ``X_train``
    (imputation, PCA, ...). ``on_step(label)`` reports progress.
    """
    y = _as_2d(y)
    if preprocess is None:
        def preprocess(X_tr, X_te):
            return X_tr, X_te, None
    step = on_step or (lambda label: None)

    def fit_and_predict(X_tr, y_tr, X_te):
        X_tr, X_te, info = preprocess(X_tr, X_te)
        fit = fit_with_conformal(make_model(), X_tr, y_tr, plan)
        pred, std = fit.model.predict(X_te, return_std=True)
        lower = upper = None
        if plan.conformal is not None:
            lower, upper = fit.predict_interval(X_te, plan.alpha)
        return fit, X_tr, X_te, info, _as_2d(pred), _as_2d(std), lower, upper

    if plan.scheme == "holdout":
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=plan.test_size, random_state=plan.random_state)
        step("Fitting on the training split")
        fit, X_tr, X_te, info, pred, std, lower, upper = fit_and_predict(X_tr, y_tr, X_te)
        return EvaluationResult(plan, y_te, pred, std, lower, upper, fit, X_tr, X_te, info)

    def fold_fit_predict(X_tr, y_tr, X_te):
        *_, pred, std, lower, upper = fit_and_predict(X_tr, y_tr, X_te)
        return {"pred": pred, "std": std, "lower": lower, "upper": upper}

    oof = nested_cv_predict(
        fold_fit_predict, X, y, plan,
        on_fold=lambda k, n: step(f"Nested CV: outer fold {k + 1} of {n}"))
    step("Fitting the final model on all rows")
    X_all, _, info = preprocess(X, X)
    fit = fit_with_conformal(make_model(), X_all, y, plan)
    return EvaluationResult(plan, y, oof.pred, oof.std, oof.lower, oof.upper,
                            fit, X_all, X_all, info)
