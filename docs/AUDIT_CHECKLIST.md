# Statistical / Logic Audit Checklist

Audit date: 2026-10-21. Items tagged **[verified]** were reproduced empirically
(numpy<2, scikit-learn, uncertainty-toolbox 0.1.x); the rest come from code review.

## Critical (results are wrong)

- [x] **1. Conformal coverage calibrated and scored on the same test set.**
  `mor_cli/grid_search_auto_detect.py`, `multioutreg/gui/pages/01_Use_Auto_Detect_MultiOutput_Regressor.py`
  called `calibrate_conformal(X_test, y_test)` and then measured coverage on `X_test`,
  so the reported coverage was in-sample and trivially ~1-alpha.
  *Fix:* hold out a calibration split from the training data.
- [x] **2. Most surrogates report std = 0 (including GP).** [verified]
  `BaseSurrogate.predict` called `MultiOutputRegressor.predict(X, return_std=True)`, which
  raises `TypeError`; the fallback was `np.zeros_like`. GP/SVR/KNN/LR/DT/GB all claimed zero uncertainty.
  *Fix:* iterate `estimators_` so models with native uncertainty (GP, BayesianRidge, ...) report it.
  - [ ] Follow-up: models with *no* native uncertainty still return std = 0 (here, in
    `AutoDetectMultiOutputRegressor.predict`, MoE hard routing, BNN with `n_mc_samples=1`).
    `NaN` would be more honest, but the GUI/report feed `best_std` straight into
    uncertainty-toolbox plots, so those need NaN handling first.
- [x] **3. Bootstrap LR resamples X and y with different indices.** [verified]
  `multioutreg/model_selection/screening.py` (`_blr`) – fitted slopes ~0 on y = 3x.
- [x] **4. `get_uq_performance_metrics` passes args in the wrong order.** [verified]
  `multioutreg/performance/metrics.py` called `uct.get_all_metrics(y_true, y_pred, y_std)`;
  the signature is `(y_pred, y_std, y_true)`. Unit test mocked uct so never caught it.
- [x] **5. Joint multi-output candidates (MoE) scored on training data vs CV for everything else.**
  `multioutreg/model_selection/auto_detect.py` compared in-sample MSE to `GridSearchCV.best_score_`.
  *Fix:* score with out-of-fold predictions on the same CV splitter.
- [ ] **6. Time-series ranking compares different scales, all in-sample.**
  ARIMA/SARIMA metrics are on the *differenced* series (in-sample one-step); LSTM on original
  scale (in-sample); `Ranker` averages ranks across these.
  *Fix:* hold out last H points, forecast every model on the original scale, then rank
  (optionally Diebold–Mariano).

## High (wrong statistics / assumptions)

- [ ] **7. "Std" is actually an interval half-width.** `screening.py` `_GBQ` [verified: 1.53 vs true 1.0],
  `gui/Grid_Search_Surrogate_Models.py` (`GradientBoostingWithUncertainty`),
  `ConformalPredictionNetworkSurrogate` (90% |residual| quantile). Divide by `2*norm.ppf(1-alpha/2)`.
- [ ] **8. Conformal quantile clamps to max residual when n_cal is too small.** [verified]
  `conformal/base.py` – when ceil((n+1)(1-alpha)) > n the interval should be infinite (or warn).
  Consider `method="higher"` for the exact order statistic.
- [ ] **9. `CVPlusConformalPredictor` is not CV+.** Uses mean(fold preds) ± quantile(OOF residuals).
  True CV+ (Barber et al.) uses per-point quantiles of {mu_{-k(i)}(x) ± R_i}; guarantee is 1-2alpha.
  Implement properly or rename and drop the guarantee claim.
- [ ] **10. Normality tested on raw y, not residuals.** [verified] `screening.py` `test_normality`.
- [ ] **11. No multiple-testing correction in screening.** 3 tests × n_outputs at alpha=0.05, OR'ed
  together. Use Holm / Benjamini–Hochberg.
- [ ] **12. Breusch-Pagan confounded by mean misspecification.** Run after a flexible fit or only when RESET passes.
- [ ] **13. Screening inconsistencies.** VIF never used; `screen()` ignores `nonlinear_dep`;
  rule keys mismatch model names (`sgp` vs `sgpr`; `et`, `nystroem_gp`, `elastic_net`, `lasso`,
  `quantile`, `ngboost`, `catboost` default to True); `pre_screen=True` silently ignored unless
  built via `with_vendored_surrogates()`.
- [ ] **14. MoE "EM" never looks at expert fit.** Labels are reassigned from the gating net's own
  `predict_proba`. Reassign by per-sample expert residual/likelihood. Soft-routing std omits
  within-expert variance (law of total variance).
- [ ] **15. Unscaled features & weak GP defaults in model selection.** Wrap SVR/KNN/MLP/GP/Lasso in
  `Pipeline(StandardScaler(), ...)`. Default `GaussianProcessRegressor()` (alpha=1e-10, no WhiteKernel,
  no `normalize_y`) collapsed to `RBF(1e-5)` on noisy data [verified]. ARD GP lacks amplitude term.
- [ ] **16. Model-selection statistics.** `GridSearchCV(cv=int)` is unshuffled KFold; winner's
  `best_score_` is optimistically biased (no nested CV); no test of meaningful difference
  (1-SE rule / Nadeau–Bengio). `mlp` scored as MLPRegressor on 100% but deployed as
  `ConformalPredictionNetworkSurrogate` on 80%.
- [ ] **17. Imputation leaks targets.** `grid_search_auto_detect.py` KNN-imputes inputs *and outputs*
  on the full frame before the split. Split first; impute X on train only; drop rows with missing y.
- [ ] **18. PCA Kaiser rule on unscaled data.** Standardize before PCA (CLI and GUI).

## Medium

- [ ] **19. KS sampling-method detector** (`mor_cli/use_ks_for_sample_bias.py`): sums per-dim D,
  no p-values, single unseeded draw, marginal KS can't separate LHS vs random, grid uses cbrt(n)
  for any d, beta grid ignores loc. Use `qmc.discrepancy` / NN-distance stats vs a simulated null.
- [ ] **20. UMAP sampling classifier**: ad-hoc absolute thresholds on UMAP distances;
  `generate_umap_plot(None)` fabricates random data. Label as heuristic; validate.
- [ ] **21. `ts_suitability.py`**: Ljung-Box on levels (trend ⇒ "suitable"); "seasonal period" = first
  lag with |ACF|>0.3 (returns 2 for any persistent series); add KPSS alongside ADF.
- [ ] **22. SARIMA `detect_seasonal_period`** includes lag 1 ⇒ s=1 for almost any series.
- [ ] **23. ARIMA/SARIMA "MAPE" is |mean signed % error|** — use `mean(abs(...))`.
- [ ] **24. ARIMA differencing loop** has no max d; forecasts/intervals on differenced scale.
  Fit `ARIMA(p,d,q)` directly.
- [ ] **25. `ForecastAutoIntervals`**: quantile branch changes `quantile` without refitting
  (lower = mean = upper); bootstrap branch uses in-sample residuals and ignores horizon growth.
- [ ] **26. `time_series/metrics.py`**: `mase` denominator should use the training series;
  `weighted_quantile_loss` is mean pinball, not Chronos WQL (2·Σpinball/Σ|y|).
- [ ] **27. Epistemic-only uncertainty** (RF/ET spread, MSE deep ensemble, MC-dropout): no aleatoric
  term ⇒ under-coverage. `_EHet` is not heteroscedastic. Document, add NLL heads, or conformalize.
- [ ] **28. Multi-fidelity**: `MultiFidelitySurrogate` is independent per-level models;
  `StackedVFM` augments with in-sample lower-level preds (use OOF); `AdditiveCorrectionVFM` fixes rho=1.
- [ ] **29. `metrics_generalized_api.py`**: `return_std` branch runs even when already `tried`;
  `predict_var` branch never sets `tried=True`; "overall" metrics flatten mixed-unit outputs.

## Low / hygiene

- [ ] **30.** Module-level `warnings.filterwarnings("ignore")` in `screening.py`, `arima.py`,
  `sarima.py`, `DataProcessor.py` — scope with `warnings.catch_warnings()`.
- [ ] **31.** `confidence_intervals.py` estimates z by Monte Carlo — use `norm.ppf`.
- [ ] **32.** Chronos-Bolt silently drops quantiles outside 0.1–0.9; median fallback is a mean.
- [ ] **33.** `conditional_coverage` bins on y_true; condition on X or predicted width instead.
- [ ] **34.** `AutoDetect` private attrs (`_model_names`, `_surrogate_constructors`) are lost by `sklearn.clone`.
- [ ] **35.** `tests/unit/performance/test_metrics_generalized.py` monkeypatches uct at module level (leaks).
- [ ] **36.** Add regression tests for each critical fix (bootstrap slope, GP std > 0, conformal
  held-out coverage, unmocked uct metrics).
