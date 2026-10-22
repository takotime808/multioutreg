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
  (optionally Diebold–Mariano). **Open decision: holdout size H.** Full writeup:
  [Appendix A](#appendix-a--item-6-time-series-model-ranking).

## High (wrong statistics / assumptions)

- [x] **1b. Multi-fidelity page had the same conformal bug as item 1.**
  `gui/pages/02_Multi_Fidelity_Surrogate_Models.py` calibrated on test-set residuals and scored
  coverage on the same test set. Now calibrates on a held-out part of the hi-fi training split.
- [x] **1c. Multi-fidelity page selected the best model by test-set MSE**, then reported metrics on
  that same test set (optimistic). Now `_select_best_candidate` scores candidates on a 20%
  validation split of the hi-fi training data; the winner is refit on all training data and
  evaluated once on the test set.
  - Also fixed: "MultiFidelity+…" candidates were scored with their *lowest*-fidelity model,
    because `MultiFidelitySurrogate.predict` defaults to `fidelity_levels[0]` and the page passed
    no `level` (same for conformal calibration). The page now calls `_predict_highest`.
  - [ ] Follow-up: the library default (`level=None` → lowest) is a trap for other callers;
    consider defaulting to the highest level or making `level` required.

- [x] **7. "Std" is actually an interval half-width.** `screening.py` `_GBQ` [verified: 1.53 vs true 1.0],
  `gui/Grid_Search_Surrogate_Models.py` (`GradientBoostingWithUncertainty`),
  `ConformalPredictionNetworkSurrogate` (90% |residual| quantile). Divide by `2*norm.ppf(1-alpha/2)`.
  *Done:* all three now divide by the matching z (and clip at 0 for crossed quantiles).
- [x] **8. Conformal quantile clamps to max residual when n_cal is too small.** [verified]
  `conformal/base.py` – when ceil((n+1)(1-alpha)) > n the interval should be infinite (or warn).
  Consider `method="higher"` for the exact order statistic.
  *Done:* returns `inf` with a warning when k > n; uses the exact k-th order statistic. The two
  copies in `figures/conformal_plots.py` now use the shared function, and the ordered-interval
  plot labels unbounded intervals.
- [x] **9. `CVPlusConformalPredictor` is not CV+.** Uses mean(fold preds) ± quantile(OOF residuals).
  True CV+ (Barber et al.) uses per-point quantiles of {mu_{-k(i)}(x) ± R_i}; guarantee is 1-2alpha.
  Implement properly or rename and drop the guarantee claim.
  *Done:* real CV+ (per-point order statistics); docstring states the 1-2alpha guarantee.
- [x] **10. Normality tested on raw y, not residuals.** [verified] `screening.py` `test_normality`.
  *Done:* `test_normality(X, y)` now tests OLS residuals (uniform X with y = 5x: raw y p < 1e-4,
  residuals pass).
- [x] **11. No multiple-testing correction in screening.** 3 tests × n_outputs at alpha=0.05, OR'ed
  together. Use Holm / Benjamini–Hochberg.
  *Done:* `ModelScreener(correction="holm" | "bh" | None)` adjusts the normality / RESET / BP
  p-values of all outputs jointly (default Holm), stores `p_adjusted` and flips FAIL→PASS when
  only the raw p was significant. Pure noise, 5 outputs, 40 runs: any-flag rate 20/40 → 3/40.
- [x] **12. Breusch-Pagan confounded by mean misspecification.** Run after a flexible fit or only when RESET passes.
  *Done:* BP only runs when RESET passes; otherwise it is reported as "not tested" (FAIL, p=None,
  excluded from the correction) so heteroscedastic models are kept rather than ruled out.
  Flexible fits were tried first and rejected: on constant-noise data with OOF residuals from RF,
  HistGB, splines or nearest-neighbour differences, BP falsely rejected 15–95% of the time
  whenever d > 1, because residual bias also varies with x.
- [x] **13. Screening inconsistencies.** VIF never used; `screen()` ignores `nonlinear_dep`;
  rule keys mismatch model names (`sgp` vs `sgpr`; `et`, `nystroem_gp`, `elastic_net`, `lasso`,
  `quantile`, `ngboost`, `catboost` default to True); `pre_screen=True` silently ignored unless
  built via `with_vendored_surrogates()`.
  *Done:* VIF failure adds BayesianRidge to `screen()` and notes it on LinearRegression;
  `screen()` includes the RF–LR gain test. Rules now use canonical keys plus aliases
  (`gpr`, `ngb`, GUI `sgp` = Nystroem, the `screen()` long names), covering every CLI/GUI name:
  `et`/`catboost` gated on non-linearity, `ngboost` on non-linearity or heteroscedasticity,
  `quantile` on non-normal residuals, scalable GPs (`nystroem_gp`, `sgpr`) and regularised linear
  models always eligible. Unknown names warn. `pre_screen` on a custom estimator list maps class
  names to rules, and falls back to all candidates if screening rules out every one.
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
- [x] **17. Imputation leaks targets.** `grid_search_auto_detect.py` KNN-imputes inputs *and outputs*
  on the full frame before the split. Split first; impute X on train only; drop rows with missing y.
  *Done:* new `drop_unusable_rows` (before the split: rows missing a target or a "drop" column)
  and `impute_train_test` (after the split: KNN fitted on training inputs only, using all input
  columns) in `utils/imputation.py`. Used by both CLI commands, the main app, page 01 and page 02
  (per fidelity level, which previously also mixed levels). `apply_imputation` is no longer called.
- [x] **18. PCA Kaiser rule on unscaled data.** Four copies of the PCA block (`mor_cli/grid_search.py`,
  `mor_cli/grid_search_auto_detect.py`, `gui/Grid_Search_Surrogate_Models.py`, GUI page 01) ran PCA
  on raw inputs, so "eigenvalue > 1" depended on the inputs' units. All four now call
  `multioutreg/utils/pca_utils.py::fit_input_pca`, which fits `StandardScaler` + PCA on train only
  and clips k to `[1, n_features]` (a manual k or a threshold of 1.0 could exceed it before).

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
- [x] **35.** `tests/unit/performance/test_metrics_generalized.py` monkeypatches uct at module level (leaks).
  Now an autouse fixture.
- [x] **36.** Add regression tests for each critical fix (bootstrap slope, GP std > 0, conformal
  held-out coverage, unmocked uct metrics).

## Environment / end-to-end (found while running the CLI and UI)

- [x] **E1.** `pyproject.toml` allowed scikit-learn 1.2/1.3, but the code uses `root_mean_squared_error`
  (1.4+). Floor raised to 1.4.
- [x] **E2.** `uv.lock` was stale and unsatisfiable on Intel macOS (torch 2.10, llvmlite 0.46 have no
  wheels). Added platform-specific `[tool.uv] constraint-dependencies`; relocked.
- [x] **E3.** umap-learn < 0.5.8 crashes with scikit-learn >= 1.8 (`force_all_finite`). Floor raised.
- [x] **E4.** CLI only worked from the repo root (loaded a GUI page by relative path) and executed
  Streamlit app scripts on import. Report/model code moved to `multioutreg/report/auto_detect_report.py`
  and `multioutreg/gui/grid_search_core.py`. `multioutreg --help` now lists commands.
- [x] **E5.** `generate_html_report` ignored `template_path` / `MOR_TEMPLATE_PATH`.
- [x] **E6.** Chronos defaulted to bfloat16 on MPS, which crashes ("BFloat16 is not supported on MPS").
  bfloat16 is now CUDA-only. The swallowed failure had also hidden `test_chronos_shapes` (restored).
- [x] **E7.** `ts-pipeline` reported `out_dir` as `<out>/<out>` (resolved after `chdir`).
- [x] **E8.** Deprecated `use_column_width` / `use_container_width` rendered warning boxes in the UI;
  replaced with `width="stretch"` (streamlit >= 1.50).
- [ ] **E9.** `lhs_detector_app` classifies the repo's own `lhs_sampled_dataset.csv` as "not LHS"
  (see items 19–20).
- [ ] **E10.** The ts pipeline picked SARIMA from the CLI and LSTM from the UI on the same data
  (unseeded LSTM + item 6 ranking).


---

## Appendix A — Item 6: time-series model ranking

Status: **not started.** This is waiting on one decision, the holdout size H (see A.4).
Everything below comes from reading the code. `ts-pipeline` has not been run on real data yet
(see `TODO.md`), so the first step is to reproduce the problem (A.6, step 0).

Paths below are relative to `multioutreg/time_series/ts_dynamic_fit/` unless they start
with `mor_cli/` or `multioutreg/`.

### A.1 What the pipeline does today

`mor_cli/ts_pipeline.py` and the UI page `multioutreg/gui/pages/05_Time_Series_Forecasting.py`
run the same steps:

1. `DataProcessor.preprocess_data` resamples the target to `freq` and fills gaps with 0
   (`data_handling/DataProcessor.py:160`).
2. They train `ARIMA`, `SARIMA` and `LSTM` on the **whole** series. Each `.run()` returns
   `(metadata, data, model)`, and `metadata["performance"]` holds RMSE / MAE / MAPE.
3. `Ranker.get_best()` (`src/ranker.py:32`) ranks the three models on each of RMSE, MAE and
   MAPE, averages the ranks, and returns the model with the lowest average rank.

### A.2 Why the ranking is wrong

The three models' numbers are not comparable, so the average rank means nothing:

| Model  | Series the error is computed on | Kind of prediction | Code |
|--------|-------------------------------------|--------------------|------|
| ARIMA  | Series differenced *d* times until ADF p ≤ 0.05 | in-sample, one step ahead (`best_model.predict()`) | `algs/arima.py:137-171` |
| SARIMA | Same, differenced | in-sample, one step ahead | `algs/sarima.py:162-199` |
| LSTM   | Original scale (inverse-transformed) | in-sample, 10 steps ahead over every training window | `algs/lstm.py:232-253` |

What this means in practice:

- **Scale.** If the series is differenced, ARIMA/SARIMA errors are on changes from one step to
  the next, not on levels. For a trending series the differences are much smaller than the
  levels, so ARIMA/SARIMA "win" on RMSE/MAE because they are measured on a smaller number,
  not because they forecast better.
- **MAPE is computed differently.** ARIMA/SARIMA compute `abs(mean(signed % error))`, so
  positive and negative errors cancel out. That is not MAPE. On a differenced series the
  denominators are near zero, so the value blows up. LSTM computes `mean(abs(% error))` and
  skips points where `|y| < 1`. The Ranker treats these as the same metric.
- **No holdout.** Every model is scored on the data it was fitted on. Models with more
  parameters are favoured. The LSTM (about 200k parameters, 200 epochs) can memorise the
  series. The ARIMA grid picks (p, q) by AIC over 64 fits, which is fine for choosing an
  order but does not tell you how well it forecasts.
- **Horizon mismatch.** ARIMA/SARIMA are scored one step ahead and LSTM ten steps ahead,
  which is a harder task.
- **Leakage in LSTM.** The `MinMaxScaler` is fitted on the full series (`algs/lstm.py:196`).

Symptom already observed: **E10.** The CLI picked SARIMA and the UI picked LSTM on the same
data. Part of that is the unseeded LSTM, and part is that the ranking is decided by small
artefacts.

### A.3 Bugs nearby (fix them in the same PR or log them separately)

You will hit these while doing item 6:

- **SARIMA `diff_count` is off by one.** `algs/sarima.py:63` increments the counter on *every*
  ADF test, including the one that passes. ARIMA (`algs/arima.py:67-68`) only increments when
  it actually differences. So a SARIMA model on a stationary series reports `diff_count = 1`,
  stores the *undifferenced* series as `diff_1_data`, and `src/infer.py:83` then integrates
  its forecast one time too many.
- **SARIMA seasonal period detection** (`algs/sarima.py:130`) returns the *first* lag with
  ACF > 0.3. For any series with lag-1 autocorrelation that is `s = 1`, which is not a real
  seasonal period. Use the largest significant ACF peak at lag ≥ 2, or take `s` from `freq`
  (7 for daily data with weekly seasonality, 12 for monthly, and so on).
- **LSTM ignores its `fut_pred` argument.** It is hard-coded to 10 (`algs/lstm.py:195`). The
  value callers pass in is `DataProcessor.load_data()[1]`, which is the *row count*
  (`DataProcessor.py:59`), not a horizon. Forecast dates also assume daily frequency
  (`algs/lstm.py:229`).
- **`time_series/metrics.py::mase`** uses the evaluation window to scale the error. It should
  use the training series (this is item 26). Fix it before using MASE below.
- **`ts-pipeline` never saves the best model.** It only writes `results.json`, while
  `main.py:94-111` does save it. This is outside the scope of item 6, but it matters for A.5
  step 5.

### A.4 The decision: holdout size H

All models are scored on the last H points of the series, so H decides what "best" means:

| Option | Pros | Cons |
|--------|------|------|
| **H = forecast horizon the user cares about** (new `--horizon`) | Scores exactly the task the model will be used for | Users must pick one; a small H (1–3) gives noisy scores |
| **H = max(seasonal period, 10% of n)** | Works with no input; a full season means SARIMA is judged fairly | Arbitrary; may not match how the model is used |
| **Fixed H (e.g. 10 or 14)** | Simple; matches the LSTM's current 10-step output | Takes too much from short series and too little from long ones |
| **Rolling-origin backtest** (k folds of H) | Most reliable ranking; makes Diebold–Mariano meaningful | k× training cost; LSTM training already takes 200 epochs |

**Recommendation (not yet agreed):** add `--holdout H` to `ts-pipeline` and the UI. Default it
to `--horizon` when that is given, otherwise `max(s, ceil(0.1·n))`. Enforce a minimum training
length (A.5 step 1) and fail with a clear error if the series is too short. Leave
rolling-origin as an optional `--folds k` that defaults to 1.

### A.5 How to implement it

1. **Split once, in the caller** (`mor_cli/ts_pipeline.py`, and the UI page at the same point).
   `y = processed_data[target]`, `y_train = y[:-H]`, `y_test = y[-H:]`. Before splitting, check
   `len(y_train) >= max(30, 2*s + 10, train_window + H + 20)`. These limits are rough: ARIMA up
   to p = 7 needs a few dozen points, SARIMA needs at least two seasons, and the LSTM needs
   enough training windows.
2. **Give every model the same interface:** `fit(y_train)` and then `forecast(H) -> np.ndarray`
   **on the original scale**. Keep `run()` as a thin wrapper for backwards compatibility.
   - ARIMA / SARIMA: keep the ADF test to choose `d`, but fit
     `sm.tsa.ARIMA(y_train, order=(p, d, q))` / `SARIMAX(..., order=(p, d, q), seasonal_order=(P, D, Q, s))`
     on the **undifferenced** series, and pick (p, q) by AIC as today. With `d` passed to
     statsmodels, `results.forecast(H)` is already on the original scale. The manual
     un-differencing in `src/infer.py:71-85` is then no longer needed, and the SARIMA
     `diff_count` bug goes away.
   - LSTM: fit the scaler on `y_train` only. Set `output_size = H` (forecasting all H steps at
     once), feed the last `train_window` training points, and inverse-transform. Seed torch and
     numpy (this also fixes E10).
3. **Score all models with one function** in `multioutreg/time_series/metrics.py`, applied to
   `(y_test, y_hat)`: RMSE, MAE, MASE (scaled by `y_train`; fix item 26 first), and sMAPE
   instead of MAPE. Delete the per-model metric code in `algs/*.py` so the three models cannot
   drift apart again.
4. **Ranking.** In `src/ranker.py`, rank on the holdout metrics. Use **MASE** as the main
   metric because it does not depend on scale. Optionally keep the average rank as a
   tie-break. When `--folds > 1`, report a Diebold–Mariano test (with the Harvey–Leybourne–Newbold
   small-sample correction) between the best and second-best model. Report it only: with one
   holdout of H ≈ 10 points the test has almost no power, so it must not decide the winner.
5. **Refit the winner on the full series** before saving it or forecasting into the future.
   Write the holdout metrics, H, and the train/test date ranges to `results.json`, so the user
   can see what the ranking was based on.

### A.6 Tests to add (`tests/unit/time_series/test_ts_ranking.py` — none exist today)

0. **Reproduce first:** run `multioutreg ts-pipeline` on a trending synthetic CSV (a random
   walk with drift, level around 1000) and record the current RMSE values. ARIMA/SARIMA should
   show a much smaller RMSE than LSTM, because they are measured on differences.
1. **Same target:** every model's metrics are computed against the same `y_test` array, with
   the same length H.
2. **Scale invariance:** multiply the series by 1000 → same winner, and MASE unchanged.
3. **No leakage:** change the values in `y_test` → fitted parameters and the scaler do not change.
4. **Sanity:** on an AR(1) series ARIMA beats a naive forecast on MASE. On a pure seasonal
   series with s = 7, SARIMA finds `s = 7`.
5. **SARIMA `diff_count`** is 0 on stationary white noise.
6. **CLI smoke test:** run `ts-pipeline` on a series of about 120 points with LSTM epochs
   patched down (e.g. 2) → `results.json` contains `holdout`, per-model metrics and `best_model`.
   The command should fail cleanly when the series is too short for the requested H.

### A.7 Done when

- [ ] H is decided and exposed as `--holdout` (CLI) and as a UI input.
- [ ] All models are scored out-of-sample on the original scale by one shared metric function.
- [ ] `Ranker` uses those metrics; `results.json` records H and the metrics.
- [ ] A.3 SARIMA bugs fixed (or logged as their own items); item 26 `mase` fixed.
- [ ] A.6 tests pass, and the CLI and UI pick the same model on the same data (closes E10).
- [x] **E11.** Streamlit pages never set a matplotlib backend, so on an interactive default (TkAgg)
  the plotting helpers' `plt.show()` opened a desktop "Figure 1" window, blocked the run until it was
  closed, and then `safe_plot_b64` saved a blank figure into the report. All GUI scripts and the
  remaining CLI commands now force `Agg`.
- [ ] **E12.** Plotting helpers in `multioutreg/figures/` should not call `plt.show()` at all; return
  the figure and let the caller decide.
