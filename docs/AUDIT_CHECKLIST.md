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
- [x] **14. MoE "EM" never looks at expert fit.** Labels are reassigned from the gating net's own
  `predict_proba`. Reassign by per-sample expert residual/likelihood. Soft-routing std omits
  within-expert variance (law of total variance).
  *Done:* E-step assigns argmax_k log π_k(x) + log N(y | μ_k(x), σ²) (σ² per output, pooled).
  Two linear experts on y = 3|x| now find the kink in 8/8 seeds (was 1/5). Also fixed:
  random-label init made the experts near-identical, so one took every sample; now k-means on
  standardised (X, Y), and an emptied expert is re-seeded with the worst-fit samples instead of
  refit on everything. Soft std = sqrt(Σ w_k (σ_k² + (μ_k − μ)²)); hard routing returns the
  chosen expert's std instead of 0.
- [x] **15. Unscaled features & weak GP defaults in model selection.** Wrap SVR/KNN/MLP/GP/Lasso in
  `Pipeline(StandardScaler(), ...)`. Default `GaussianProcessRegressor()` (alpha=1e-10, no WhiteKernel,
  no `normalize_y`) collapsed to `RBF(1e-5)` on noisy data [verified]. ARD GP lacks amplitude term.
  *Done:* `BaseSurrogate(scale_inputs=...)`; GP, SVR, KNN, CPN (MLP), Lasso and ElasticNet
  surrogates default to `scale_inputs=True`, and AutoDetect grid-searches the same models as
  scaler pipelines (grid keys are un-prefixed before building the surrogate). GP defaults to
  `C·RBF + White` with `normalize_y=True` (`gp_sklearn.default_gp_kernel`); AutoDetect searches
  RBF vs Matern-5/2. On the verified case: test MSE 6.2 → 0.03. ARD GP is now `C·ARD-RBF + White`
  on standardised inputs with `normalize_y`. Screening factories use the same.
  - [ ] Follow-up: RFF / Nystroem GP grids (`length_scale`, `gamma`) are still in raw input units.
- [x] **16. Model-selection statistics.** `GridSearchCV(cv=int)` is unshuffled KFold; winner's
  `best_score_` is optimistically biased (no nested CV); no test of meaningful difference
  (1-SE rule / Nadeau–Bengio). `mlp` scored as MLPRegressor on 100% but deployed as
  `ConformalPredictionNetworkSurrogate` on 80%.
  *Done:* int `cv` → `KFold(shuffle=True, random_state=0)`, one split list shared by every
  candidate (incl. joint MoE/BNN). `selection_scores_` records each candidate's mean CV score and
  Nadeau–Bengio corrected SE per output. `selection="one_se"` picks the least complex candidate
  within 1 SE of the best (default stays `"best"`). `mlp` is grid-searched as `_CPNEstimator`, i.e.
  the deployed CPN with the same `random_state`. Not done: nested CV — the docstring says the
  winner's CV score is optimistic; the CLI and GUI already report metrics on a held-out test set.
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

- [x] **19. KS sampling-method detector** (`mor_cli/use_ks_for_sample_bias.py`): sums per-dim D,
  no p-values, single unseeded draw, marginal KS can't separate LHS vs random, grid uses cbrt(n)
  for any d, beta grid ignores loc. Use `qmc.discrepancy` / NN-distance stats vs a simulated null.
  *Done:* new shared detector `multioutreg/sampling/detect.py::detect_sampling_method`. It computes
  stratum occupancy, centered L2 discrepancy (`qmc.discrepancy`, n ≤ 1000) and mean NN distance,
  and compares them with the same statistics on designs simulated by Random / LHS / Sobol
  (Sobol + Halton) of the same n and d. The statistics are combined into one Mahalanobis distance
  and a Monte Carlo p-value per method; the method with the highest p wins, or "Uncertain" if
  all p < 0.05. `n_sim` is 199 / 99 / 49 depending on n, so the smallest attainable p is always
  < 0.05. Grids are recognised by structure (few levels per input; full factorial, including
  when extra columns are derived features), and unscrambled Sobol / Halton by exact match.
  `compare_sample_methods` now maps inputs through their YAML CDFs (probability integral
  transform) and runs the detector on known bounds, so any marginal works. It reports a per-input
  KS test with p-values as a check of the YAML marginals only, and has `--seed`. Generators take an
  `rng`; grid levels are `floor(m^(1/d))` at equal-probability stratum centres; beta uses `loc`; and
  YAML `"None"` strings are read as missing. Validation (10 seeded designs per cell, n ∈ {20, 100, 300},
  d ∈ {2, 5, 10}): see item 20.
- [x] **20. UMAP sampling classifier**: ad-hoc absolute thresholds on UMAP distances;
  `generate_umap_plot(None)` fabricates random data. Label as heuristic; validate.
  *Done:* the thresholds are gone. `generate_umap_plot` (figures + the duplicate in
  `report_plotting_utils`), `infer_sampling_and_plot_umap` and the `detect_sampling` CLI take the
  method from `detect_sampling_method`. UMAP is now only the picture: it does not preserve
  distances, so its embedding cannot separate designs. `None` / empty input raises `ValueError`;
  fewer than 4 points gives "Uncertain" in the plot caption, and fewer than 10 points skips the MC
  test ("Uncertain"). The KMeans coloring was dropped.
  Validation (inputs rescaled to an arbitrary box, default `n_sim`, 10 seeds per cell,
  n ∈ {20, 100, 300} × d ∈ {2, 5, 10}, 90 designs per row): Random 85/90, LHS 81/90,
  scrambled Sobol 86/90, Halton → "Sobol" 77/90, normal inputs → "Uncertain" 71/90. At n ≥ 100 every
  cell is ≥ 8/10, and most misses are "Uncertain" (about alpha = 5%). At n = 20, normal inputs are
  often called "Random" (power is low) and Halton in d = 10 is 4/10. The old UMAP rules had no
  validation; on the repo's example datasets they called the LHS file "not LHS" (E9).
  - [ ] Follow-up: designs with non-uniform marginals (e.g. LHS of normals) are "Uncertain" unless
    the marginals are supplied (`compare_sample_methods`), and the reports test `X_train` after
    the split / imputation / PCA, which hides a true LHS. Plan:
    [Appendix B](#appendix-b--sampling-detector-known-limits).
- [x] **21. `ts_suitability.py`**: Ljung-Box on levels (trend ⇒ "suitable"); "seasonal period" = first
  lag with |ACF|>0.3 (returns 2 for any persistent series); add KPSS alongside ADF.
  *Done:* KPSS (level) runs alongside ADF; the series is stationary only when ADF rejects and KPSS
  does not. Otherwise Ljung-Box runs on the first differences (new key `tested_on`), so a random
  walk is no longer "suitable" (its increments are white noise). The season comes from the new
  shared `multioutreg/time_series/seasonality.py::detect_seasonal_period`: the highest local ACF peak
  at lag ≥ 2 above max(0.3, 1.96/√n), with a dip before it, preferring the smallest such peak
  within 80% of the best (not a multiple of the period). Non-stationary series are scanned on the
  linearly detrended levels, then on the differences. Validation (50 seeds, n = 200–240): white
  noise, trends, random walks with and without drift → no period (random walk 48/50); AR(0.9) 47/50
  none (was always 2); s = 7 → 50/50; trend + s = 12 → 50/50; random walk + s = 7 → 50/50.
  New keys `kpss_statistic`, `kpss_pvalue`; the existing keys are unchanged.
- [x] **22. SARIMA `detect_seasonal_period`** includes lag 1 ⇒ s=1 for almost any series.
  *Done:* delegates to the shared detector above (returns 0 when there is no season, as before).
  When s < 2 the grid only tries P = Q = 0: seasonal terms with s < 2 always failed in statsmodels,
  so 12 of the 16 fits were wasted.
- [ ] **23. ARIMA/SARIMA "MAPE" is |mean signed % error|** — use `mean(abs(...))`.
- [ ] **24. ARIMA differencing loop** has no max d; forecasts/intervals on differenced scale.
  Fit `ARIMA(p,d,q)` directly.
- [ ] **25. `ForecastAutoIntervals`**: quantile branch changes `quantile` without refitting
  (lower = mean = upper); bootstrap branch uses in-sample residuals and ignores horizon growth.
- [ ] **26. `time_series/metrics.py`**: `mase` denominator should use the training series;
  `weighted_quantile_loss` is mean pinball, not Chronos WQL (2·Σpinball/Σ|y|).
- [x] **27. Epistemic-only uncertainty** (RF/ET spread, MSE deep ensemble, MC-dropout): no aleatoric
  term ⇒ under-coverage. `_EHet` is not heteroscedastic. Document, add NLL heads, or conformalize.
  *Done:* `DeepEnsembleSurrogate` and `BNNSurrogate` take `loss="nll"` (new default): each network
  has a mean and a softplus variance head trained on the Gaussian NLL, and the std is that of the
  equal-weight mixture, sqrt(mean noise var + var of the member / MC-pass means) (Lakshminarayanan
  2017; Kendall & Gal 2017). `loss="mse"` keeps the old epistemic-only spread, documented as
  under-covering. RF / ET (`RandomForestSurrogate`, `ExtraTreesRegressorSurrogate`, the grid-search
  `*WithUncertainty` wrappers and the RF interval plots in `figures/confidence_intervals.py`) now
  share `surrogates/forest_uncertainty.py`: std = sqrt(tree var(x) + c), where c = max(held-out
  MSE − mean held-out tree var, 0) per output, held-out = out-of-bag trees (RF) or 5-fold refits
  (ET, `bootstrap=False`). This makes the mean predicted variance match the held-out MSE; c is
  constant in x and smaller than the noise variance, because fully grown trees already fit some of
  the noise. `include_noise=False` returns the spread alone. `_EHet` keeps its bootstrap-RF spread
  and adds an RF noise model fitted on out-of-bag `r² − v`, so its std follows the noise.
  Validation (n_train = 400, d = 3, test n = 3000, two outputs: noise 0.1–0.6 and 0.3, two seeds),
  coverage of 90% Gaussian intervals before → after: deep ensemble 0.14–0.22 → 0.80–0.87,
  MC-dropout 0.34–0.43 → 0.92–0.95, `_EHet` 0.40–0.47 → 0.88–0.93, RF 0.80–0.86 → 0.86–0.92,
  ET 0.80–0.86 → 0.85–0.92. Correlation of predicted std with the true noise std: 0.97 (deep
  ensemble), 0.94 (MC-dropout), 0.95 (`_EHet`). Tests: coverage and heteroscedasticity tests
  in `tests/unit/surrogates/test_{rf,extra_trees}_sklearn.py`, `test_{bnn,deep_ensemble}_pytorch.py`,
  `tests/unit/model_selection/test_screening.py`, `tests/unit/gui/test_Grid_Search_Surrogate_Models.py`.
- [x] **28. Multi-fidelity**: `MultiFidelitySurrogate` is independent per-level models;
  `StackedVFM` augments with in-sample lower-level preds (use OOF); `AdditiveCorrectionVFM` fixes rho=1.
  *Done:* `StackedVFMSurrogate` (new `oof_cv=5`, `random_state=0`) trains each level on out-of-fold
  lower-level predictions at inputs that are also lower-level training points (nested designs);
  other inputs, and `predict`, use the full models. Duplicate rows share a fold, and fold models
  are fitted only when inputs overlap and are dropped after `fit`. `AdditiveCorrectionVFM` is now
  Kennedy-O'Hagan AR1: `f_hi = rho * f_lo + delta`, with `rho="fit"` (default) the per-output OLS
  slope of `Y_hi` on out-of-fold `f_lo(X_hi)` (falls back to 1 if `f_lo` is constant there) or a
  fixed number (`rho=1.0` is the old model). Std is `sqrt(rho^2 s_lo^2 + s_delta^2)`, and
  `predict_components` returns `(rho * f_lo, delta)`. `MultiFidelitySurrogate` stays the uncoupled
  baseline (auto-detect uses it as a single-level container); its docstring now says so and
  points to the coupled models. The MF GUI page also offers `AdditiveVFM+<base>+rho1`.
  Simulation (12 seeds, 120 lo / 25 nested hi points, GP correction): with near-noise-free lo data OOF
  costs a little (stacked GP-lo MSE 0.0006 → 0.0008) and fitted rho cuts AR1 error ~3× with a GP lo
  model (0.0018 → 0.0005). With noisy lo data (σ=0.5) OOF helps (stacked RF 0.17 → 0.12), and
  fitted rho without OOF breaks (GP-lo MSE 2.48 vs 0.057 with OOF). With an RF lo model, fitted rho
  is worse than rho=1 (it scales up RF's smoothing error), hence the `+rho1` GUI candidate.
  Tests: `tests/unit/surrogates/test_multi_fidelity_fixes.py`.
- [x] **29. `metrics_generalized_api.py`**: `return_std` branch runs even when already `tried`;
  `predict_var` branch never sets `tried=True`; "overall" metrics flatten mixed-unit outputs.
  *Done:* `get_uq_performance_metrics_flexible` now uses `_predict_with_uncertainty` (the requested
  API first, then return_std → return_cov → predict_std → predict_var) instead of its own copy, so
  a successful per-estimator `MultiOutputRegressor` pass is never overwritten and each API is
  called at most once. A plain model with `return_std` (e.g. sklearn GP) no longer raises when no
  method is given. "Overall" metrics are computed after dividing each output by the std of its
  `y_test` column (documented; they are now in std units and do not change when one output's units
  change). Also fixed: `return_cov` misread sklearn's multi-target `(n, n, n_targets)` covariance.
  Tests: `tests/unit/performance/test_metrics_generalized_fixes.py`, and for 21–22
  `tests/unit/time_series/test_seasonality.py`.

- [x] **29b. Grid-search app / `grid-search` CLI select on the test set.**
  `gui/Grid_Search_Surrogate_Models.py` and `mor_cli/grid_search.py` pick the combo with the lowest
  `mean_squared_error(y_test, pred)` and then report metrics on that same test set (as 1c). They
  also build SVR/KNN/GP/Lasso/ElasticNet directly from sklearn classes, so item 15's scaling and GP
  kernel fixes do not reach them (`gpr` grid: RBF / Matern with alpha=1e-4, no amplitude).
  Found while fixing: the "grid search" never searched. `ParameterGrid({"combos": [configs] * n})`
  treats the list as values of one key, so every iteration got the whole config list and
  `combo[:n_outputs]` was always the first n configs: every output got GP-RBF, nothing else ran.
  The GUI conformal block also calibrated on test residuals and scored coverage on them (item 1).
  *Done:* both now use `grid_search_core.default_surrogate_defs()` (one list; the CLI had drifted
  and lacked et/svr/dt/cpn/ngb) and `select_per_target()`: each config is scored per output on a
  20% validation split of the training data, winners are refit on all training data, the test
  set is used once for metrics. MoE replaces the per-output winners only if its validation MSE is
  lower. GP/KNN/SVR/ElasticNet/Lasso are scaled pipelines; GP uses `C·RBF + White` / Matern.
  `mfs_lr` (could never fit: `fit((X, y))`) replaced by a plain `linear` baseline. GUI conformal
  holds out a calibration split from training, as on page 01. On
  `example_dataset_3_outputs.csv`: test R² 0.91 / 0.82 / 0.86 (old, always GP-RBF) →
  0.98 / 1.00 / 1.00 (ARD GP chosen per output). The Streamlit app itself was not run.

## Low / hygiene

- [ ] **30.** Module-level `warnings.filterwarnings("ignore")` in `screening.py`, `arima.py`,
  `sarima.py`, `DataProcessor.py` — scope with `warnings.catch_warnings()`.
- [ ] **31.** `confidence_intervals.py` estimates z by Monte Carlo — use `norm.ppf`.
- [ ] **32.** Chronos-Bolt silently drops quantiles outside 0.1–0.9; median fallback is a mean.
- [ ] **33.** `conditional_coverage` bins on y_true; condition on X or predicted width instead.
- [x] **34.** `AutoDetect` private attrs (`_model_names`, `_surrogate_constructors`) are lost by `sklearn.clone`.
  Worse in practice: `clone` raised `RuntimeError` because `__init__` rewrapped `estimators` /
  `param_spaces` in `list()`. They are now stored as given, and `__sklearn_clone__` carries
  `_model_names`, `_surrogate_constructors`, `fidelity_levels` and unfitted copies of the
  registered multi-output candidates.
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
- [x] **E9.** `lhs_detector_app` classifies the repo's own `lhs_sampled_dataset.csv` as "not LHS"
  (see items 19–20).
  *Cause:* the app used every numeric column, including `response` (|Spearman ρ| with x1 = 0.36
  > 0.2 threshold). Its "discrepancy" formula was also not a standard discrepancy, and it was
  compared with `np.random.rand` designs that were not min-max scaled like the data. The inputs
  themselves are a perfect LHS (100/100 strata filled in each dimension).
  *Fix:* the app, the Sampling & Bias dashboard and the dataset-examples page let the user pick the
  input columns, and leave out response-like names (`y*`, `response*`, `output*`, `target*`) by
  default via `guess_input_columns`. `detect_sampling` has `--columns` / `--exclude`. The app uses
  the shared detector: LHS p = 0.50, Random / Sobol p = 0.005. Also fixed a crash in the
  single-column histogram.
- [ ] **E10.** The ts pipeline picked SARIMA from the CLI and LSTM from the UI on the same data
  (unseeded LSTM + item 6 ranking).
- [x] **E13.** CI's "Generate badge" step failed on Python 3.12: `uv sync` resolved setuptools 82,
  which no longer ships `pkg_resources`, and coverage-badge 1.1.2 imports it. The `test` extra now pins
  `setuptools<81`; relocked.

## Multi-fidelity page and report review (2026-10-21/22)

Found by running `gui/pages/02_Multi_Fidelity_Surrogate_Models.py` end to end (Streamlit
`AppTest`, example 3-level dataset, conformal on) and checking every plot and caption in the
report. User guide: `docs/multi_fidelity_gui.md`.

- [x] **MF1.** SHAP explained the *lowest*-fidelity model when the winner was `MultiFidelity+…`
  (`MultiFidelitySurrogate.predict` defaults to level 0; see 1c follow-up). [verified]
  *Fix:* the report explains `_HighestFidelityView`, which predicts the highest level.
- [x] **MF2.** PDPs were always "not supported" (no multi-fidelity model has `estimators_`), and
  `feature_names` was set to the *output* names. *Fix:* `_HighestFidelityView.estimators_` are
  per-output sklearn regressors (`_OutputColumn`, `RegressorMixin` first so sklearn ≥ 1.6 tags it
  as a regressor); input names are passed. SHAP plots also label inputs by name now.
- [x] **MF3.** Gaussian-vs-conformal plot used *test* residuals as calibration residuals, so the
  conformal curve matched the diagonal by construction. *Fix:* passes the calibration residuals.
- [x] **MF4.** Uncertainty table showed `dummy_metric: 0.0`. *Fix:* sharpness, 95% coverage,
  miscoverage, Gaussian NLL (omitted when any std = 0); also shown in the UI.
- [x] **MF5.** Sampling section used the UMAP threshold heuristic, which labelled the uniform
  random example design "LHS". *Fix:* `detect_sampling_method` on the full highest-fidelity input
  design (Random, p = 0.835); plot is a scatter for ≤ 2 inputs, UMAP otherwise.
- [x] **MF6.** `CatBoostSurrogate.predict(X)` returned shape `(n_outputs, n, 2)` with the default
  `use_uncertainty=True` (raw `[mean, variance]` per row), so `StackedVFM+CatBoost` crashed.
  [verified] *Fix:* returns the mean column only (also for `_conformal_point_predict`); test added.
- [x] **MF7.** Report text: training size counted only hi-fi rows (now per level); model type
  now names the winning candidate; residual plot moved from the sampling section to Diagnostics
  with a caption; coverage/conformal captions corrected; error-histogram sign stated; `$R^2$`
  rendered literally; "lower sharpness = better calibrated" and "guaranteed" conformal wording
  fixed in `report/template.html`.
- [x] **MF8.** `template.html` rendered every per-output prediction plot twice (inside an HTML
  comment, which Jinja still renders). Now a Jinja comment.
- [x] **MF9.** Added report sections (optional in the shared template, so other pages are
  unchanged): Summary, Fidelity Levels, Multi-Fidelity vs HF-only baseline, Model Selection.
- [x] **MF10.** Results were not reproducible: base surrogates were built with no arguments
  (`SparseGPSurrogate` defaults to `random_state=None`), which produced a spurious −69% MF-vs-HF
  difference between identical models. *Fix:* "Fix random seed" control (default on, seed 0)
  drives all splits, each surrogate's seed argument (`_SEED_KWARG`), and the residual-plot
  bootstrap (`plot_residuals_multioutput_with_regplot(seed=...)`). Two seeded runs give a
  byte-identical report. An independent winner's baseline reuses its own per-level models.
- [x] **MF11.** Same test-residual leak as MF3 in `gui/grid_search_core.py` (~line 652,
  `np.abs(y_test - best_pred)` passed to `plot_conformal_vs_gaussian`). *Fix:* both
  `grid_search_core.generate_html_report` and `report/auto_detect_report.generate_html_report`
  (same leak) take `calibration_residuals` and draw the plot only when given; the grid-search
  and auto-detect pages pass the residuals from their held-out calibration split.
- [x] **MF12.** `gui/Grid_Search_Surrogate_Models.py` and `gui/pages/01_Use_Auto_Detect_MultiOutput_Regressor.py`
  still passed `uncertainty_metrics={"dummy_metric": 0.0}` to the report. *Fix:* MF4's helper moved
  to `utils.metric_utils.gaussian_uncertainty_metrics` and is used by all three pages (test set;
  also shown in the UI).
- [ ] **MF13.** Other pages and the CLI still build surrogates unseeded and print "Random seed: 0"
  in the report although only the splits use it (see E10 for the same issue in the ts pipeline).
- [ ] **MF14.** Other pages still use `generate_umap_plot`'s threshold heuristic for sampling
  inference (MF5); see Appendix B.
- [ ] **MF15.** `template.html` overview still says results cover "all model outputs and
  fidelities"; the multi-fidelity report evaluates only the highest fidelity.


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


---

## Appendix B — Sampling detector: known limits

Status: **planned, not started.** Follow-up to items 19, 20 and E9. Two open decisions (B.4).
The numbers below were measured on 2026-10-21 with the detector in
`multioutreg/sampling/detect.py` (10 seeds per row unless stated otherwise).

### B.1 The two limits

**L1. Inputs that are not uniform.** The detector's statistics (strata filled, centered
discrepancy, nearest-neighbour distance) assume the design was drawn uniformly on a box. An LHS
of normal or lognormal inputs therefore comes out "Uncertain". Today the only way around this is
`compare_sample_methods` with a YAML that gives every input's distribution, and most users do not
have one.

**L2. The reports test the wrong matrix.** `auto_detect_report.py:203`,
`gui/grid_search_core.py:612` and `gui/pages/02_Multi_Fidelity_Surrogate_Models.py:121` call
`generate_umap_plot(X_train)`. By then `X_train` is:

| What happened to it | Where | Effect on a true LHS (n = 200, d = 3) |
|---|---|---|
| nothing (full design) | — | LHS 10/10 |
| 75% train split | `grid_search_auto_detect.py:71`, page 01 `:296` | LHS 1/10, Uncertain 9/10 |
| further 80% calibration split (60% left) | `grid_search_auto_detect.py:110` | Uncertain 10/10 |
| PCA on standardised inputs | `fit_input_pca` (when enabled) | Uncertain 10/10 |
| KNN-imputed values | `impute_train_test` | imputed points are not design points |

A random subset of an LHS is not an LHS (most strata are empty), and PCA mixes the columns. Also,
nothing stops the user from listing a response or a derived feature as an input column.

### B.2 Plan for L1: statistics that do not depend on the input distributions

If each input goes through its own increasing transform (any marginal distribution), two things
do not change. One is the order of the points along each input. The other, approximately, is the
ratio of neighbouring gaps between sorted values, because a smooth transform stretches two
adjacent gaps by about the same factor. So the design can be tested without knowing the
marginals:

- **Spacing ratio** (marginal stratification, separates LHS / Sobol from Random). For each
  input, sort the values, take the gaps `s_i`, drop the outer 5% at each end (that is where the
  transform's slope changes fastest), and compute `r_i = s_{i+1} / (s_i + s_{i+1})`. For i.i.d.
  sampling `r ~ Uniform(0, 1)`, so `E|r - 0.5| = 0.25`. Stratified designs space their points more
  evenly. Use the mean of `|r - 0.5|` over inputs.
- **Rank discrepancy** (joint structure, separates Sobol from LHS / Random). Centered discrepancy
  of the rank-transformed design `(rank - 0.5) / n`. Ranks are unchanged by any increasing
  transform. LHS and i.i.d. designs have the same rank pattern (independent random permutations),
  but a Sobol / Halton design does not.

Prototype (n = 100, d = 3, 40 seeds, mean ± sd):

| Inputs drawn from | Random: SR / rank-CD | LHS: SR / rank-CD | Sobol: SR / rank-CD |
|---|---|---|---|
| uniform | 0.252±0.012 / −3.04±0.12 | 0.150±0.008 / −3.00±0.12 | 0.166±0.018 / −3.62±0.02 |
| normal | 0.252±0.012 / −3.04±0.12 | 0.150±0.008 / −3.00±0.12 | 0.166±0.018 / −3.62±0.02 |
| gamma(2) | 0.252±0.012 / −3.04±0.12 | 0.150±0.008 / −3.00±0.12 | 0.166±0.018 / −3.62±0.02 |
| lognormal(0, 2) | 0.253±0.012 / −3.04±0.12 | 0.151±0.008 / −3.00±0.12 | 0.166±0.018 / −3.62±0.02 |

Random and LHS are about 8 standard deviations apart on the spacing ratio, and Sobol is about 5
apart from both on rank discrepancy. The results are the same for every input distribution. These
statistics need no fitted distributions, so the null designs can still be simulated on the unit
cube.

Steps:

1. In `detect.py`, add `spacing_ratio(U)` and `rank_log_cd(U)` (the latter skipped above
   `MAX_N_DISCREPANCY`, like `log_cd`), and a `marginals` argument:
   - `"uniform"`: today's statistics, plus the two new ones. Most power when the inputs really
     are uniform on a box.
   - `"any"`: only the transform-invariant statistics (spacing ratio, rank discrepancy). Valid
     for any continuous marginals.
   - `"auto"`: run both and report `"uniform"` if it finds a consistent method, otherwise
     `"any"`. Say in the explanation which one was used. (**Decision B-1**)
2. Inputs with tied or rounded values make gaps of 0, which breaks the spacing ratio. Treat an
   input with more than 5% zero gaps as discrete, and add that rule to the existing
   `levels <= n/2` rule.
3. **Meaning of "Random" in `"any"` mode:** i.i.d. points whose inputs are independent, with any
   marginals. Observational data usually has correlated inputs, which raises the rank discrepancy,
   so it still comes out "Uncertain". Data that really is i.i.d. from, say, a normal distribution
   will now be called "Random", which is correct. Document this in the module docstring and in the
   LHS app's guide.
4. `compare_sample_methods` keeps its YAML transform: known marginals give the most power. When
   any input's KS p < 0.05 (YAML does not fit), rerun with `marginals="any"` and say so.

Risks to check in validation: densities with a near-zero gap in the middle (bimodal), very small
n (< 20), where the 5% trim leaves few gaps, and inputs rounded to 2–3 decimals.

### B.3 Plan for L2: test the design, not the training matrix

1. **Detect once, before any splitting.** In `mor_cli/grid_search_auto_detect.py`, page 01,
   page 02, `gui/grid_search_core.py` and `Grid_Search_Surrogate_Models.py`, call
   `detect_sampling_method` on the **raw input columns**. Use every row whose inputs are all
   present, including rows later dropped because a response is missing: those points were still
   sampled. Do this before `train_test_split`, imputation, scaling or PCA.
2. **Pass the result into the report instead of recomputing it.** Add
   `sampling: Optional[SamplingDetection]` and `X_design: Optional[np.ndarray]` (for the UMAP
   picture) to `generate_html_report` in `report/auto_detect_report.py` and to the grid-search
   report in `gui/grid_search_core.py`. When they are missing, fall back to today's behaviour, but
   label it "tested on training subset" so the caveat is visible.
3. **Never test imputed values.** Rows with any missing input are left out of the detection, and
   the report says how many.
4. **Multi-fidelity (page 02):** each fidelity level is its own design, so detect per level and
   show one line per level.
5. **Subset-aware null for callers that only have a subset.** Add
   `detect_sampling_method(X, subsample_of=N)`. Each null design is simulated with `N` points and
   then thinned to `len(X)` at random. This makes "75% of an LHS" testable. Its power will be
   lower than for the full design, and validation measures how much.
6. **Report text.** Show the columns tested, n, the per-method p-values and which marginals mode
   was used. State that the result describes the inputs only.
7. **Coordination.** `grid_search_core.py`, `Grid_Search_Surrogate_Models.py` and page 02 are
   being edited for item 29b and the multi-fidelity follow-up. Do steps 1–2 for those files after
   that work is merged, or in the same branch.

### B.4 Open decisions

- **B-1. Default `marginals` mode.** Recommendation: `"auto"` (prefer `"uniform"`, fall back to
  `"any"`). It gives the most power when inputs are uniform and still works when they are not.
  The alternative, `"any"` everywhere, is simpler and never wrong about the marginals, but loses
  the centered discrepancy and NN statistics, which are what separate LHS from Sobol in higher d.
- **B-2. Whether rows with a missing response count as design points** (B.3 step 1).
  Recommendation: yes. They were sampled; the response is what is missing.

### B.5 Tests to add

- `tests/unit/sampling/test_detect.py`:
  - LHS / Random / Sobol pushed through normal, gamma, lognormal and beta(0.5, 0.5) inverse CDFs
    are classified correctly with `marginals="any"`.
  - `"auto"` gives the same answer as today's code on uniform inputs.
  - Inputs rounded to 1 decimal are treated as discrete.
  - Two correlated normal inputs (ρ = 0.8) come out "Uncertain".
  - A random 75% of an LHS, with `subsample_of=N`, comes out LHS.
- `tests/integration`: run `grid_search_auto_detect` end to end on an LHS of 200 points, with PCA
  on and 5% of the inputs missing. The report says LHS and states the columns and the number of
  rows tested.

### B.6 Done when

- [ ] `marginals` modes implemented; the B.2 table reproduced as a test.
- [ ] Validation over input distribution × design × n ∈ {20, 50, 100, 300} × d ∈ {2, 5, 10}: at
  least 90% correct for n ≥ 50, and false "Uncertain" near alpha (5%), recorded here.
- [ ] Every report and GUI page detects on the full raw design (B.3 steps 1–4), with the caveat
  label on the fallback path.
- [ ] `subsample_of` implemented and its power measured.
- [ ] The LHS app and `detect_sampling` CLI expose `marginals` (default per B-1).

