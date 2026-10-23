# Branch Integration Plan

Survey of every branch other than `main` (as of 2026-10-22, `main` = `11d27b4`
"Audit critical fixes (#25)"). For each branch: what it was attempting, what is
already on `main`, what still needs to be rebuilt, and what must change so the
rebuilt feature is correct.

Source pointers use `<sha>:<path>` so the original code can still be read with
`git show <sha>:<path>` — **as long as the commits are kept reachable** (see
[Before deleting](#before-deleting-branches)).

---

## 1. Inventory

| Branch | Tip | Unique commits | Status vs `main` | Verdict |
|---|---|---|---|---|
| `audit-critical-fixes` (local) | `cded10f` | 56 | Tree identical to `main` (squash-merged as #25) | Delete, nothing to take |
| `integrate-acf` (local + origin) | `573c7f8` | 57 | Only `docs/references.rst` differs | Take references page (§2.1), then delete |
| `active-learning-prototype` (**local only, never pushed**) | `edc9c6e` | 1 | Adds `docs/active_learning.md` | Take AL plan doc (§2.2), then delete |
| `origin/crossval-datasize-dep` | `9f72fa2` | 4 | Based on pre-audit `49242bd` | Rebuild CV auto-selection + MF notebook (§2.3) |
| `origin/ts-review` | `dd4eb27` | 2 | Based on pre-audit `49242bd` | Rebuild TS forecaster expansion (§2.4) — largest item |
| `origin/moo-regret` | `e62d31c` | 7 | 95 commits behind | Rebuild regret metrics + model download (§2.5) |
| `origin/pca_sample_detection` | `b8cea01` | 1 | 116 commits behind | Mostly superseded; take PCA display fallback only (§2.6) |
| `origin/main-backup-pre-ai` | `49242bd` | 0 | Ancestor of `main` | Delete, nothing to take |
| `origin/gh-pages` | — | — | Sphinx deploy target (no shared history) | **Keep** — used by `.github/workflows/gh_pages.yml` |

---

## 2. Items to integrate

Ordered roughly cheapest → most involved. Each item lists **Keep**, **Fix /
change** (problems found in the branch code, or conflicts with the audited
`main`), and **Tests**.

### 2.1 Reorganized references page — from `integrate-acf`

*Attempt:* regroup `docs/references.rst` into one page with topical sections
("Where to Start", "Current Codebase" by area, "Roadmap Items" incl. "Joint
Metrics and Conformal Regions", "Active Learning and Sequential Design",
"Cross-Cutting Additions").

- **Keep:** `573c7f8:docs/references.rst` wholesale. Checked: every arXiv/DOI
  link on `main` is still present; the branch adds 29 more.
- **Fix / change:** add citation B1 from `crossval-datasize-dep`
  (Mainini et al. 2022, *Analytical Benchmark Problems for Multifidelity
  Optimization Methods*, arXiv:2204.07867) under Multi-Fidelity, linked to
  `examples/mf_surrogates.ipynb` (§2.3).
- **Tests:** `make -C docs html` builds without new warnings.

### 2.2 Active learning plan doc — from `active-learning-prototype`

*Attempt:* design doc for an active-learning / sequential-design module
(uncertainty sources, multi-output aggregation, acquisition functions, batch
selection, the loop, multi-fidelity AL with cost-aware acquisitions,
benchmarks, testing plan, delivery phases, risks).

- **Keep:** `edc9c6e:docs/active_learning.md`, the `active_learning` toctree
  entry in `docs/index.rst`, and the two-line link added to
  `docs/surrogate_roadmap.md`.
- **Fix / change:** none needed to land it; it is a plan, not code. The AL
  module itself is future work and should be built from this doc.
- **Note:** this branch exists only locally — it must be merged or pushed
  before any local cleanup or the doc is lost.

### 2.3 Dataset-size-dependent CV + multi-fidelity benchmark notebook — from `crossval-datasize-dep`

*Attempt:* (a) pick the CV scheme from the training-set size instead of a
fixed `cv=3`; (b) make the Multi-Fidelity GUI page select models by CV score
instead of by test-set score; (c) record the actual CV scheme in every report;
(d) add an analytical MF benchmark notebook.

Source: `9f72fa2` — `multioutreg/model_selection/auto_detect.py`,
`multioutreg/gui/pages/02_Multi_Fidelity_Surrogate_Models.py`,
`tests/integration/test_cv_auto_selection.py`, `examples/mf_surrogates.ipynb`.

**(a) `cv="auto"` in `AutoDetectMultiOutputRegressor`**

- **Keep:** the size rule, as a public `select_cv_strategy(n_samples)`:

  | n_samples | splitter |
  |---|---|
  | ≤ 20 | `LeaveOneOut` |
  | 21–100 | `RepeatedKFold(5, 3)` |
  | 101–1000 | `KFold(10, shuffle=True)` |
  | > 1000 | `KFold(5, shuffle=True)` |

  plus the `"loocv"` string alias and a `_cv_description(cv)` helper for reports.
- **Fix / change:**
  - `main` rewrote `auto_detect.py` (audit item 34: `__sklearn_clone__`,
    `random_state`, `_splitter()`, `selection="one_se"`). Do **not**
    cherry-pick; extend `_splitter()` to take `n_samples` and handle
    `"auto"`/`"loocv"` there.
  - The branch set `self.cv_description_` inside `__init__`. That breaks the
    sklearn estimator contract that item 34 just fixed (no derived attributes
    in `__init__`). Set `cv_strategy_` / `cv_description_` in `fit()` only.
  - Thread `self.random_state` into the auto splitters (branch hard-coded 0).
  - `LeaveOneOut` yields one sample per fold, so per-fold `r2` is undefined and
    `GridSearchCV` returns NaN scores. With LOO either require an error-type
    scoring (`neg_mean_squared_error`, `neg_mean_absolute_error`) or fall back
    to `KFold(n)` with a warning when scoring is `r2`.
  - `selection="one_se"` uses the Nadeau–Bengio corrected SE, whose
    `n_test/n_train` term assumes K-fold. Verify it with `RepeatedKFold`
    (folds = `n_splits * n_repeats`) and LOO, or fall back to `"best"` with a
    warning under LOO.
  - `RepeatedKFold(5, 3)` triples fit cost on 21–100 rows. Fine for cheap
    models, but `skip_expensive`/pre-screening should be applied before CV.
  - Decide on the default. The branch changed it from `3` to `"auto"`, which
    changes results for existing callers. Recommended: make `"auto"` the
    default, note it in the changelog, and update `with_vendored_surrogates`
    and the `test_init_valid_estimators` assertion.
- **Tests:** port `test_cv_auto_selection.py` (boundary sizes 20/21/100/101/
  1000/1001, `"auto"` resolution, `"loocv"`, int/splitter pass-through, clone
  round-trip with `cv="auto"`, LOO + `r2` behaviour).

**(b) Multi-Fidelity page: CV-based selection**

- **Keep:** for each candidate, K-fold over the **highest-fidelity** rows
  only, keeping all lower-fidelity data in every fold (lower-fidelity data is
  training-only, so this is legitimate); pick the lowest mean CV MSE; refit the
  winner on all data; report on the held-out test split.
- **Fix / change:**
  - The branch removed test-set MSE/RMSE/R² from the results table. Show both:
    **CV score** (used for selection) and **test score** (unbiased report of
    the chosen model only).
  - Fit cost becomes candidates × folds full MF fits. Keep the progress bar
    updating per fold and honour `skip_expensive`.
  - Respect the page's fixed-seed option when building the splitter.
  - Wrap the fold fitting in a reusable function in
    `multioutreg/surrogates/` or `model_selection/` instead of inline page
    code so it can be unit-tested without Streamlit.

**(c) Report CV label**

- **Keep:** pass `best_model.cv_description_` to the report from
  `mor_cli/grid_search_auto_detect.py` and page 01; label pure hold-out runs as
  `"None (hold-out test split)"` in `mor_cli/grid_search.py` and
  `gui/Grid_Search_Surrogate_Models.py` instead of `"None"`.
- Also take the one-line fix in `report/template.html`: `$R^2$` →
  `R<sup>2</sup>` (MathJax is not loaded in the report).

**(d) `examples/mf_surrogates.ipynb`**

- **Keep:** the notebook — 4-level Forrester functions from Mainini et al.,
  noisy samples (n = 10/25/50/80), comparison of single-fidelity RF,
  `MultiFidelitySurrogate` (2 and 4 levels), `AdditiveCorrectionVFM`,
  `StackedVFMSurrogate`; RMSE table, prediction plots,
  `predict_components()` decomposition, uncertainty bands, and an n_f1 sweep
  over 10 seeds.
- **Fix / change:** re-run it against `main` — audit item 28 changed stacked
  VFM to out-of-fold stacking features and a fitted `rho`, so the numbers and
  some conclusions in the summary table will change. Add `ipykernel` to the
  `[examples]` extra. Take the README install wording (`pip install -e .[all]`,
  `[examples]` for notebooks).

### 2.4 Time-series forecaster expansion — from `ts-review`

*Attempt:* turn `multioutreg.time_series` into a full forecasting toolkit
sharing the `ChronosForecaster` contract
(`fit(y)` → `predict(prediction_length, quantiles)` → `ForecastResult` with
quantiles shaped `[n_series, Q, H]`): bridge every surrogate to forecasting via
lag features, add walk-forward CV, add Prophet and NeuralForecast adapters,
clean up the vendored `ts_dynamic_fit` package, and rebuild the GUI page.

Source: `dd4eb27` — `multioutreg/time_series/{lag_features,lag_forecaster,surrogate_forecaster,cv,uncertainty,figures,prophet_adapter,neuralforecast_adapter}.py`,
`multioutreg/time_series/ts_dynamic_fit/**`,
`multioutreg/gui/pages/05_Time_Series_Forecasting.py`,
`docs/integrated_surrogates.md` (TS section), 15 test files under
`tests/unit/time_series/` + `tests/integration/test_ts_integration.py` +
`tests/unit/gui/pages/test_Time_Series_Forecasting.py`.

Build in this order:

**(a) `ts_dynamic_fit` hygiene (still broken on `main`)**

- **Keep:**
  - replace `logging.basicConfig` + `open("logs/…")` writes in
    `algs/arima.py`, `algs/sarima.py`, `algs/lstm.py`,
    `data_handling/DataProcessor.py`, `src/ranker.py` with module loggers
    (importing the package currently reconfigures the root logger, and fitting
    fails unless a `logs/` dir exists);
  - fix `src/infer.py` and `src/visualize.py` star/relative imports
    (`from algs.arima import *`) to absolute package imports;
  - replace the `"Hello from kiga-staffocaster!"` `__init__.py` with real
    exports (`ARIMA, SARIMA, LSTM, LSTMModel, DataProcessor, Ranker,
    ForecastAutoIntervals, visualize_model`);
  - `fillna(method="bfill")` → `.bfill().ffill().fillna(0)` (pandas 2.x
    deprecation);
  - LSTM: keep the fitted scaler on the instance instead of
    `joblib.dump(…, "temp/scaler.pkl")`.
- **Fix / change:** the branch stopped writing `temp/scaler.pkl` but
  `src/infer.py:95` still `joblib.load`s it — rebuilding without this fix
  breaks LSTM inference. Read the scaler from the fitted model's result dict.
  Make `keras`/`torch` imports lazy so the package imports without `[lstm]`.
- **Tests:** port `test_arima.py`, `test_sarima.py`, `test_data_processor.py`,
  `test_analyzer.py`, `test_ranker.py`, `test_forecast_auto_intervals.py`
  (none of these modules are tested on `main`). Reconcile `test_sarima.py`
  with audit item 22 (SARIMA seasonality) already on `main`.

**(b) Fix `mase()` first (bug on `main`)**

`multioutreg/time_series/metrics.py:mase` scales by the naive error of the
*test* window. That is not MASE, and for a 1-step window
`y_true[1:] - y_true[:-1]` is empty, so the result is NaN. Change the
signature to `mase(y_true, y_pred, y_train, seasonality=1)` and scale by the
in-sample seasonal-naive MAE of `y_train`. Update Chronos/financial callers.
While here, consider normalizing `weighted_quantile_loss` by `sum(|y|)`, which
is how the Chronos paper defines WQL. The docstring cites that paper but the
code does not normalize.

**(c) `lag_features.py`**

- **Keep:** `make_lag_features(series, n_lags, horizon, include_time_features)`
  using `sliding_window_view` → `X (n, n_lags[+4])`, `y (n, horizon)`; optional
  sin/cos day-of-week + month features from a `DatetimeIndex`;
  `rolling_window_features(series, windows)` → rolling mean/std per window.
- **Fix / change:** rolling features at row *t* must use only values ≤ *t*.
  Check the alignment `roll[n_lags : n_lags + n]` in `LagFeatureForecaster`.
  Row *i*'s last lag is index `i + n_lags − 1`, so the rolling row should be
  `roll[n_lags − 1 + i]`. The branch's offset includes the target value,
  which leaks the label into the features.

**(d) `lag_forecaster.py` — `LagFeatureForecaster`**

*Purpose:* wrap any vendored surrogate or sklearn regressor as a recursive
forecaster, with uncertainty from `return_std` (GP family), the surrogate's
own conformal wrapper, or a residual split-conformal fallback.

- **Keep:** the API and three uncertainty modes (`"return_std"`,
  `"conformal"`, `"none"`), recursive multi-step forecasting with rolling
  features recomputed from the growing buffer, and output as `ForecastResult`.
- **Fix / change** (all bugs in the branch):
  1. `predict(alpha=…)` is ignored. The conformal quantile is hard-coded to
     90% at fit time (`np.ceil((n+1)*0.9)`) and `conformal_predict(alpha=0.1)`
     is hard-coded in the loop. Store the calibration residuals and compute
     the quantile from `alpha` at predict time.
  2. Uncertainty does not grow with the horizon: every step reuses the 1-step
     sigma. Either use `propagate_uncertainty_recursive` (from the branch's own
     `uncertainty.py`) or, better, calibrate **per-horizon** conformal
     residuals by running recursive forecasts over the calibration windows.
  3. `include_time_features=True` adds 4 columns during fit, but
     `_recursive_forecast` never builds them, so `predict` fails with a
     feature-count mismatch. Generate future timestamps from the index
     frequency and append the features at each step.
  4. `horizon > 1` trains a direct multi-output model (`y` is `(n, horizon)`)
     but prediction still recurses one step at a time, taking only
     `ravel()[0]`. Pick one strategy: `strategy="recursive"` (train
     1-step) or `strategy="direct"` (train H outputs, predict once).
  5. In conformal mode the model is fit on only the first 80% and never
     refit, so the most recent data (the most informative for forecasting)
     is never used for the point forecast. Refit on all data after
     calibration, or document the trade-off.
  6. The passed-in surrogate is fitted in place. `clone()` it in `fit()` so
     the same instance can be reused across `WalkForwardCV` folds and GUI runs.
  7. The Gaussian conversion `sigma = q / z_0.9` then re-derives arbitrary
     quantiles assuming normality. For conformal mode, emit the requested
     quantiles directly from the empirical residual distribution instead.

**(e) `surrogate_forecaster.py` — `AutoSurrogateForecaster`**

*Purpose:* run `AutoDetectMultiOutputRegressor.with_vendored_surrogates` on
the lag matrix, then wrap the winner in `LagFeatureForecaster`.

- **Fix / change:**
  - `detector.best_estimator_` does not exist on `AutoDetect` (main stores
    per-output `estimators_`/`models_`), so `fit()` raises `AttributeError`.
    Use the selected estimator for output 0.
  - Model selection on lag features with shuffled `KFold` leaks future
    information into the training folds. Pass the time-ordered splitter
    `TimeSeriesSplitWrapper` (from `cv.py`) as `cv`.
  - Clone the winner before handing it to `LagFeatureForecaster`.

**(f) `cv.py` — walk-forward CV**

- **Keep:** `walk_forward_splits(n, min_train, horizon, step, max_train)`
  (expanding or rolling window), `WalkForwardCV.evaluate/summary` returning
  per-fold `TSFoldResult` (sMAPE, MASE, WQL), `TimeSeriesSplitWrapper` as an
  sklearn CV object.
- **Fix / change:** pass `y_train` to the fixed `mase` (§2.4b); default
  `step=horizon` rather than 1, because `step=1` means one refit per
  observation; drop the
  `if q_arr is not None` dead branch; make the wrapper subclass
  `BaseCrossValidator` as the docs claim.

**(g) `uncertainty.py`, `figures.py`**

- **Keep:** `gaussian_quantiles`, `conformal_interval_from_residuals`
  (finite-sample `ceil((n+1)(1−α))/n` level, correct),
  `propagate_uncertainty_recursive` (AR(1) error-variance sum, correct for
  |ρ| < 1), and `plot_forecast_result` (quantile fan chart).
- **Fix / change:** `conformal_interval_from_residuals` should use
  `np.quantile(..., method="higher")` to keep the coverage guarantee; the
  `rho == 1` branch is fine; `propagate_uncertainty_recursive` silently takes
  only element 0 of an array input — either vectorize it or reject arrays.

**(h) `prophet_adapter.py`, `neuralforecast_adapter.py`**

- **Keep:** `ProphetForecaster` and `NeuralForecaster` (N-BEATS / N-HiTS)
  with lazy `_require_*()` import guards and the shared `ForecastResult`
  contract; optional extras `[prophet]`, `[neuralforecast]` added to `[all]`.
- **Fix / change:** both map native (lower, median, upper) to arbitrary
  quantiles by linear interpolation, and extrapolate outside the native
  interval. Configure Prophet's `interval_width` / NeuralForecast's
  `levels` from the requested quantiles instead, and only interpolate inside.
  `NeuralForecaster.fit` defers training to `predict` — train in `fit` so
  walk-forward CV timing is meaningful and repeated `predict` calls do not
  retrain.

**(i) Package exports + docs**

- **Keep:** new `time_series/__init__.py` exports; top-level
  `multioutreg/__init__.py` re-exporting `ChronosForecaster`,
  `ForecastResult`, `check_ts_suitability`, `ts_metrics`; the "Time Series
  Forecasters" section of `docs/integrated_surrogates.md`.
- **Fix / change:** top-level imports must stay lazy/optional. Importing
  `multioutreg` must not require `torch`/`chronos`. Keep `check_ts_suitability`
  from `main`'s audited version (audit item 21), not the branch's.

**(j) GUI page `05_Time_Series_Forecasting.py`**

- **Keep:** three tabs — "Chronos (Zero-Shot)" (+ Prophet / NeuralForecast
  when installed), "ARIMA / SARIMA / LSTM", "Surrogate Forecasters"
  (`LagFeatureForecaster` + walk-forward CV summary + fan chart, and an Auto
  option backed by `AutoSurrogateForecaster`).
- **Fix / change:** the page builds its own sklearn-only `_build_surrogate`
  list (8 models). Use the vendored surrogate registry so all surrogates are
  available. Run `check_ts_suitability` before fitting and show its warnings.
  Merge with `main`'s current page, which has audit changes, rather than
  replacing it.
- **Tests:** port the 15 TS unit test files, the integration test, and the
  page test (AppTest) after updating them for the fixes above. Mark
  Prophet/NeuralForecast tests with `pytest.importorskip`.

### 2.5 Multi-objective regret metrics + model download — from `moo-regret`

*Attempt:* new `multioutreg.metrics` subpackage with hypervolume, scalarized
and ε-indicator regret plus an iterative `RegretTracker`; show them with
user-set pass/fail thresholds in the Auto-Detect GUI page and the HTML
report; let users download the trained model as `.joblib`.

Source: `e62d31c` — `multioutreg/metrics/{__init__,multiobjective_regret}.py`,
`multioutreg/metrics/README.md`, `tests/unit/metrics/test_multiobjective_regret.py`,
page 01 diff, `report/template.html` regret section.

- **Keep:**
  - `pareto_mask`, mixed min/max via `minimize=[...]`, default HV reference
    point `worst + 10% span`, exact 1-D/2-D HV, pymoo HV for higher
    dimensions, scalarized regret with flexible weight shapes and `reduce`
    options, additive ε-indicator, `RegretTracker`;
  - GUI: HV reference margin / custom reference, scalarization weights,
    max-acceptable thresholds with ✅/❌ status lines;
  - report: "Multi-Objective Regret" section (`regret_metrics`,
    `regret_bounds`, `regret_settings`);
  - "Download Trained Model (.joblib)" button next to the HTML report download
    (session-state persistence is already on `main`);
  - the README in `multioutreg/metrics/` (rationale for each metric/default),
    moved into Sphinx docs.
- **Fix / change:**
  - **The regret definition is wrong for surrogate evaluation.** The branch
    compares the front of *predicted* values to the front of *true* values
    (`HV(front(Y_true)) − HV(front(Y_pred))`, clipped at 0). An optimistic
    model can then score zero regret while choosing bad designs, and the
    clipping hides it. Standard decision regret evaluates the **true**
    objectives at the points the model **selects**:
    - HV: `HV(front(Y_true)) − HV(Y_true[pareto_mask(Y_pred)])` (≥ 0 by
      construction);
    - scalarized: for each `w`, `x* = argmin_x w·Y_pred(x)`, regret
      `w·Y_true(x*) − min_x w·Y_true(x)` (≥ 0);
    - ε: ε-indicator of `Y_true[pareto_mask(Y_pred)]` vs `front(Y_true)`.
    Keep the current "front-fidelity" comparison as an optional separate
    metric if wanted, but name it differently.
  - Scalarized regret with raw weights is dominated by the largest-scale
    objective. Normalize objectives (e.g. by the true ideal/nadir range)
    before applying weights.
  - The m > 2 Monte Carlo HV fallback is a Python loop over 20 000 samples
    with a fixed seed and no error estimate. Vectorize it, expose `n_samples`,
    and warn when it is used; prefer exact HV via pymoo.
  - `pymoo` was added as a **hard** dependency. Make it an optional extra
    `[moo]` with the 2-D exact path and MC fallback when absent.
  - The branch also bumped `requires-python` to `>=3.10,<4`. Align with
    `main`'s current CI matrix (3.10–3.12) instead of taking the diff.
  - The branch rewrote all of `report/template.html` (≈550 lines) against an
    old base. Port only the regret section into `main`'s template.
  - The `.joblib` download pickles whatever is in session state. Torch-backed
    surrogates should be saved with their state dicts or documented as
    pickle-only. Add a short "loading the model" snippet to the docs.
  - Regret on page 01 only makes sense when outputs are objectives to be
    optimized. Gate it behind a checkbox with per-output min/max selection.
- **Tests:** port `test_multiobjective_regret.py`; add cases where an
  optimistic predictor previously scored 0 regret; 3-objective HV
  against pymoo; mixed min/max.

### 2.6 PCA sampling-detection fallback — from `pca_sample_detection`

*Attempt:* classify the sampling method (Grid/LHS/Sobol/Random) from a 2-D
PCA projection with hand-tuned thresholds on nearest-neighbour-distance std,
silhouette and entropy, as a fallback when UMAP fails; plus two GUI pages and
an early `MultiFidelitySurrogate`.

- **Already superseded on `main`:**
  - classification: `multioutreg/sampling/detect.py`
    (`detect_sampling_method`, audit items 19/20/E9) uses Monte Carlo tests
    of stratification, discrepancy and NN spacing against simulated designs.
    The branch's absolute thresholds depend on scale and n, which is why its
    commit message says "PCA not detecting correctly". Do not port
    `pca_sampling_detector.py`, `classify_sampling.py`, the threshold logic in
    `umap_plot_classify.py`, `PCA_Sample_detection.py`, or
    `05_PCA_Sampling_Datasets_Examples.py` (main's
    `05_Sampling_Datasets_Examples.py` covers it);
  - `MultiFidelitySurrogate`: on `main` in a more complete form.
- **Keep (display only):** in `figures/umap_plot_classify.py`, fall back to a
  PCA projection when `umap` cannot be imported or fails, with axis labels
  and title saying which projection was used. Move `import umap` from module
  top level into the function so the module imports without `umap-learn`.
  Optionally add a user toggle UMAP/PCA on the Sampling dashboard. PCA
  preserves global distances, so it is often the more honest picture of a
  design.
- **Tests:** monkeypatch `umap` import failure → PCA path returns a plot and
  `"projection": "PCA"`.

---

## 3. Suggested rebuild order

1. §2.1 references + §2.2 AL doc (docs only, one PR).
2. §2.4b `mase` fix + §2.4a `ts_dynamic_fit` hygiene (small, unblocks TS work).
3. §2.3 `cv="auto"` + report CV labels + MF page CV selection + notebook.
4. §2.6 PCA display fallback (small).
5. §2.4c–j TS forecaster expansion (split into: lag features + forecaster +
   CV; adapters; GUI page).
6. §2.5 regret metrics (new semantics) + GUI/report + model download.

---

## Before deleting branches

Deleting a branch makes its commits unreachable; GitHub and local `git gc`
will eventually drop them, and the `<sha>:<path>` pointers above stop working.
Archive the tips as tags first (cheap, keeps the source readable):

```bash
for b in crossval-datasize-dep ts-review moo-regret pca_sample_detection integrate-acf; do
  git tag "archive/$b" "origin/$b"
done
git tag archive/active-learning-prototype active-learning-prototype   # local-only branch
git push origin 'refs/tags/archive/*'
```

Then delete:

```bash
# remote (keep gh-pages)
git push origin --delete crossval-datasize-dep ts-review moo-regret \
  pca_sample_detection integrate-acf main-backup-pre-ai
# local
git branch -D audit-critical-fixes integrate-acf active-learning-prototype
```
