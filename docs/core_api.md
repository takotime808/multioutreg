# Core API

The classes and functions most workflows need. Every name links to its full entry in
the generated {doc}`api`, which documents all public modules.

## Fit a multi-output surrogate

| Name | What it does |
|---|---|
| {py:class}`~multioutreg.model_selection.auto_detect.AutoDetectMultiOutputRegressor` | Picks the best model and hyperparameters **per output** by cross-validation (the CV scheme is chosen from the sample size), then refits; `predict(X, return_std=True)` gives mean and std |
| {py:class}`~multioutreg.model_selection.screening.ModelScreener` | Cheap pre-screen that drops candidate families unlikely to win, before the full search |
| {py:func}`~multioutreg.model_selection.evaluation.plan_evaluation` / {py:func}`~multioutreg.model_selection.evaluation.run_evaluation` | Choose and run an honest evaluation (hold-out or nested CV, split conformal or CV+) for any estimator factory |
| {py:class}`~multioutreg.gui.grid_search_core.PerTargetGridSearch` | The grid search behind the main app and the `grid_search` command, as an estimator |

## Surrogates

All surrogates follow the scikit-learn API and return `(mean, std)` from
`predict(X, return_std=True)`. A few common ones:

| Name | Notes |
|---|---|
| {py:class}`~multioutreg.surrogates.gp_sklearn.GaussianProcessSurrogate` | Exact GP; the best-calibrated std on small data |
| {py:class}`~multioutreg.surrogates.rf_sklearn.RandomForestSurrogate` | Std from the spread of the trees |
| {py:class}`~multioutreg.surrogates.moe_surrogate.MixtureOfExpertsSurrogate` | Joint multi-output model with gated experts |

The full list, with when to use each, is on {doc}`integrated_surrogates`.

## Multi-fidelity

| Name | What it does |
|---|---|
| {py:class}`~multioutreg.surrogates.stacked_vfm.StackedVFMSurrogate` | Nonlinear recursive coupling across any number of levels (Perdikaris et al. 2017) |
| {py:class}`~multioutreg.surrogates.stacked_vfm.AdditiveCorrectionVFM` | Two levels, `f_hi = ρ·f_lo + δ` (Kennedy & O'Hagan 2000) |
| {py:class}`~multioutreg.surrogates.multi_fidelity.MultiFidelitySurrogate` | Independent model per level; the baseline the others must beat |

## Uncertainty

| Name | What it does |
|---|---|
| {py:class}`~multioutreg.conformal.split_conformal.SplitConformalPredictor` | Distribution-free intervals from a held-out calibration set |
| {py:class}`~multioutreg.conformal.cv_plus.CVPlusConformalPredictor` | CV+ intervals, for data too small to hold out a calibration set |
| {py:func}`~multioutreg.conformal.metrics.conformal_summary` | Coverage and width per output |

## Active learning

| Name | What it does |
|---|---|
| {py:class}`~multioutreg.active.loop.ActiveLearner` | Pool-based loop: fit, score candidates with an acquisition, label the best, repeat |
| {py:class}`~multioutreg.active.multi_fidelity.MultiFidelityActiveLearner` | Chooses the point **and** the fidelity level under a cost budget |
| {py:func}`~multioutreg.active.benchmarks.get_benchmark` | Analytic test problems with a known answer, including multi-fidelity pairs |

## Multi-objective metrics

{py:func}`~multioutreg.metrics.multiobjective_regret.hypervolume_regret`,
{py:func}`~multioutreg.metrics.multiobjective_regret.scalarized_regret`,
{py:func}`~multioutreg.metrics.multiobjective_regret.epsilon_regret` and
{py:class}`~multioutreg.metrics.multiobjective_regret.RegretTracker`; see
{doc}`multiobjective_metrics`.

## Reports

| Name | What it does |
|---|---|
| {py:func}`~multioutreg.report.auto_detect_report.generate_html_report` | The report for an `AutoDetectMultiOutputRegressor` run |
| {py:func}`~multioutreg.report.build.build_report_data` | Collect results into {py:class}`~multioutreg.report.data.ReportData` (verdict, findings, chart data) |
| {py:func}`~multioutreg.report.render.render_report` | Render `ReportData` as one self-contained HTML file |
| {py:class}`~multioutreg.report.verdict.Thresholds` | The pass / caution limits behind the verdict |

How to read the result: {doc}`reading_the_report`.
