# Expanding the Surrogate Offerings

Candidate surrogates and methods for extending the multi-fidelity and
multi-output capabilities of `multioutreg`. See
[Integrated Surrogates](integrated_surrogates.md) for what is available today.

*Last reviewed 2026-10-26, against `main` at #43.*

## Status at a Glance

| Area | Done | Open |
|---|---|---|
| Active learning | Acquisitions, query-by-committee, batch selection, the loop, cost-aware multi-fidelity acquisition (`multioutreg.active`, prototype) | Max-value entropy search, CLI and GUI, NaN std in Auto-Detect |
| Benchmarks | 18 problems in `multioutreg.active.benchmarks.BENCHMARKS`, including DTLZ2; AL, cost-savings, multi-output and hypervolume notebooks | The rest of the port list; the multi-fidelity and Himmelblau notebooks |
| Multi-objective metrics | Exact N-D `hypervolume()` (#36); decision regret as the default, front comparison as `front_*`; joint `energy_score` and `variogram_score` | Multivariate CRPS variants, joint coverage of prediction regions |
| Multi-fidelity surrogates | `MFKSurrogate` (candidate 1: SMT co-kriging, maximum-likelihood `rho`) | The other five candidates; the Stacked VFM interval follow-up |
| Multi-output surrogates | Multi-task GP (1), PCA output reduction (2), Gaussian copula (5); energy and variogram scores | `RegressorChain` (3), natively joint models (4); joint conformal regions |
| Cross-cutting | Benchmark suite (1), std recalibration (2), model persistence (5), extrapolation flag (6) | Items 3, 4, 7–12; wiring 2, 5 and 6 into the GUI, CLI and report |

## Current State

- **Multi-output:** every base surrogate is per-output (one model per output
  column, via `MultiOutputRegressor` or an internal `estimators_` list). Only
  `BNNSurrogate`, `DeepEnsembleSurrogate` and `MixtureOfExpertsSurrogate`
  predict outputs jointly, and they do so through a shared network rather than
  an explicit model of output correlation.
- **Multi-fidelity:**
  - `MultiFidelitySurrogate` fits one surrogate per fidelity level with no
    coupling between levels.
  - `StackedVFMSurrogate` feeds lower-level predictions in as input features
    (NARGP-like).
  - `AdditiveCorrectionVFM` implements a two-level AR1 correction with a
    constant per-output scale `rho`, fit by least squares.
- None of the multi-fidelity models take part in the Auto-Detect or
  grid-search CLIs; they are opt-in only.

## Multi-Fidelity Candidates

Ordered by value relative to implementation effort.

| # | Addition | Rationale | Effort |
|---|---|---|---|
| 1 | **`MFKSurrogate` / `MFKPLSSurrogate`** wrapping `smt.applications.MFK` / `MFKPLS` | True recursive co-kriging (Le Gratiet). Learns `rho` and the correction GP jointly by maximum likelihood, for any number of levels. `smt` is already an optional extra, so it follows the `GPXSurrogate` / `KPLSSurrogate` pattern. | Low |
| 2 | **Input-dependent `rho(x)`** in `AdditiveCorrectionVFM` | Replace the constant per-output scale with a polynomial or GP in `x`, the standard extension of Kennedy–O'Hagan. | Low |
| 3 | **Fidelity as an input** | Treat fidelity as a continuous feature (mesh size, timestep, tolerance) with a product kernel, as in BoTorch's `SingleTaskMultiFidelityGP` or a GPyTorch equivalent. Handles sample points that are not nested across levels, and fidelity that is a dial rather than a fixed set of levels. | Medium |
| 4 | **Proper NARGP** | GP with kernel `k(x)·k(f_lo) + k(x)` on `[x, f_lo(x)]`, with uncertainty propagated by Monte Carlo. `StackedVFMSurrogate` with GP levels approximates this but lacks the kernel structure. | Medium |
| 5 | **Multi-fidelity neural networks** | Either a composite network (Meng & Karniadakis 2020) with linear and nonlinear correction sub-networks, or pretraining on low-fidelity data followed by fine-tuning on high-fidelity data. Reuses `_torch_utils.py` and the deep-ensemble tooling. | Medium |
| 6 | **Multi-fidelity deep GP** (Cutajar et al. 2019, via emukit or GPyTorch) | Strongest option for nonlinear relationships between levels, at a substantially higher implementation and compute cost. | High |

### Supporting Methods

- **Cost-aware choice of the next fidelity to run:** multi-fidelity expected
  improvement, max-value entropy search, or variance reduction per unit cost.
  Variance reduction per unit cost and multi-fidelity EI are in
  `multioutreg.active.multi_fidelity`; max-value entropy search is still to do
  (see [Active Learning](#active-learning)).
- **Evaluation:** accuracy-vs-cost curves, and the value each level adds (e.g.
  high-fidelity error with and without the low-fidelity data).
- **Model selection:** a selection path in Auto-Detect that understands
  fidelity levels, so the multi-fidelity surrogates are no longer opt-in only.

## Multi-Output Candidates

| # | Addition | Rationale | Effort |
|---|---|---|---|
| 1 | **Multi-task GP (ICM / LMC)** via GPyTorch `MultitaskKernel` + `MultitaskGaussianLikelihood` | The standard way to model correlation between outputs in a GP. Also handles outputs observed on different rows. GPyTorch is already an optional extra; can reuse the patterns in `sparse_gp_gpytorch.py`. | Medium |
| 2 | **Output-reduction wrapper (PCA, kernel PCA or POD)** | Wraps any surrogate: fit it on the top-k output components, then map predictions and variances back to the original outputs. The standard approach for field outputs (spectra, pressure distributions, time series) that would otherwise need hundreds of independent models. | Low |
| 3 | **`RegressorChain` wrapper with Monte Carlo uncertainty propagation** | Cheap way to capture outputs that depend on each other in sequence. | Low |
| 4 | **Natively joint models** | `PLSRegression`; a joint-tree random forest (sklearn's `RandomForestRegressor` supports multiple outputs directly, which the current wrapper splits apart); CatBoost with `loss_function="MultiRMSE"`; reduced-rank regression. | Low each |
| 5 | **Gaussian copula on residuals** | Wrapper that adds output correlation to any per-output surrogate: estimate the correlation of normalized residuals and expose `predict(X, return_cov=True)`. `multioutreg.performance.metrics_generalized_api` already accepts `return_cov`. | Low–Medium |

### Supporting Methods

- **Joint metrics:** energy score, variogram score, multivariate CRPS, and
  joint coverage of the prediction region. These test whether output
  correlations are captured, which per-output metrics cannot.
- **Joint conformal regions:** the `multioutreg.conformal` module is per-output.
  Max-norm or Mahalanobis nonconformity scores would give simultaneous coverage
  across all outputs.

## Combined Multi-Output and Multi-Fidelity

A multi-task GP with fidelity as an additional task dimension handles both in
one model, including outputs that only exist at some fidelity levels, which is
common in simulation data.

## Active Learning

The detailed design, including multi-fidelity active learning, is on the
[Active Learning Plan](active_learning.md) page.

### Current State

`multioutreg.active` (prototype, phases 1–2 of the
[Active Learning Plan](active_learning.md#delivery-phases)) provides:

- **Acquisitions** (`acquisition.py`): `max_variance` on standardized std
  (`output_scale`), `upper_confidence_bound`, `expected_improvement` and
  `probability_of_improvement`, with per-output weights and `aggregate`.
- **Query-by-committee:** `CommitteeSurrogate` fits bootstrap copies of any
  surrogate so KNN, linear models and plain trees can take part.
  `predict_mean_std` refuses an all-zero std instead of reading it as
  certainty.
- **Batch selection:** `select_batch` uses greedy local penalization.
- **Loop:** `ActiveLearner` (ask/tell, `space_filling` initial designs,
  `heldout_metrics`).
- **Multi-fidelity:** `fidelity_info`, `level_trust`, `effective_costs`,
  `variance_reduction_per_cost`, `mf_expected_improvement` and
  `MultiFidelityActiveLearner`.

The [Active Learning on Benchmark Functions](tutorials/active_learning_benchmarks.ipynb)
tutorial covers all of these, and the
[Cost Savings](tutorials/cost_savings.ipynb) tutorial prices them against
passive designs. None of it is exposed through `mor_cli`, the GUI or the
reports yet.

### Candidates

| # | Addition | Rationale | Effort | Status |
|---|---|---|---|---|
| 1 | **Acquisition functions** on `(mean, std)` arrays of shape `(n, n_outputs)` | Max variance (sum or max of standardized std across outputs) for global surrogate accuracy; UCB/LCB, EI and PI for optimization; conformal interval width as a coverage-backed alternative. | Low | Done, except the conformal-width score |
| 2 | **Query-by-committee** for surrogates without `return_std` | Disagreement across a bootstrap ensemble or across the Auto-Detect candidates, so KNN, linear models and plain trees can take part. | Low | Bootstrap done; Auto-Detect candidates open (see Cross-Cutting item 7) |
| 3 | **Batch selection** | Combine uncertainty with a distance penalty to points already chosen (k-means++ / maximin style) so a batch does not cluster in one region. | Low–Medium | Done |
| 4 | **Active-learning loop** | Initial LHS/Sobol design → fit → score a QMC candidate pool → pick a batch → call a user-supplied simulator → refit, stopping on a budget or when held-out metrics plateau. | Medium | Done |
| 5 | **Cost-aware multi-fidelity acquisition** | Variance reduction per unit cost, multi-fidelity EI or max-value entropy search across levels (the item under Multi-Fidelity → Supporting Methods). | Medium–High | Variance reduction and MF-EI done; MES open |
| 6 | **CLI and GUI** | A `mor_cli` command that writes a CSV of the next points to run, and a Streamlit page following the existing page patterns. | Low–Medium | Not started |
| 7 | **Integrated variance (ALC / MF-ALC)** for GPs | Scores a candidate by how much it reduces variance over the whole domain rather than at the point itself; phase 3 of the plan. | Medium | Not started |

### Issues to Resolve

- **Zero std read as certainty (partly fixed):** `predict_mean_std` now
  raises on an all-zero std, so the active-learning path is safe.
  `AutoDetectMultiOutputRegressor.predict` itself still fills `std` with zeros
  when the selected model has no `return_std`, so other callers (GUI, reports)
  still see false certainty. Returning NaN needs the downstream NaN handling
  listed under [Cross-Cutting](#issues-to-resolve-first) first.
- ~~**Output scaling**~~: done. `output_scale` standardizes std per output
  before `max_variance` combines them.
- **Sampling detection:** an adaptively built design is no longer Random, LHS
  or Sobol and will likely be reported as "Uncertain". Reports should say when
  a design was built adaptively.

## Cross-Cutting Additions

Gaps that are not specific to multi-fidelity or multi-output modelling, but
that the candidates above either depend on or would benefit from.

### Candidates

All items except 1 are not started.

| # | Addition | Rationale | Effort |
|---|---|---|---|
| 1 | **Benchmark suite** of analytic test problems, in `multioutreg.active.benchmarks` (**done**; see [Benchmark Suite](#benchmark-suite) below for what is in and what is next) | Nothing in the package compared surrogates on problems with a known answer. Without it, the multi-fidelity candidates and the accuracy-vs-cost evaluation above have nothing to be judged on. | Low |
| 2 | **Std recalibration** (variance scaling or isotonic, Kuleshov et al. 2018) | All calibration in the package today is conformal; nothing corrects a model's own `std` when it is over- or under-dispersed. Acquisition functions and the joint metrics consume `std` directly. `uncertainty-toolbox` (a core dependency) already provides a std recalibrator. | Low |
| 3 | **Global sensitivity analysis** (first-order and total Sobol indices) | A primary use of a fitted surrogate, and absent from the package; the SHAP and PDP figures answer a different question. Saltelli/Jansen estimators can run on `scipy.stats.qmc` samples, and repeating them over posterior draws gives indices with uncertainty. | Low |
| 4 | **Gradient-enhanced surrogates** (`smt` GEKPLS) | Many simulators return adjoint gradients; using them cuts the number of samples needed substantially. `smt` is already an optional extra. | Low |
| 5 | **Model persistence** | There is no save/load API for fitted surrogates (`joblib` is used only in reports and time series), so a model trained in the GUI or CLI cannot be reused. Store the package version, feature and output names, and the fidelity levels alongside the model. | Low |
| 6 | **Extrapolation flag** | Nothing warns when a prediction point lies outside the training design. A convex-hull, Mahalanobis or nearest-neighbour distance check fits alongside the existing sampling and DOE diagnostics. | Low |
| 7 | **Ensembling Auto-Detect candidates** | Auto-Detect keeps one winner. Stacking, or a model-averaged mixture of the candidates' predictive distributions, usually beats it, and the candidates' disagreement is the committee estimate that query-by-committee needs (see [Active Learning](#active-learning)). | Medium |
| 8 | **Optimization on the surrogate** | Acquisition functions choose where to sample, but nothing optimizes the fitted surrogate itself: inverse design, constrained optimization, or a Pareto front across outputs (the natural use of a multi-output surrogate). | Medium |
| 9 | **Categorical and mixed-integer inputs** (`smt` mixed-integer kernels, GPyTorch categorical kernels) | Categorical columns are currently dropped (`gui/pages/lhs_detector_app.py`). | Medium |
| 10 | **Missing outputs in per-output wrappers** | Mask missing values per output instead of dropping whole rows. Simpler than the multi-task GP route above, and common when simulation runs fail partway. | Low–Medium |
| 11 | **Hyperparameter search beyond grid search** (random search, Optuna) | Model selection uses `GridSearchCV` only, which scales poorly as candidates and parameters grow. `ray[tune]` and `hyperopt` are commented out of `pyproject.toml`. | Low–Medium |
| 12 | **Extrapolation for out-of-bounds (OOB) predictions** | Item 6 only detects OOB points; nothing controls what the surrogate returns there. Tree models (random forest, extra trees, the gradient-boosting family) hold the boundary value flat, GPs revert to their constant mean, and the networks are unconstrained, so the same OOB query gets very different answers across Auto-Detect candidates. Add a per-surrogate `extrapolation` policy, driven by the item 6 distance: `"warn"` (default), `"clip"` inputs to the training bounds, a fitted linear or polynomial trend added beyond the bounds (or a GP with a linear mean), and `std` inflated with distance from the design. Benchmark each policy on held-out OOB regions of the benchmark-suite problems. | Medium |

### Benchmark Suite

`multioutreg.active.benchmarks` holds NumPy versions of analytic problems with
known answers. Further candidates come from BenchmarkFcns
(<https://github.com/mazhar-ansari-ardeh/BenchmarkFcns>, MIT; ~310
functions). Port the formulas rather than depend on the package: PyPI ships
only a source distribution (needs a C++ compiler and Eigen), and it requires
Python ≥ 3.11 while CI still tests 3.10. The package is still useful in a
test to cross-check the ports.

**In the package:** Forrester, Branin, Currin, Branin–Currin, Ishigami,
Hartmann-3, Park and Borehole; and, from the first BenchmarkFcns batch,
Friedman 1, Gramacy & Lee, Himmelblau, ZDT1–3 (with `zdt_front` for the exact
Pareto front), Viennet 2, multi-fidelity Branin (`branin_mf`) and the
heterogeneous function.

**Notebooks on the benchmarks.** Done: the
[Active Learning on Benchmark Functions](tutorials/active_learning_benchmarks.ipynb) and
[Cost Savings](tutorials/cost_savings.ipynb) tutorials; `examples/hypervolume_regret.ipynb`
(ZDT1, plus a three-objective DTLZ2 defined inline, #36); and the
[Multi-Output Surrogates tutorial](tutorials/multi_output_surrogates.ipynb), which uses ZDT1 and ZDT3
(per-output model selection, decision regret against `zdt_front`), Friedman 1 (SHAP and
partial dependence against the five inert inputs) and Gramacy & Lee (GP and conformal
coverage by region). Viennet 2 was left out: all three outputs are quadratics, so a GP fits
each exactly and it says nothing about model selection; Kursawe (next to port) is the better
multi-target example. Still to build:

| Notebook | Benchmarks | What it demonstrates |
|---|---|---|
| Multi-fidelity, scale and warping | Heterogeneous, `branin_mf` (add to `examples/mf_surrogates.ipynb`) | The heterogeneous function's scale between levels varies with `x`, so a constant `rho` cannot fit it and stacking (which sees `x`) should win. In `branin_mf` the low level sees stretched inputs, which no scaled correction matches. |
| Batch active learning | Himmelblau | Four equal minima: a good `select_batch` batch spreads across all four basins. The AL tutorial's batch section uses Branin (three minima); Himmelblau would make the spread easier to see. |

**Next to port:**

| Benchmark | Dim → outputs | Use |
|---|---|---|
| DTLZ2 | d → m | Outputs on a sphere, so strongly dependent: tests the multi-task GP, PCA output reduction and copula / energy-score candidates. Already written inline in `examples/hypervolume_regret.ipynb`; move it into `benchmarks.py` with a `dtlz_front`. |
| Kursawe | 3 → 2 | One smooth and one non-smooth output; trees should win the second. |
| BNH (with constraints) | 2 → 2 + 2 constraints | Objectives and constraints predicted together; regret on the feasible set. CF1–CF10 are harder versions. |
| Hartmann-6 | 6 → 1 | Standard smooth GP and Bayesian-optimization benchmark. |
| Six-hump camel | 2 → 1 | Six local minima, two global: EI vs max-variance, visually. |
| Step (N1–N3) | any → 1 | Discontinuous: GPs fail and trees win, which shows why model selection matters. |
| Rosenbrock, Styblinski-Tang | scalable → 1 | Dimension and sample-size sweeps for `cv="auto"` and kernel choice. |
| Wing weight, Piston, OTL circuit, Robot arm | 10, 7, 6, 8 → 1 | Engineering functions with physical inputs; each has a low-fidelity level, so they also give higher-dimensional multi-fidelity tests. |
| Xiong 1-D, Gano | 1, 2 → 1 | Small plottable multi-fidelity companions to Forrester. |

**Not worth porting:** the CEC 2014–2022 competition suites (shifted and
rotated optimizer landscapes that need large data files), extremely rugged
optimization functions such as Eggholder, Cross-in-tray and Rana (no
surrogate fits them, so they only show that everything fails), and the
systematic bi-fidelity versions of classic optimization functions.

### Issues to Resolve First

- ~~**Default fidelity level**~~: done. `MultiFidelitySurrogate.predict` now
  uses the *highest* level (the last in `fidelity_levels`) when `level` is not
  given (follow-up to item 1c in the [audit checklist](AUDIT_CHECKLIST.md)).
- **NaN std downstream:** returning NaN instead of zero std (the first issue
  under [Active Learning](#active-learning)) first needs NaN handling in the
  GUI and report plots, which pass `best_std` directly to uncertainty-toolbox.

### Open Follow-ups from Branch Integration

Left over after the old feature branches were rebuilt on `main` (#29–#31).
Open unless marked done:

- **Fitted `rho` in `AdditiveCorrectionVFM`:** found by
  `examples/mf_surrogates.ipynb` (4-level Forrester, reference XC4). `rho` is
  the least-squares slope of `Y_hi` on `f_lo(X_hi)`, which is biased whenever
  the discrepancy `f_hi - rho * f_lo` is correlated with `f_lo`. On Forrester
  f4 → f1 it fits 0.44 where 2 makes the correction linear, and `rho=1` beats
  it up to 18 high-fidelity points. Estimate `rho` jointly with `delta`
  (regress on `[f_lo(X), X]`, or maximise the correction GP's marginal
  likelihood over `rho`, as in Kennedy–O'Hagan). `MFKSurrogate` now does the
  latter and fits `rho` = 2.00 on Forrester, so users needing AR1 have an
  unbiased option; `AdditiveCorrectionVFM` itself is unchanged.
- **Stacked VFM intervals:** the same notebook shows the 4-level GP stack is
  the most accurate model but only about half the test grid falls inside its
  nominal 95% band, because the top-level `std` treats lower-level
  predictions as exact. Propagate lower-level variance (e.g. by sampling the
  lower-level posteriors) or recalibrate by default.
- **More benchmark notebooks:** the Forrester notebook is the first piece of
  the benchmark suite (candidate 1); the notebooks planned under
  [Benchmark Suite](#benchmark-suite) are the next pieces.
- ~~**UMAP import in the sampling figure**~~: done. `figures/umap_plot_classify.py`
  imports `umap` inside `generate_umap_plot`, so the module imports without
  `umap-learn` and the PCA fallback runs; a test reloads it with `umap` unavailable.
- ~~**Regret semantics in `multioutreg.metrics`**~~: done. `hypervolume_regret`,
  `scalarized_regret` and `epsilon_regret` are now decision regret: the true objectives
  evaluated at the candidates on the predicted Pareto front, so `Y_true` and `Y_pred` must be
  the same candidates. An optimistic model no longer scores 0. The front comparison is kept as
  `front_hypervolume_regret`, `front_scalarized_regret` and `front_epsilon_regret`;
  `RegretTracker`, GUI page 01, the report text and both regret notebooks use the new
  semantics. Caveat: identical predicted rows are mutually non-dominated, so a constant
  predictor picks every candidate and scores 0 hypervolume and ε regret (documented in
  `multioutreg/metrics/README.md`).
- ~~**Conformal predictors and package surrogates**~~: done. `CloneMixin` (inherited by every
  `BaseSurrogate` and `ConformalMixin` surrogate) supports `sklearn.clone` and provides generic
  `get_params` / `set_params` built from the constructor arguments, so `SplitConformalPredictor`
  and `CVPlusConformalPredictor` accept the package's own surrogates (tested for every plain
  surrogate). Using them in `GridSearchCV` still needs `__sklearn_tags__` (scikit-learn ≥ 1.6).
- **SHAP figure run time:** `figures.shap_multioutput.generate_shap_plot` uses SHAP's exact
  explainer, which took 86 s for 100 rows of the 10-input Friedman 1 problem with a GP. A
  `KernelExplainer` on k-means background data gave the same ranking in about 2 s. Cap the
  rows or switch explainers for non-tree models.
- **Branch cleanup:** the local `ts-review` branch and
  `origin/main-backup-pre-ai` are fully merged and can be deleted. Tag any
  tip whose source is still needed (e.g. `archive/crossval-datasize-dep` on
  `9f72fa2`) before deleting, so the commit stays reachable.

## Recommended Next Batch

The benchmark suite from the original first batch is done, and the active-learning
prototype went in ahead of the surrogate work. Everything below is low effort and
uses dependencies already declared in `pyproject.toml`.

**Correctness fixes first.** These are small and change results users already
see:

1. ~~Decision regret as the default in `multioutreg.metrics`.~~ Done.
2. ~~`MultiFidelitySurrogate.predict` defaulting to the highest level.~~ Done.
3. ~~`get_params` / `set_params` on `BaseSurrogate`, so the conformal predictors
   accept the package's own surrogates.~~ Done (in `CloneMixin`).
4. ~~A lazy `umap` import in `figures/umap_plot_classify.py`.~~ Done.

**Then the original first batch, minus the benchmark suite:**

5. ~~`MFKSurrogate` (SMT co-kriging). This also replaces the biased least-squares
   `rho` for users who need AR1.~~ Done: `MFKSurrogate` (`surrogates/mfk_smt.py`)
   wraps `MFK` (or `MFKPLS` with `n_comp`) per output, for any number of levels
   and non-nested designs. On Forrester with 20 low- and 5 high-fidelity points
   it fits `rho` = 2.00 (the true value) and an RMSE of 0.002, against 0.44 for
   the least-squares `rho`; `rho_` exposes the fitted factors.
6. ~~The PCA output-reduction wrapper.~~ Done: `PCAOutputSurrogate`
   (`surrogates/pca_output.py`). Fits any surrogate on the top-k components of
   standardized `Y` (`n_components` is a count or a variance fraction); the std
   adds the training truncation error, and `return_cov=True` gives the full
   output covariance.
7. ~~The multi-task GP (GPyTorch).~~ Done: `MultiTaskGPSurrogate`
   (`surrogates/multitask_gp_gpytorch.py`). ICM kernel in Hadamard form with a
   noise level per output; `Y` may contain NaN for unobserved outputs, and
   `output_corr_` holds the learned correlation.
8. ~~The Gaussian copula wrapper, plus energy-score and variogram-score metrics
   (with DTLZ2 ported as their benchmark).~~ Done: `GaussianCopulaSurrogate` (`surrogates/gaussian_copula.py`; `R` from
   rank-normalized out-of-fold residuals), `energy_score`, `variogram_score`
   and `gaussian_samples` in `multioutreg.metrics`, and `dtlz2` / `dtlz_front`
   (any number of objectives) in `multioutreg.active.benchmarks`.
9. ~~Std recalibration and the extrapolation flag.~~ Done:
   `StdRecalibratedSurrogate` (`surrogates/std_recalibration.py`; per-output
   std factors from out-of-fold predictions via uncertainty-toolbox, or
   closed-form NLL) and `ExtrapolationDetector`
   (`sampling/extrapolation.py`; `knn`, `mahalanobis`, `hull` or `box`).
   Neither is wired into the GUI, CLI or report yet.
10. ~~Model persistence.~~ Done: `multioutreg.persistence.save_model` /
    `load_model`, storing input and output names, fidelity levels and package
    versions, and warning on a version mismatch. `load_model` also opens the
    Grid Search app's download and plain `joblib` dumps. The GUI pages still
    use their own serializers.

**Then wire active learning into the product:** the `mor_cli` ask/tell command
and a Streamlit page (Active Learning candidate 6).

New surrogates should follow the existing conventions: inherit from
`ConformalMixin` (or `BaseSurrogate`), implement `predict(X, return_std=...)`,
add a row to [Integrated Surrogates](integrated_surrogates.md), and include
unit tests under `tests/unit/surrogates/`.

## References

- Kennedy, M. C., & O'Hagan, A. (2000). Predicting the output from a complex
  computer code when fast approximations are available. *Biometrika*, 87(1).
- Le Gratiet, L., & Garnier, J. (2014). Recursive co-kriging model for design
  of computer experiments with multiple levels of fidelity. *International
  Journal for Uncertainty Quantification*, 4(5).
- Perdikaris, P., et al. (2017). Nonlinear information fusion algorithms for
  data-efficient multi-fidelity modelling. *Proc. R. Soc. A*, 473.
- Cutajar, K., et al. (2019). Deep Gaussian processes for multi-fidelity
  modeling. *arXiv:1903.07320*.
- Meng, X., & Karniadakis, G. E. (2020). A composite neural network that learns
  from multi-fidelity data. *Journal of Computational Physics*, 401.
- Álvarez, M. A., Rosasco, L., & Lawrence, N. D. (2012). Kernels for
  vector-valued functions: a review. *Foundations and Trends in Machine
  Learning*, 4(3).
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
  prediction, and estimation. *JASA*, 102(477).
- Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global
  optimization of expensive black-box functions. *Journal of Global
  Optimization*, 13(4).
- Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2010). Gaussian
  process optimization in the bandit setting: no regret and experimental
  design. *ICML*.
- Kuleshov, V., Fenner, N., & Ermon, S. (2018). Accurate uncertainties for
  deep learning using calibrated regression. *ICML*.
- Saltelli, A., et al. (2010). Variance based sensitivity analysis of model
  output. Design and estimator for the total sensitivity index. *Computer
  Physics Communications*, 181(2).
- Bouhlel, M. A., & Martins, J. R. R. A. (2019). Gradient-enhanced kriging
  for high-dimensional problems. *Engineering with Computers*, 35(1).
- Settles, B. (2009). Active learning literature survey. *Computer Sciences
  Technical Report 1648*, University of Wisconsin–Madison.
