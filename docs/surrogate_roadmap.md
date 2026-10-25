# Expanding the Surrogate Offerings

Candidate surrogates and methods for extending the multi-fidelity and
multi-output capabilities of `multioutreg`. See
[Integrated Surrogates](integrated_surrogates.md) for what is available today.

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
  This would sit alongside the `multioutreg.sampling` module (see
  [Active Learning](#active-learning)).
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

`multioutreg` does not do active learning. Every fitting path
(`AutoDetectMultiOutputRegressor.fit`, the grid-search CLIs, the GUI pages and
the multi-fidelity surrogates) fits once on a fixed dataset. There is no
acquisition function and nothing that suggests the next point to sample. The
`multioutreg.sampling` module only infers which design a dataset came from
(Random, LHS or Sobol); its `scipy.stats.qmc` calls simulate comparison designs
and are never used to choose new points.

### What Already Exists

- **Predictive uncertainty:** most surrogates (GPs, Bayesian ridge, forest
  uncertainty, NGBoost, deep ensembles, BNN, ...) implement
  `predict(X, return_std=True)`. `AutoDetectMultiOutputRegressor.predict`
  combines this per output, and
  `metrics_generalized_api._predict_with_uncertainty` also handles
  `return_cov`, `predict_std` and `predict_var`.
- **Adaptive conformal intervals:** `SplitConformalPredictor` with normalized
  scores gives intervals whose width depends on `X`, a coverage-backed measure
  of local uncertainty.
- **QMC generators:** `scipy.stats.qmc` is already a dependency, for the
  initial design and for candidate pools.
- **Multi-fidelity variance:** `MultiFidelitySurrogate.predict(X, level=...,
  return_std=True)`, `StackedVFMSurrogate` and `AdditiveCorrectionVFM` give a
  variance for each fidelity level.
- **Diagnostics:** the DOE pair plots, UMAP plot and sampling detector can show
  where new points land relative to the original design.

### Candidates

| # | Addition | Rationale | Effort |
|---|---|---|---|
| 1 | **Acquisition functions** on `(mean, std)` arrays of shape `(n, n_outputs)` | Max variance (sum or max of standardized std across outputs) for global surrogate accuracy; UCB/LCB, EI and PI for optimization; conformal interval width as a coverage-backed alternative. | Low |
| 2 | **Query-by-committee** for surrogates without `return_std` | Disagreement across a bootstrap ensemble or across the Auto-Detect candidates, so KNN, linear models and plain trees can take part. | Low |
| 3 | **Batch selection** | Combine uncertainty with a distance penalty to points already chosen (k-means++ / maximin style) so a batch does not cluster in one region. | Low–Medium |
| 4 | **Active-learning loop** | Initial LHS/Sobol design → fit → score a QMC candidate pool → pick a batch → call a user-supplied simulator → refit, stopping on a budget or when held-out metrics plateau. | Medium |
| 5 | **Cost-aware multi-fidelity acquisition** | Variance reduction per unit cost, multi-fidelity EI or max-value entropy search across levels (the item under Multi-Fidelity → Supporting Methods). | Medium–High |
| 6 | **CLI and GUI** | A `mor_cli` command that writes a CSV of the next points to run, and a Streamlit page following the existing page patterns. | Low–Medium |

A new `multioutreg/active/` package (or `multioutreg/sampling/acquisition.py`)
would hold items 1–5.

### Issues to Resolve First

- **Zero std read as certainty:** `AutoDetectMultiOutputRegressor.predict`
  fills `std` with zeros when the selected model has no `return_std`. A
  variance-based acquisition would treat those outputs as perfectly known and
  never sample for them. It should return NaN, raise, or fall back to a
  committee estimate.
- **Output scaling:** raw std summed across outputs lets the largest-scale
  output dominate, so stds need standardizing before they are combined.
- **Sampling detection:** an adaptively built design is no longer Random, LHS
  or Sobol and will likely be reported as "Uncertain". Reports should say when
  a design was built adaptively.

## Cross-Cutting Additions

Gaps that are not specific to multi-fidelity or multi-output modelling, but
that the candidates above either depend on or would benefit from.

### Candidates

| # | Addition | Rationale | Effort |
|---|---|---|---|
| 1 | **Benchmark suite** of analytic test problems, in `multioutreg.active.benchmarks` (see [Benchmark Suite](#benchmark-suite) below for what is in and what is next) | Nothing in the package compares surrogates on problems with a known answer. Without it, the multi-fidelity candidates and the accuracy-vs-cost evaluation above have nothing to be judged on. | Low |
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

**Notebooks on the first batch.** Done: the
[Multi-Output Surrogates tutorial](tutorials/multi_output_surrogates.ipynb) uses ZDT1 and ZDT3
(per-output model selection, decision regret against `zdt_front`), Friedman 1 (SHAP and
partial dependence against the five inert inputs) and Gramacy & Lee (GP and conformal
coverage by region). Viennet 2 was left out: all three outputs are quadratics, so a GP fits
each exactly and it says nothing about model selection; Kursawe (next to port) is the better
multi-target example. Still to build:

| Notebook | Benchmarks | What it demonstrates |
|---|---|---|
| Multi-fidelity, scale and warping | Heterogeneous, `branin_mf` (add to `examples/mf_surrogates.ipynb`) | The heterogeneous function's scale between levels varies with `x`, so a constant `rho` cannot fit it and stacking (which sees `x`) should win. In `branin_mf` the low level sees stretched inputs, which no scaled correction matches. |
| Batch active learning | Himmelblau | Four equal minima: a good `select_batch` batch spreads across all four basins. |

**Next to port:**

| Benchmark | Dim → outputs | Use |
|---|---|---|
| DTLZ2 | d → m | Outputs on a sphere, so strongly dependent: tests the multi-task GP, PCA output reduction and copula / energy-score candidates. |
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

- **Default fidelity level:** `MultiFidelitySurrogate.predict` uses the
  *lowest* level when `level` is not given (see the open follow-up to item 1c
  in the [audit checklist](AUDIT_CHECKLIST.md)). It should default to the
  highest level, or require `level`, before the multi-fidelity surrogates are
  added to Auto-Detect.
- **NaN std downstream:** returning NaN instead of zero std (the first issue
  under [Active Learning](#active-learning)) first needs NaN handling in the
  GUI and report plots, which pass `best_std` directly to uncertainty-toolbox.

### Open Follow-ups from Branch Integration

Left over after the old feature branches were rebuilt on `main` (#29–#31):

- **Fitted `rho` in `AdditiveCorrectionVFM`:** found by
  `examples/mf_surrogates.ipynb` (4-level Forrester, reference XC4). `rho` is
  the least-squares slope of `Y_hi` on `f_lo(X_hi)`, which is biased whenever
  the discrepancy `f_hi - rho * f_lo` is correlated with `f_lo`. On Forrester
  f4 → f1 it fits 0.44 where 2 makes the correction linear, and `rho=1` beats
  it up to 18 high-fidelity points. Estimate `rho` jointly with `delta`
  (regress on `[f_lo(X), X]`, or maximise the correction GP's marginal
  likelihood over `rho`, as in Kennedy–O'Hagan).
- **Stacked VFM intervals:** the same notebook shows the 4-level GP stack is
  the most accurate model but only about half the test grid falls inside its
  nominal 95% band, because the top-level `std` treats lower-level
  predictions as exact. Propagate lower-level variance (e.g. by sampling the
  lower-level posteriors) or recalibrate by default.
- **More benchmark notebooks:** the Forrester notebook is the first piece of
  the benchmark suite (candidate 1); the notebooks planned under
  [Benchmark Suite](#benchmark-suite) are the next pieces.
- **UMAP import in the sampling figure:** `figures/umap_plot_classify.py`
  falls back to PCA when UMAP fails, but still runs `import umap` at module
  level, so the module cannot be imported without `umap-learn` and the
  fallback never runs in that case. Move the import into the function and add
  a test that patches the import to fail and expects `"projection": "PCA"`.
- **Regret semantics in `multioutreg.metrics`:** `hypervolume_regret(Y_true, Y_pred)`,
  `scalarized_regret` and `epsilon_regret` still compare the *predicted* front with the
  true front, the definition the branch-integration review flagged for change. With the
  difference clipped at zero, an optimistic model scores zero: in the multi-output tutorial a
  ZDT1 surrogate whose `f2` is shifted down by 0.5 scores 0 while the accurate surrogate scores
  0.042, although both pick the same designs. Decision regret evaluates the true objectives at
  the designs the model selects (`hypervolume_regret(Y_true, Y_true[pareto_mask(Y_pred)])`;
  for scalarized regret, `w·Y_true(argmin w·Y_pred) − min w·Y_true`). Make it the default,
  keep the front comparison under a different name, and update the GUI page 01 and report
  wording.
- **Conformal predictors and package surrogates:** `SplitConformalPredictor` and
  `CVPlusConformalPredictor` copy their estimator with `sklearn.clone`, which fails on the
  package's own surrogates (`GaussianProcessSurrogate` has no `get_params`). The tutorials pass
  scikit-learn regressors instead. Either implement `get_params` / `set_params` on
  `BaseSurrogate` or reuse `_clone_surrogate` from `auto_detect.py`.
- **SHAP figure run time:** `figures.shap_multioutput.generate_shap_plot` uses SHAP's exact
  explainer, which took 86 s for 100 rows of the 10-input Friedman 1 problem with a GP. A
  `KernelExplainer` on k-means background data gave the same ranking in about 2 s. Cap the
  rows or switch explainers for non-tree models.
- **Branch cleanup:** the local `ts-review` branch and
  `origin/main-backup-pre-ai` are fully merged and can be deleted. Tag any
  tip whose source is still needed (e.g. `archive/crossval-datasize-dep` on
  `9f72fa2`) before deleting, so the commit stays reachable.

## Recommended First Batch

Low effort, using dependencies already declared in `pyproject.toml`:

1. The benchmark suite, so the additions below can be compared.
2. `MFKSurrogate` (SMT co-kriging).
3. The PCA output-reduction wrapper.
4. The multi-task GP (GPyTorch).
5. The Gaussian copula wrapper, plus energy-score and variogram-score metrics.
6. Std recalibration and the extrapolation flag.
7. Model persistence.

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
