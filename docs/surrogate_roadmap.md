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
  This would sit alongside the `multioutreg.sampling` module.
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

## Recommended First Batch

Low effort, using dependencies already declared in `pyproject.toml`:

1. `MFKSurrogate` (SMT co-kriging).
2. The PCA output-reduction wrapper.
3. The multi-task GP (GPyTorch).
4. The Gaussian copula wrapper, plus energy-score and variogram-score metrics.

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
