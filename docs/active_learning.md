# Active Learning Plan

How `multioutreg` will choose the next points to evaluate, so that a
surrogate reaches a target accuracy (or finds a good design) with fewer runs
of an expensive simulator or experiment. This page expands the
[Active Learning](surrogate_roadmap.md#active-learning) section of the
surrogate roadmap into a design and a delivery plan. Citations (AL1–AL22,
MF\*) refer to the tables on the {doc}`references` page.

## Goals and Non-Goals

**Goals**

1. **Global surrogate accuracy** is the main use case: spend a fixed budget of
   simulator runs so the fitted surrogate is accurate across the whole input
   domain, for every output (AL20, AL21, AL22).
2. **Optimization** of one output, or of a weighted sum of outputs, through
   the classical Bayesian optimization criteria (EI, PI, UCB) (AL2, AL3, AL4).
3. **Multi-fidelity** problems: choose both *where* to sample and *at which
   fidelity level*, trading information against cost (see
   [Multi-Fidelity Active Learning](#multi-fidelity-active-learning)).
4. Work with **any surrogate in the package**, not only GPs, by falling back
   to committee disagreement when a model has no predictive std.
5. A **simulator-agnostic loop**: an `ask` / `tell` interface for simulators
   that run outside Python (HPC queues, lab experiments), and a `run` helper
   for Python callables.

**Non-goals (for now)**

- Replacing BoTorch or emukit. Gradient-based acquisition optimization,
  Monte-Carlo batch acquisitions (qEI, qNEI) and knowledge-gradient methods
  are out of scope; acquisitions are scored on a finite candidate pool.
- Multi-objective optimization (hypervolume improvement, ParEGO).
- Constrained optimization and categorical inputs.

## Current State

`multioutreg` does not do active learning. Every fitting path
(`AutoDetectMultiOutputRegressor.fit`, the grid-search CLIs, the GUI pages and
the multi-fidelity surrogates) fits once on a fixed dataset. The
`multioutreg.sampling` module only *infers* which design a dataset came from
(Random, LHS or Sobol).

What can be reused:

| Building block | Where | Used for |
|---|---|---|
| `predict(X, return_std=True)` | GPs, Bayesian ridge, forest uncertainty, NGBoost, deep ensembles, BNN, sparse GPs, SMT models | Variance-based and BO acquisitions |
| Normalized split-conformal intervals | `multioutreg.conformal.SplitConformalPredictor` | Coverage-backed uncertainty score |
| `scipy.stats.qmc` (Sobol, LHS, Halton) | dependency already | Initial designs and candidate pools |
| Per-level predictions | `MultiFidelitySurrogate.predict(X, level=...)`, `StackedVFMSurrogate.predict(X, level=...)` | Multi-fidelity acquisition |
| AR1 decomposition | `AdditiveCorrectionVFM` (`surrogate_lo_`, `surrogate_delta_`, `rho_`) | Splitting high-fidelity variance by level |
| DOE / UMAP plots | `multioutreg.figures.doe_plots`, `umap_plot_classify` | Showing where new points land |

## Architecture

A new package, `multioutreg/active/`:

| Module | Contents |
|---|---|
| `acquisition.py` | Uncertainty extraction (`predict_mean_std`, `CommitteeSurrogate`), output standardization and aggregation, and acquisition functions on `(mean, std)` arrays: `max_variance`, `upper_confidence_bound`, `expected_improvement`, `probability_of_improvement` |
| `batch.py` | `select_batch`: greedy top-k with a distance penalty around already-chosen points (local penalization / maximin style) |
| `loop.py` | `ActiveLearner`: initial design, candidate pool, `ask` / `tell` / `run`, history and stopping rules |
| `multi_fidelity.py` | Per-level variance reduction, cost-aware acquisitions, and `MultiFidelityActiveLearner` |
| `benchmarks.py` | Analytic test functions, including standard multi-fidelity pairs, for tests, docs and the tutorial notebook |

### Conventions

- **Shapes.** Every acquisition takes `mean` and `std` arrays of shape
  `(n_candidates, n_outputs)` and returns a score of shape `(n_candidates,)`;
  larger is better. Acquisitions are plain functions, so they can be used
  without the loop.
- **Inputs live in a box.** The learner takes `bounds` of shape
  `(n_features, 2)`. Candidate pools are drawn in the unit cube and scaled;
  distances for batch diversity are measured in the unit cube, so input units
  do not matter.
- **Model factory.** The learner takes a zero-argument callable that returns
  an unfitted surrogate, and refits from scratch each iteration. That is
  simpler and safer than warm starts, and the fit cost is negligible next to
  an expensive simulator.

### Uncertainty Sources

1. **Native std:** `model.predict(X, return_std=True)`.
2. **Committee (query-by-committee, AL7, AL8):** `CommitteeSurrogate` wraps
   any model factory, fits `n_members` copies on bootstrap resamples, and
   reports the mean and the standard deviation across members. This brings
   KNN, linear models and plain trees into active learning.
3. **Conformal width (later):** half-width of normalized split-conformal
   intervals as the score, for a coverage guarantee on the uncertainty itself.

**Zero std is not certainty.** `BaseSurrogate.predict(return_std=True)` and
`AutoDetectMultiOutputRegressor.predict` return zeros for an output whose
model has no native uncertainty. A variance-based acquisition would treat that
output as perfectly known and never sample for it. `predict_mean_std` raises
when an output's std is identically zero and points to `CommitteeSurrogate`.
Changing `AutoDetectMultiOutputRegressor` to return NaN there is tracked
separately; it changes a public return value.

### Multi-Output Aggregation

Raw stds summed across outputs let the largest-scale output dominate. Each
output's std is divided by the standard deviation of that output's observed
training values, then combined with `"sum"` (default; total standardized
variance), `"max"` (the worst-known output drives sampling), or `"mean"`.
Optional per-output `weights` express priorities between outputs.

For the optimization criteria, outputs are first scalarized with `weights`
(`g = Y @ w`), and the std of `g` is `sqrt(sum w_j^2 std_j^2)`, which assumes
independent outputs. Per-output surrogates are independent by construction,
so this matches what the models actually say.

### Acquisition Functions

| Name | Formula (per candidate) | Goal |
|---|---|---|
| `max_variance` | `agg_j (std_j / s_j)^2` | Global accuracy (ALM, AL5) |
| `upper_confidence_bound` | `±mu_g + sqrt(beta) · sigma_g` | Optimization (GP-UCB, AL3) |
| `expected_improvement` | `I · Φ(I / sigma_g) + sigma_g · φ(I / sigma_g)`, `I = best - mu_g - xi` (minimizing) | Optimization (EGO, AL2, AL19) |
| `probability_of_improvement` | `Φ((best - mu_g - xi) / sigma_g)` | Optimization (AL4) |

Integrated-variance reduction (ALC, AL6) is the better global-accuracy
criterion, because it rewards points that reduce uncertainty in their
neighbourhood rather than only at themselves. It needs a joint posterior
covariance, so it comes after the GP-specific helpers in phase 3.

### Batch Selection

With batch size `q > 1`, taking the top `q` scores clusters the batch on one
ridge of the acquisition surface. `select_batch` picks greedily: after each
pick, every remaining candidate's score is multiplied by
`1 - exp(-d^2 / (2 l^2))`, where `d` is the unit-cube distance to the pick
(AL10, AL11, AL12). The default `l` is the typical spacing of a space-filling
design of `n_train + q` points, `(n_train + q)^(-1/d)`. Training points can
be penalized as well (`penalize_train=True`) for models whose std does not
shrink near data (committees of trees).

### The Loop

```text
initial design (scrambled Sobol or LHS, n_init points)
  └─► evaluate simulator ─► fit surrogate ─► draw candidate pool (fresh Sobol each iteration)
        ▲                                         │
        │                                         ▼
        └──── tell(X_new, Y_new) ◄── ask(q): score pool, select_batch
```

- `ask(n)` returns the next `n` points without evaluating anything; `tell(X, Y)`
  appends results and marks the model stale. `run(simulator, n_iter, ...)`
  chains them for Python simulators.
- **History:** one record per iteration (number of samples, acquisition max,
  mean standardized std over the pool, and held-out metrics when test data is
  supplied) for convergence plots.
- **Stopping rules:** budget (number of evaluations); maximum standardized std
  over the pool below `std_tol`; held-out RMSE improvement below a relative
  tolerance for `patience` iterations.
- **Reproducibility:** one `random_state` seeds the initial design and every
  candidate pool.

## Multi-Fidelity Active Learning

With several fidelity levels, each query is a pair `(x, level)`. A
low-fidelity run is cheap but only partly informative about the
high-fidelity response; a high-fidelity run is fully informative but
expensive. The aim is the most accurate **high-fidelity** surrogate per unit
of **total cost**.

### Problem Setup

- Levels are ordered cheapest to most expensive, `l = 0, ..., L-1`, with known
  costs `c_l` per evaluation (wall-clock or core-hours), given by the user.
  Cost depending on `x` (mesh size, convergence) is out of scope for the first
  version.
- One simulator per level: `simulators = {"lo": f_lo, "hi": f_hi}`.
- A total budget `B` in cost units. Accuracy is tracked against cumulative
  cost, not against the number of samples.
- The surrogate is any multi-fidelity model in the package:
  `AdditiveCorrectionVFM` (two levels) or `StackedVFMSurrogate` (any number).

### Information Per Level

The acquisition needs to know how much a run at level `l` and point `x` would
reduce uncertainty about `f_hi(x)`.

**AR1 models (`AdditiveCorrectionVFM`).** The model is
`f_hi(x) = rho · f_lo(x) + delta(x)`, so the high-fidelity variance splits into
two parts:

```text
var_hi(x) = rho^2 · var_lo(x) + var_delta(x)
```

- A **low-fidelity** run at `x` can only reduce the first term:
  `R_lo(x) = rho^2 · var_lo(x)`.
- A **high-fidelity** run updates `delta`. The AR1 fit needs `f_lo` at the
  high-fidelity points, so by default a high-fidelity query also runs the low
  fidelity at the same `x` (a *nested* design, which also suits the
  out-of-fold logic in `_OOFPredictor`). It then reduces both terms,
  `R_hi(x) = rho^2 · var_lo(x) + var_delta(x)`, and costs `c_hi + c_lo`.
  With `nested=False` it reduces only `var_delta(x)` and costs `c_hi`.

This decomposition corrects itself: once the low-fidelity model is well
sampled, `var_lo` is small everywhere, `R_lo` falls towards zero, and the
remaining `var_delta`, which only high-fidelity runs can reduce, takes over.
The cheap level cannot absorb the whole budget unless it really is that
informative.

**Other models (`StackedVFMSurrogate` and anything with
`predict(X, level=...)`).** There is no analytic split, so the plan follows
the correlation factor of multi-fidelity EI (Huang et al. 2006, AL16):

```text
R_l(x) = corr_l^2 · var_hi(x),   corr_{L-1} = 1
```

where `corr_l` is the correlation between level-`l` and high-fidelity
predictions over the candidate pool (one value per output, clipped to
`[0, 1]`). This is cruder: it does not change with `x`, and
`StackedVFMSurrogate` reports only the top level's own std (see its
docstring). It is the fallback, not the main path.

### Cost-Aware Acquisitions

| Name | Score for `(x, l)` | Goal |
|---|---|---|
| `variance_reduction_per_cost` | `agg_j R_{l,j}(x) / s_j^2 / C_l` | Global high-fidelity accuracy |
| `mf_expected_improvement` | `EI_hi(x) · corr_l(x) · C_{L-1} / C_l` (Huang et al. 2006) | Optimization of the high-fidelity output |
| MF max-value entropy search (later) | Information gain about `max f_hi` per unit cost (AL14, AL15) | Optimization, better calibrated than MF-EI |

`C_l` is the effective cost of a query at level `l` (`c_hi + c_lo` for a
nested high-fidelity query). The learner scores every `(x, l)` pair in the
pool and takes the best. For MF-EI with AR1 models, `corr_l(x)` is
`rho · std_lo(x) / std_hi(x)` for the low level and 1 for the high level.

### Safeguards

- **Seeding:** each level starts with its own space-filling design;
  high-fidelity seed points are a subset of the low-fidelity ones, so the
  initial design is nested.
- **Minimum high-fidelity share:** `min_hi_fraction` forces a high-fidelity
  query whenever high-fidelity spending falls below that share of the budget
  spent so far. This guards against a low-fidelity model that is
  overconfident (small `var_lo`) but biased.
- **Budget overrun:** a query whose cost exceeds the remaining budget is
  skipped in favour of the best affordable one; the loop stops when nothing
  is affordable.
- **Evaluation:** accuracy vs cumulative cost against two baselines: random
  high-fidelity sampling, and single-fidelity active learning on the high
  level alone. Report the share of the budget spent at each level.

### Open Questions

- Pointwise reduction ignores that a run at `x` also informs nearby points.
  An integrated (ALC-style) multi-fidelity criterion needs joint covariances
  per level, which only the GP surrogates provide.
- Batches that mix levels (for example 1 high-fidelity run plus 10
  low-fidelity runs per HPC allocation) need a batch-cost model; the
  prototype picks one query at a time.
- `StackedVFMSurrogate` should propagate lower-level uncertainty (Monte
  Carlo through the augmented features) before its std is used for
  acquisition; until then the correlation fallback is used.
- Fidelity as a continuous input (roadmap MF item 3) would turn the level
  choice into a search over a continuous cost dial.

## Benchmarks

`multioutreg.active.benchmarks` will provide analytic functions with known
bounds, so tests, docs and the notebook share one definition:

| Function | Dim | Outputs | Fidelities | Use |
|---|---|---|---|---|
| Forrester | 1 | 1 | hi / lo | Visual 1-D demos, MF |
| Branin | 2 | 1 | hi | Global accuracy and EI |
| Currin exponential | 2 | 1 | hi / lo (Xiong et al. 2013) | MF |
| Branin–Currin | 2 | 2 | hi | Multi-output aggregation |
| Ishigami | 3 | 1 | hi | Strong nonlinearity, QBC |
| Hartmann-3 | 3 | 1 | hi | Optimization |
| Park | 4 | 1 | hi / lo | MF in 4-D |
| Borehole | 8 | 1 | hi / lo | MF in 8-D, physical units |

## Testing Plan

- **Unit:** acquisition values on hand-computed cases (EI at `std = 0`, UCB
  sign under `minimize`, PI bounds), standardization invariance (scaling one
  output does not change `max_variance`), zero-std detection, batch diversity
  (no two picks closer than `l / 2` on a flat score), and benchmark values at
  known optima (Branin, Hartmann-3).
- **Loop:** `ask` / `tell` round-trip shapes; the budget is respected;
  reproducible with a fixed seed; active learning beats random sampling on
  Forrester with a GP at a fixed budget (seeded, loose margin).
- **Multi-fidelity:** costs are accounted exactly; `min_hi_fraction` holds;
  with an uninformative low level (`corr ≈ 0`) the learner spends almost
  nothing on it; with a perfectly correlated, 10x cheaper low level it
  prefers low-fidelity runs.

## Delivery Phases

| Phase | Scope | Status |
|---|---|---|
| 1 | `acquisition.py`, `batch.py`, `loop.py`, `benchmarks.py`, unit tests, tutorial notebook | Prototype |
| 2 | `multi_fidelity.py` with AR1 variance split, correlation fallback, cost-aware acquisitions, `MultiFidelityActiveLearner` | Prototype |
| 3 | GP-specific integrated-variance (ALC) and MF-ALC; conformal-width score; NaN std in `AutoDetectMultiOutputRegressor` | Planned |
| 4 | `mor_cli` command that writes a CSV of the next points (ask/tell through files), a Streamlit page, and an "adaptive design" note in reports and sampling detection | Planned |
| 5 | MF max-value entropy search; validation against BoTorch reference implementations (AL17) | Planned |

The prototype lives on the `active-learning-prototype` branch; the tutorial
is `docs/tutorials/active_learning_benchmarks.ipynb` on that branch.

## Risks

- **Model refit cost.** Refitting a GP every iteration is `O(n^3)`; past a
  few hundred points, use a sparse GP or a larger batch.
- **Miscalibrated std.** Acquisitions trust the std they are given. Tree
  ensembles and deep ensembles are often overconfident away from data; the
  diagnostics (calibration plots in `figures/uct_calibration_extension.py`)
  should be checked on the final design.
- **Adaptive designs fail the sampling detector.** An actively built design
  is neither random, LHS nor Sobol and will be reported as "Uncertain";
  reports must say the design was built adaptively.
