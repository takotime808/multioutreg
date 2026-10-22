# Parallelization and Distributed Computing Plan

How `multioutreg` will spend less wall-clock time per run: first by removing
work it does not need to do, then by running the remaining independent fits in
parallel, and finally by moving heavy jobs off the web app onto more compute.
The public GUI runs on **Streamlit Community Cloud**
([multioutreg.streamlit.app](https://multioutreg.streamlit.app/)), so most of
this page is about getting the best out of a small, shared container. The last
sections cover what more compute would unlock.

## Goals and Non-Goals

**Goals**

1. Every GUI page finishes a typical run (≤ 2 000 rows, ≤ 10 inputs, ≤ 5
   outputs) on Streamlit Community Cloud without hitting the resource limits.
2. One place to control parallelism (`n_jobs`), so library code, the CLI and
   the GUI all respect the same CPU and memory budget.
3. Parallel results are **identical** to serial results for the same seed.
4. A clear path to bigger hardware (workstation, cloud VM, GPU, cluster)
   without rewriting the model-selection code.

**Non-goals (for now)**

- Multi-node training of a single model (data-parallel neural nets, distributed
  GP solvers).
- A paid hosting migration. Hosting alternatives are listed so the decision
  can be made later.
- Real-time multi-user scaling of the public demo beyond what Community Cloud
  provides.

## Deployment Target: Streamlit Community Cloud

Approximate limits from the Streamlit docs (stated as of February 2024; they
"may change at any time without notice"):

| Resource | Minimum | Maximum | What it means for us |
|---|---|---|---|
| CPU | 0.078 cores | 2 cores | The best possible speed-up from parallelism is **about 2×**, and less under load. Removing work matters more than parallelizing it. |
| Memory | 690 MB | 2.7 GB | Process-based workers each re-import numpy, sklearn, torch and shap. Two or three of them can use up the whole budget. |
| Storage | none | 50 GB | Enough for the dependencies. Large wheels (CUDA builds of `torch`) slow down cold starts. |
| Sleep | 12 hours without traffic | n/a | First visit after sleep pays a full cold start (dependency import plus any warm-up). |
| Paid upgrade | None | n/a | Streamlit has said there are no plans for paid Community Cloud tiers. More compute means hosting somewhere else. |

Sources: [Streamlit docs: Manage your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app),
[Common app problems: Resource limits](https://blog.streamlit.io/common-app-problems-resource-limits),
[forum FAQ: "This app has gone over its resource limits"](https://discuss.streamlit.io/t/faq-this-app-has-gone-over-its-resource-limits/62973).

Other Streamlit behaviour that affects runtime:

| Behaviour | Effect today | Mitigation |
|---|---|---|
| The whole script reruns on every widget interaction | A download click or expander toggle can rebuild results that were already computed | `st.session_state` for results, `st.cache_data` / `st.cache_resource` for pure functions and fitted models |
| One Python process serves all sessions | Two users running a grid search at once share the same 2 cores and 2.7 GB | Cap `n_jobs` per session; refuse or queue oversized jobs |
| Container CPU count is not the host CPU count | `os.cpu_count()` can report the host's cores, far more than 2 | Size pools with `joblib.cpu_count()` (which respects cgroup quotas) or an explicit setting, never `os.cpu_count()` or `n_jobs=-1` |
| No background workers | Long jobs block the session until they finish | Progress bars now. A job queue on separate compute later (see [Distributed Options](#distributed-options)) |

## Current State

Joblib appears in exactly one compute path, and nothing in the GUI is cached.

| Area | Where | Parallel today? | Notes |
|---|---|---|---|
| SARIMA order search | `time_series/ts_dynamic_fit/algs/sarima.py:120` | `Parallel(n_jobs=-1)` | The only explicit parallel loop. `-1` means "all cores", which is risky on a shared container |
| Auto-detect model selection | `model_selection/auto_detect.py:305-308` | No | `GridSearchCV(...)` without `n_jobs`, one output and one estimator at a time |
| Grid Search page | `gui/grid_search_core.py::select_per_target` | No | Loops over configs × outputs |
| Multi-fidelity page | `gui/pages/02_Multi_Fidelity_Surrogate_Models.py::_select_best_candidate` | No | One candidate at a time, with a progress bar |
| CV+ conformal | `conformal/cv_plus.py` | No | K fold models fitted one after another |
| Forest noise estimate | `surrogates/forest_uncertainty.py:72` | No | K-fold refits of the forest |
| Deep ensembles | `surrogates/deep_ensemble_pytorch.py:144` | No | Members trained in sequence; torch uses its own intra-op threads |
| Sampling-method detection | `sampling/detect.py:307-309` | No | `n_sim` simulated designs (199 for n ≤ 500) per candidate method |
| Active-learning committee | `active/acquisition.py::CommitteeSurrogate` | No | Members fitted in sequence |
| Streamlit caching | all GUI pages | No `st.cache_data` / `st.cache_resource` anywhere | Results live only in local variables for the current rerun |
| Native threads | LightGBM, XGBoost, CatBoost, torch, BLAS | Implicit | Each library picks its own thread count. Combined with joblib this oversubscribes the CPU. On macOS it can also deadlock (see the OpenMP note in the test instructions) |

### Fix first: the Grid Search page selects twice

`gui/Grid_Search_Surrogate_Models.py` contains a duplicated block. After
`select_per_target` picks a model on a validation split (line ~437), the page
runs the following again (lines ~461–590):

1. **PCA a second time** on inputs that were already PCA-transformed (line ~467).
2. **A Cartesian search over every combination of configs across outputs**
   (`ParameterGrid({"combos": [configs] * n_outputs})`, line ~547). It scores
   each combination on the **test set** and overwrites the first result.

The second search grows exponentially with the number of outputs. It also
undoes the leakage fix that `select_per_target` was written for. Deleting the
duplicated block is the largest runtime win on this page, and it is a
correctness fix as well.

## Where the Time Goes

Fit counts for one run, measured from the default candidate lists in this
environment. Adding LightGBM, XGBoost or CatBoost increases them. `C` is the
number of grid configs, `k` the number of CV folds and `m` the number of outputs.

| Workload | Formula | m = 1 | m = 3 | m = 5 | Independent units |
|---|---|---|---|---|---|
| Grid Search page, `select_per_target` (C = 77) | `C·m + m` | 78 | 234 | 390 | every (config, output) pair |
| Grid Search page, duplicated Cartesian block | `m·Cᵐ` | 77 | ≈ 1.37 M | ≈ 1.4 × 10¹⁰ | every combination (should be deleted) |
| Auto-detect (21 estimators, 93 grid points, k = 3) | `(93·k + 21)·m` | 300 | 900 | 1 500 | every (estimator, params, fold, output) |
| Multi-fidelity page, 2 levels, 6 base surrogates | `5·6` candidates × levels (+ stacked out-of-fold fits) | ≈ 30 candidates | same | same | every candidate |
| Sampling detection (n ≤ 500) | `199 × candidate methods` | 199 per method | same | same | every simulated design |
| SARIMA order search | product of the p, q, P, Q range sizes | grid size | — | — | every order |
| CV+ conformal | `k` (+1 refit) | 6 for k = 5 | same | same | every fold |
| Deep ensemble | `n_estimators` networks | 5 | same | same | every member |

Almost every hot loop is **embarrassingly parallel**: the units do not share
state, and the results are reduced with a simple `min` or `mean` at the end.
This makes them good candidates for `joblib`.

## Parallelization Options

| Option | Granularity | Memory cost | Fits Community Cloud? | Good for | Watch out for |
|---|---|---|---|---|---|
| **Remove or cache work** (delete duplicate block, `st.cache_*`, smaller grids) | n/a | Saves memory | **Yes, the best value** | Everything | Cache keys must include the data hash and every setting |
| **joblib, `loky` (processes)** | one fit per task | **High**: each worker re-imports the stack (hundreds of MB with torch, shap) | Only with `n_jobs=2` and lazy imports | Pure-Python or GIL-holding fits (GP optimizer restarts, statsmodels) | Pickling cost for large `X`; workers outlive the request |
| **joblib, `threading`** | one fit per task | Low (shared memory) | **Yes** | Fits that release the GIL: sklearn trees, LightGBM, XGBoost, numpy/BLAS-heavy code | No gain for code that holds the GIL |
| **Native library threads** (`n_jobs` on RF/ET, `nthread` in LightGBM/XGBoost, `torch.set_num_threads`) | inside one fit | Low | Yes | Large single models | Multiplies with joblib workers. Set one level, not both |
| **`threadpoolctl`** | caps BLAS/OpenMP pools | none | Yes | Preventing oversubscription | Must wrap the parallel region |
| **Dask (`joblib` backend or `dask.distributed`)** | task graph, multi-node | Scheduler and workers on separate machines | No (needs external workers) | Large grids on a cluster, data larger than memory | Ops overhead; a separate service to run |
| **Ray (`ray.util.joblib`, Ray Tune)** | task / trial, multi-node | Separate cluster | No | Hyper-parameter search at scale, AL loops with many simulators | Heavy install (already only an `examples` extra) |
| **Job queue (RQ, Celery, Cloud Tasks) + workers** | one whole run per job | Separate compute | **Front-end only**: Streamlit submits, workers compute | Long reports and big grid searches without blocking the UI | Needs storage for results and a job-status page |
| **Serverless fan-out (Modal, AWS Lambda, Cloud Run jobs)** | one fit or one run per call | Pay per use | Front-end only | Bursty, occasional heavy jobs | Cold starts; package size limits |
| **GPU** (XGBoost `device="cuda"`, LightGBM `device="gpu"`, torch, GPyTorch) | inside one fit | GPU memory | No GPU on Community Cloud | Deep ensembles, BNN, SGPR, large boosting | Small data (< ~10k rows) is often faster on CPU |

## Plan for Streamlit Community Cloud

With at most 2 cores, the order of the work matters: **remove work, then
cache, then parallelize at one level**.

### Phase 0: Remove work (no new dependencies)

| Item | Where | Expected effect |
|---|---|---|
| Delete the duplicated PCA and Cartesian selection block | `gui/Grid_Search_Surrogate_Models.py` ~461–590 | From exponential to linear in outputs; restores the validation-split selection |
| Default `skip_expensive=True` on the hosted app, off locally | GUI sidebars | Drops exact GP, ARD GP, GPX, KPLS, SGPR from the default run |
| Size guards: warn above N rows / m outputs / C configs, offer to subsample | GUI pages | Prevents runs that cannot finish within the limits |
| Lazy-import heavy optional libraries (`torch`, `shap`, `umap`, `chronos`) inside the functions that use them | figures, surrogates, pages | Lower baseline memory, faster cold start, cheaper process workers |
| Bound SHAP cost: summarize the background with `shap.sample` / `shap.kmeans` and explain at most ~200 rows | `figures/shap_multioutput.py` | Model-agnostic `shap.Explainer` cost no longer grows with `n²` |
| Use a CPU-only `torch` wheel for the hosted build, if the build tool allows it | deployment requirements | Smaller install, faster cold start after sleep |

### Phase 1: Caching and session state

| What | Streamlit API | Cache key |
|---|---|---|
| Parsed upload and missing-value summary | `st.cache_data` | file bytes hash |
| `ModelScreener` results and sampling-method detection | `st.cache_data` | data hash + settings |
| Fitted best model and its predictions | `st.cache_resource` (or `st.session_state`) | data hash + every sidebar setting + seed |
| Report HTML / figures | `st.session_state` | set once per run, reused by download buttons |
| Re-rendering only the report area | `st.fragment` | n/a |

Use a size bound (`max_entries`) and a TTL on the caches, so memory stays
under 2.7 GB when several users upload different datasets.

### Phase 2: One parallelism setting, used everywhere

Add `multioutreg/utils/parallel.py` with a single source of truth:

```python
def get_n_jobs(n_jobs=None) -> int:
    """Explicit argument > MULTIOUTREG_N_JOBS env var > joblib.cpu_count(), capped at 2 on Streamlit Cloud."""
```

and a context manager that caps native thread pools inside parallel regions:

```python
with parallel_region(n_jobs) as n:     # threadpoolctl.threadpool_limits(1) for BLAS/OpenMP
    results = Parallel(n_jobs=n, prefer="threads")(delayed(fit_one)(...) for ...)
```

| Call site | Change | Backend on Cloud | Why |
|---|---|---|---|
| `AutoDetectMultiOutputRegressor` | new `n_jobs` parameter, passed to `GridSearchCV(n_jobs=...)`; outer loop over outputs stays serial | threads | sklearn already parallelizes over (params, fold); one level only |
| `select_per_target` | `Parallel` over (config, output) pairs; inner `n_jobs=1` | threads | 234 independent fits for m = 3 |
| `_select_best_candidate` (multi-fidelity) | `Parallel` over candidates; progress updated as results arrive (`return_as="generator_unordered"`) | threads | about 30 independent candidates |
| `CVPlusConformalPredictor.fit` | `Parallel` over folds | threads | k independent fold models |
| `estimate_noise_variance` | `Parallel` over folds, or rely on forest `n_jobs` | threads | do not use both levels |
| `detect_sampling_method` | `Parallel` over simulated designs, vectorize the statistics where possible | processes locally, threads on Cloud | pure numpy/scipy, cheap to pickle |
| SARIMA `Parallel(n_jobs=-1)` | replace with `get_n_jobs()` | processes | statsmodels holds the GIL; `-1` is unsafe on shared hosts |
| `DeepEnsembleSurrogate`, `CommitteeSurrogate` | optional `n_jobs` over members; `torch.set_num_threads(1)` inside workers | processes (local / VM only) | members are independent |

**Rules**

- Parallelize at **one level only**. When the outer loop is parallel, inner
  estimators get `n_jobs=1` and native thread pools are capped at 1.
- Default to the `threading` backend on Community Cloud. Use `loky` only where
  profiling shows the GIL is the bottleneck, and only once Phase 0's lazy
  imports keep each worker small.
- Keep results reproducible: derive every task's seed from the parent seed
  (`np.random.SeedSequence(seed).spawn(n)`), never from worker order.
- Keep the macOS OpenMP rule: set `OMP_NUM_THREADS=1` and import `lightgbm`
  before other OpenMP libraries in tests and workers.

### Phase 3: Measure

| Benchmark | Dataset | Metric | Target on Community Cloud |
|---|---|---|---|
| Grid Search page, default settings | 500 rows × 5 inputs × 3 outputs | wall time, peak RSS | < 60 s, < 1.5 GB |
| Auto-detect page | same | wall time, peak RSS | < 90 s, < 1.5 GB |
| Multi-fidelity page, 2 levels | 400 low + 80 high rows | wall time | < 60 s |
| Cold start after sleep | none | time to first render | < 30 s |
| Serial vs parallel equality | all of the above | chosen model and metrics | identical for a fixed seed |

Add a `benchmarks/` script that records these numbers per commit (locally and
on a CI runner sized like the Cloud container: 2 vCPU, ~2.5 GB). The CI check
should fail on a > 25 % regression.

## What More Compute Would Unlock

### By hardware tier

| Tier | Typical resources | Practical parallelism | Grid Search (m = 3) wall time vs Cloud | What becomes reasonable |
|---|---|---|---|---|
| **Streamlit Community Cloud** | ≤ 2 cores, ≤ 2.7 GB, no GPU | threads, `n_jobs ≤ 2` | 1× (baseline) | ≤ ~2 000 rows, `skip_expensive`, small grids, single split validation |
| **Laptop / workstation** | 8–16 cores, 16–64 GB | `loky`, `n_jobs = cores` | ~4–8× faster | Exact GP up to a few thousand rows, full default grids, k-fold selection, all report figures |
| **Single cloud VM** | 32–96 cores, 128–384 GB | `loky`, `n_jobs = cores` | ~15–40× faster | Nested CV, repeated seeds for stable model ranking, larger grids (e.g. `RandomizedSearchCV` with hundreds of draws) |
| **GPU VM** | 1–8 GPUs + CPU cores | torch / GPyTorch / XGBoost on GPU, CPU pool for sklearn | large gains for neural and GP models only | Deep ensembles and BNNs with many members, SGPR with more inducing points, n ≈ 10⁴–10⁵ |
| **Cluster (Ray / Dask)** | many nodes | distributed tasks | scales with grid size until per-task overhead dominates | Hyper-parameter search at scale, many-dataset batch runs, active learning with parallel simulator calls |

Speed-ups are upper bounds for the embarrassingly parallel selection loops.
Single-model fits (one exact GP, one SARIMA) do not get faster from more
workers, only from faster cores or a GPU.

### By feature

| Feature | On Community Cloud | With more compute |
|---|---|---|
| Model selection | Single validation split, default grids, expensive models skipped | k-fold or nested CV, repeated over seeds, one-standard-error rule on stable scores |
| Grid size | Tens of configs per model family | `RandomizedSearchCV` / Bayesian search (Optuna, Ray Tune) over hundreds of configs |
| Gaussian processes | Exact GP only for small n; RFF / Nyström approximations otherwise | Exact GP to a few thousand rows; SGPR on GPU beyond that |
| Uncertainty | Split conformal | CV+ / jackknife+ conformal (k extra fits), larger deep ensembles |
| Multi-fidelity | 2–3 levels, about 30 candidates | Every model-type × base-surrogate × level combination, with per-candidate hyper-parameter search |
| Explanations | SHAP on a sample of rows | Full-data SHAP, PDP/ICE on every feature pair |
| Sampling detection | `n_sim` reduced for large n | Full `n_sim` at every n; tighter p-values |
| Time series | Narrow (p, q, P, Q) ranges | Wider SARIMA grids, LSTM / Chronos on GPU |
| Active learning | Small candidate pools, sequential simulator calls | Large candidate pools, batch acquisition (`active/batch.py`) with the batch evaluated in parallel on the simulator side |

## Distributed Options

When runs outgrow one container, the GUI should stay a thin front end and hand
the compute to workers:

```text
Streamlit page  --submit(job spec, data ref)-->  queue  -->  worker pool (VM / cluster / serverless)
      ^                                                            |
      +------- poll job status / fetch results <-- object store <--+
```

| Approach | Effort | Running cost | Best fit |
|---|---|---|---|
| Run the GUI on a larger VM (Docker image, same code) | Low | VM cost, always on | Internal use, predictable load |
| Job queue (RQ + Redis, or Celery) + worker VM | Medium | Redis + workers | Long reports and grid searches without blocking the UI |
| `joblib` with a Dask or Ray backend | Low code change, medium ops | Cluster cost | Scaling the existing `Parallel` loops unchanged |
| Ray Tune / Optuna for model selection | Medium | Cluster cost | Replacing exhaustive grids with adaptive search |
| Serverless fan-out (Modal, Cloud Run jobs, Lambda) | Medium | Pay per call | Occasional heavy jobs from a free front end |

Because Phase 2 routes every loop through `joblib.Parallel`, moving to a Dask
or Ray cluster is mostly a backend switch
(`with joblib.parallel_config(backend="ray")`) rather than a rewrite.

Hosting the GUI elsewhere (a larger VM, a container platform, Snowflake-hosted
Streamlit, Hugging Face Spaces) should be compared on CPU, memory, sleep
policy and cost when the need arises. Check each provider's current limits
before choosing, because they change often.

## Delivery Plan

| Phase | Work | Effort | Depends on |
|---|---|---|---|
| 0 | Delete the duplicated selection block; size guards; `skip_expensive` default on Cloud; lazy imports; bounded SHAP | Low | — |
| 1 | `st.cache_data` / `st.cache_resource` / `st.session_state` on all heavy pages | Low | 0 |
| 2a | `utils/parallel.py` (`get_n_jobs`, `parallel_region`); replace SARIMA `n_jobs=-1` | Low | — |
| 2b | `n_jobs` in `AutoDetectMultiOutputRegressor`, `select_per_target`, multi-fidelity candidates, CV+ | Medium | 2a |
| 2c | Parallel sampling-detection simulations; optional `n_jobs` on ensembles and committees | Medium | 2a |
| 3 | Benchmarks script, CI runner sized like Community Cloud, regression check | Medium | 0–2 |
| 4 | Optional job queue + worker deployment for heavy runs | High | 3 |

## Testing

- **Equality:** for each parallelized function, `n_jobs=1` and `n_jobs=2` give
  the same chosen models, scores and predictions for a fixed seed.
- **Thread caps:** a test asserts that BLAS/OpenMP pools are limited to 1
  inside `parallel_region` (via `threadpoolctl.threadpool_info()`).
- **Memory:** a smoke test runs the default Grid Search on the benchmark
  dataset under a 2.5 GB limit (e.g. `ulimit -v` or a container limit in CI).
- **No `n_jobs=-1` in library code:** a lint/grep check, so new code goes
  through `get_n_jobs()`.

## Open Questions

1. Should the hosted app expose an "advanced compute" switch that is disabled on
   Community Cloud and enabled when `MULTIOUTREG_N_JOBS` is set?
2. Which page is the priority for a background-job version: Grid Search, Auto-detect
   or Multi-fidelity?
3. Is there a budget for a small always-on VM, or should heavy runs stay local
   (CLI) and the hosted app remain a demo?
