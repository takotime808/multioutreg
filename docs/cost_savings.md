# Cost Savings and ROI

What does adding `multioutreg` to an existing simulation or experiment workflow save? This page
summarizes measured answers: the number of expensive-model runs each workflow needs for the
same result, priced in money, wall-clock time and core-hours. Every number comes from the
tutorial notebook {doc}`tutorials/cost_savings`, which you can re-run with your own cost
figures.

## Summary

Each row compares the run count of a common existing workflow against `multioutreg` at the
**same accuracy**, on analytic benchmarks where the true answer is known. Medians over seeds.
Dollar and time figures use the default cost profile below (a 2-hour, 32-core job, about
$8.20 per run, 8 jobs in parallel).

| Task | Existing workflow | Runs | With `multioutreg` | Runs | Fewer runs | Saved (default profile) |
|---|---|---:|---|---:|---:|---|
| Uncertainty propagation (8-input borehole: mean, std and 99th percentile) | 10,000-sample Monte Carlo on the simulator | 10,000 | GP surrogate on a Sobol design, Monte Carlo on the surrogate | 320 | **31x** | $79k, 620k core-hours, 100 days |
| Design optimization (Branin) | Random search | 829 | Bayesian optimization, `ActiveLearner(acquisition="ei")` | 24 | **35x** | $6.6k, 8 days |
| Design optimization (Hartmann-3) | Random search | 914 | Bayesian optimization | 28 | **33x** | $7.3k, 9 days |
| Surrogate training (Hartmann-3, NRMSE 0.3) | 5×5×5 full-factorial grid | 125 | Space-filling or active design | 50 | **2.5x** | $615, 18 hours |
| Multi-fidelity surrogate (Currin, NRMSE 0.1) | Active learning on high fidelity only | 36 | `MultiFidelityActiveLearner` with a 10x cheaper model | 10 (cost-equivalent) | **3.6x** | $213 |
| Multi-fidelity surrogate (Park, NRMSE 0.01) | Active learning on high fidelity only | 51 | `MultiFidelityActiveLearner` | 33 (cost-equivalent) | **1.5x** | $148 |
| Model selection compute (10 model families, 3 outputs) | Full cross-validated grid search | 127 s | `AutoDetectMultiOutputRegressor(pre_screen=True)` | 29 s | **77% less time** | same chosen models and test R² |

A full-factorial grid sweep is a stronger baseline than random search, but it still needs
196 runs on Branin (14×14) and 512 on Hartmann-3 (8×8×8) to get as close to the optimum, so
Bayesian optimization is 8x to 18x cheaper than a grid as well.

All of `multioutreg`'s own compute in these scenarios (surrogate fits, candidate scoring) took
5 to 41 seconds on a laptop per scenario. It is included in the savings above and is
negligible next to a single simulator run.

## Where the savings come from

```{mermaid}
flowchart LR
    subgraph before["Existing workflow"]
        A1[Fix a grid or random design] --> A2[Run every case] --> A3[Analyse / optimise on raw runs]
    end
    subgraph after["With multioutreg"]
        B1[Small space-filling design] --> B2[Fit surrogate]
        B2 --> B3{Good enough?<br/>CV error / EI}
        B3 -- no --> B4[ask: next batch<br/>or cheaper fidelity] --> B5[Run batch on your queue] --> B2
        B3 -- yes --> B6[Many-query analysis on the surrogate]
    end
```

1. **Many-query analyses run on the surrogate, not the simulator.** Monte Carlo uncertainty
   propagation, sensitivity studies and parameter sweeps need thousands of evaluations; a
   surrogate answers them in under a second after a few hundred training runs. For means and
   standard deviations, about 80 runs already beat a 10,000-run study; tail quantiles need
   more.
2. **Runs go where they are informative.** Bayesian optimization (`"ei"`, `"ucb"`, `"pi"`)
   concentrates runs near the optimum instead of sweeping the domain, the largest saving
   measured here.
3. **Space-filling instead of grids.** A full-factorial grid repeats every input level and its
   size jumps by `k^d`. Sobol or active designs reach the same surrogate accuracy in fewer
   runs, and the gap grows with the number of inputs.
4. **Cheap models carry part of the load.** When a coarse model correlates with the expensive
   one, the multi-fidelity learner spends most of the budget on it, while checking the
   correlation on high-fidelity data and keeping at least 25% of spending there.
5. **Stop when good enough.** The sequential loop can stop as soon as a cross-validated error
   estimate reaches the target, instead of finishing a budget fixed in advance.
6. **Less model-search compute.** Statistical pre-screening skips model families the data does
   not justify before the cross-validated search.

## Pricing it for your workflow

The run counts are properties of the problem. The money and time are properties of your
simulator. The notebook's `SimulationCost` turns runs into money, core-hours and wall-clock
time:

```python
COST = SimulationCost(
    hours_per_run=2.0,        # wall-clock hours for one run
    cores_per_run=32,         # cores one run occupies
    usd_per_core_hour=0.05,   # compute price
    usd_fixed_per_run=5.0,    # licenses, consumables, set-up time per run
    parallel_jobs=8,          # runs executing at the same time
)
```

The same measured run counts, re-priced for three kinds of simulator (totals over the six
scenarios in the summary table, not counting model-selection compute):

| Cost profile | Per run | Money saved | Core-hours saved | Wall-clock saved |
|---|---:|---:|---:|---:|
| Cheap: 10 min, 4 cores, no fixed cost, 32 in parallel | $0.03 | $383 | 7,660 | 2.5 days |
| Default: 2 h, 32 cores, $5 fixed, 8 in parallel | $8.20 | $94,218 | 735,360 | 120 days |
| Expensive: 24 h, 128 cores, $50 fixed, 4 in parallel | $204 | $2.3M | 35.3M | 2,873 days |

As a rough annual estimate:

```text
annual saving ≈ Σ over study types  (studies per year) × (runs_baseline − runs_multioutreg) × (cost per run)
```

For example, a team with the default cost profile that runs four uncertainty studies and ten
3-input optimizations a year would, at the ratios above, save about 4 × 9,680 + 10 × 886 ≈
47,600 runs: roughly $390k, 3 million core-hours, and about 16 months of wall-clock time on 8
parallel slots. Your own ratios
depend on input dimension, smoothness and accuracy target, so re-run the notebook's scenarios
on a problem that resembles yours before committing to a number.

## What changes in your pipeline

Only the design-of-experiments step. The simulator does not need to be Python: `ActiveLearner`
exposes the loop as `ask` / `tell`, so the existing job scripts, scheduler and post-processing
stay as they are.

```python
from multioutreg.active import ActiveLearner

learner = ActiveLearner(model_factory, bounds, batch_size=8)    # 8 = free queue slots
X0 = learner.initial_design(16)
learner.tell(X0, submit_and_wait(X0))                           # your existing job submission

while cv_error(learner) > target:
    X_next = learner.ask()                                      # next batch of inputs
    learner.tell(X_next, submit_and_wait(X_next))

surrogate = learner.fit()
```

- For optimization, pass `acquisition="ei"` (and `weights=` for a weighted sum of outputs).
- With a cheap and an expensive model, use `MultiFidelityActiveLearner` with per-level costs.
- For model selection, set `pre_screen=True` on `AutoDetectMultiOutputRegressor`.
- The fitted surrogate is an ordinary `multioutreg` estimator: score it with the
  performance metrics, add conformal intervals, export an HTML report.

See the {doc}`active_learning` plan and the {doc}`tutorials/active_learning_benchmarks`
tutorial for the loop's options.

## When it does not pay

- **Cheap simulators.** If a run takes seconds, run it directly; the cheap profile above saves
  little money, mostly calendar time.
- **Rough or high-dimensional responses.** A GP surrogate works well with the 8 inputs of the
  borehole model; with dozens of inputs, discontinuities or regime changes, it needs many more
  training runs. Check cross-validated error before trusting a surrogate for a decision.
- **A good space-filling design is already in place.** Max-variance active learning is roughly
  break-even with a Sobol design on smooth problems (ahead on one benchmark, behind on the
  other); the remaining gain is the stopping rule.
- **Weakly correlated cheap models.** Multi-fidelity savings depend on the correlation between
  levels. On Currin the multi-fidelity surrogate reaches a moderate target 3.6x cheaper but
  levels off, and for tighter targets high-fidelity data is needed.
- **Pre-screening on borderline data.** Skipping a heavy model family can cost a little
  accuracy on a nonlinear output; compare against the full search once before enabling it
  everywhere.
- **Final decisions.** Confirm a design chosen on a surrogate with a real run. Bayesian
  optimization already evaluates its best point; surrogate-only analyses should budget a few
  confirmation runs.

## Reproducing

The numbers on this page come from {doc}`tutorials/cost_savings` (about 15 minutes on a
laptop). Edit `SimulationCost` in its first section to price the ledger for your simulator;
the run counts do not depend on it.
