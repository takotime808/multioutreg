# Multi-Objective Regret in `multioutreg.metrics`

This module provides three complementary regret metrics for multi-objective model evaluation and iterative optimization:

- **Hypervolume regret**: `HV(true_front) − HV(Y_true[picks])`, with `hypervolume(Y)` exact in any number of objectives  
- **Scalarized regret**: average (or distribution) of `w·Y_true(argmin_x w·Y_pred(x)) − min_x w·Y_true(x)` across weight vectors `w`  
- **ε-indicator regret**: additive improvement the picks would need to weakly dominate the true front

All metrics are formulated under a **minimization** convention (flip signs for maximization or pass `minimize=[...]`).

### Decision regret (the default) versus front fidelity

`hypervolume_regret`, `scalarized_regret` and `epsilon_regret` are **decision regret**. The
model's *picks* are the candidates on the Pareto front of `Y_pred`, and the regret evaluates the
**true** objectives at those picks: what acting on the model costs. `Y_true` and `Y_pred` must
therefore hold the same candidates, row for row (same shape); a mismatch raises `ValueError`.
Decision regret is always ≥ 0 and is 0 when the picks contain the true front. A model cannot
lower it by over-promising: shifting every prediction by a constant leaves the picks unchanged.

The previous behaviour, comparing the *predicted* front with the true front, is kept as
`front_hypervolume_regret`, `front_scalarized_regret` and `front_epsilon_regret`. These measure
how well the predictions reproduce the front, accept arrays with different numbers of rows, and
are clipped at zero, so an optimistic model can score 0 (in the multi-output tutorial a ZDT1
surrogate whose `f2` is shifted down by 0.5 scores 0 front regret).
`front_hypervolume_regret(Y_pred, Y_true)`, with the arguments swapped, is positive when the
prediction is optimistic. The two are related by
`hypervolume_regret(Y_true, Y_pred) == front_hypervolume_regret(Y_true, Y_true[pareto_mask(Y_pred)])`
for a fixed reference point.

**Ties.** Identical predicted rows are mutually non-dominated, so a model that predicts the same
vector everywhere (e.g. a mean predictor) picks *every* candidate and scores 0 hypervolume and ε
decision regret. Check the number of picks (`pareto_mask(Y_pred).sum()`) alongside the regret
when predictions can tie. `scalarized_regret` breaks ties by taking the first candidate.

---

## Why these three?

### 1) Hypervolume regret — global coverage of trade-offs
Hypervolume (HV) is the standard Pareto quality indicator. It measures the dominated volume between a Pareto front and a **reference point** that is worse than all points. Reporting HV **regret** (difference between the true and predicted fronts) directly quantifies how much Pareto area your model/front fails to capture.

**Default choice:**  
If no reference is provided, we derive a safe, data-dependent reference as:
```
worst + 10% * span   (per objective, with a 1.0 fallback if span = 0)
```
This keeps the reference large enough to avoid clipping HV and stable across datasets with different scales, without requiring manual tuning.

For the decision-regret functions (and `RegretTracker`) the default is computed from `Y_true`
only, since only true values are ever measured; it is the same for every model scored on the
same test set. `front_hypervolume_regret` computes it from **both** `Y_true` and `Y_pred`, so
there it moves with the prediction: fix `reference_point` when comparing several models.
For objectives with `minimize=False`, give the reference point's coordinate negated (the
worst value in your own units, times −1).

**Exact in any dimension:** `hypervolume(Y, reference_point)` and both hypervolume regrets are
exact for any number of objectives. 1-D and 2-D use closed forms; for 3+ objectives the front
is sliced along the last objective and the slab volumes are summed recursively down to the
2-D sweep. With `pymoo` installed (`pip install multioutreg[moo]`) its faster exact algorithm
is used for 3+ objectives; the values agree. The pure-NumPy path takes well under two seconds
for fronts of hundreds of points with 3–4 objectives; for 5+ objectives with large fronts,
install `pymoo`.

`examples/hypervolume_regret.ipynb` walks through decision regret and front fidelity.

### 2) Scalarized regret — preference-aware loss
Many real-world decisions are made with some (possibly unknown) preference over objectives. Weighted sums `w·Y` are a simple and widely used scalarization. For each weight vector the model picks the candidate with the best predicted `w·Y_pred`; the regret is how much worse that candidate's true `w·Y_true` is than the true best. Averaging over **multiple** weights summarizes how much utility a decision-maker might lose by acting on the model. `front_scalarized_regret` instead compares `min_x w·Y_true` with `min_x w·Y_pred`; it is negative for pessimistic predictions and positive for optimistic ones.

**Default choice:**  
`scalarized_regret` requires `weights`; each row is clipped to non-negative values and normalized to sum to 1. The callers that choose weights for you sample them uniformly over the simplex (Dirichlet(1)), which gives a spread of preferences without favoring any one objective:
- `RegretTracker` samples **16** weight vectors (`n_weight_samples`) on its first `step`.
- The GUI's **Auto** mode samples **8** weight vectors; its **Uniform** mode uses the single vector `1/m`.

When the number of objectives is 1, a scalar weight of `1.0` is used.

### 3) ε-indicator regret — dominance-based tightness
The additive ε-indicator reports how far (additively, per objective) a set must be shifted so it **weakly dominates** the true front. It is a crisp, dominance-centric notion and complements HV and scalarized regret. `epsilon_regret` applies it to the true values at the model's picks (ε=0 means the picks contain the true front); `front_epsilon_regret` applies it to the predicted front itself.

**Default choice:**  
ε has no additional parameters.

**Definition.** With `T` the true front and `P` either the true values at the picks
(`epsilon_regret`) or the predicted front (`front_epsilon_regret`), after flipping any
`minimize=False` objectives to minimization:

```
ε-regret = max(0,  max_{t ∈ T}  min_{p ∈ P}  max_i (p_i − t_i))
```

Read it inside out:

1. `max_i (p_i − t_i)` is how far predicted point `p` falls short of true point `t` on its
   *worst* objective.
2. `min_{p ∈ P}` picks the predicted point that covers `t` best.
3. `max_{t ∈ T}` takes the true point that is covered worst.

So ε is the smallest amount you could subtract from **every** objective of every predicted point
so that each true Pareto point is weakly dominated. HV regret is an average over the front;
ε is worst-case. One badly missed region of the front sets ε even when the rest is perfect.

**Worked examples** for `front_epsilon_regret` (two objectives, both minimized, `T = [[0, 1], [1, 0]]`):

| `Y_pred` | ε-regret | Why |
|---|---|---|
| `T` | `0.0` | Identical fronts |
| `T + [0.2, 0.0]` | `0.2` | Every point is 0.2 worse on objective 1 |
| `[[0, 1]]` | `1.0` | The corner `[1, 0]` is missed; the nearest prediction is 1.0 worse on objective 2 |
| `T − 0.5` | `0.0` | An optimistic front dominates `T`, and the result is clipped at 0 |

**Things to watch:**

- **Units matter.** The indicator is additive, so it is in the units of the objectives and is
  dominated by the objective with the largest scale. In the example above, if objective 2 were
  multiplied by 1000, the same 0.2 shift on objective 2 would give ε = 200. Standardize the
  objectives (or divide by known ranges) before comparing ε across objectives or datasets.
- **Optimism scores 0 in the front version.** With `front_epsilon_regret`, a model that
  over-promises gets ε = 0. `epsilon_regret` scores the picks' true values, so it does not.
- **Empty fronts** return `inf` in `front_epsilon_regret` (e.g. an empty `Y_pred`); the decision
  functions raise `ValueError` for empty input.
- **Maximization** is handled by `minimize=[...]`: those columns are negated internally, so ε
  stays non-negative and in the original units.
- **Cost:** after Pareto filtering it is `O(|T| · |P| · m)`, which is cheap next to `pareto_mask`
  itself.

**Where it shows up:** `RegretTracker.step` records it under `"eps"` in `history`; the AutoDetect
GUI page reports it as **ε-regret** on the test split and lets you set a **Max ε-regret (≤)** bound.

---

## Practical guidance

- **When to tune the HV reference:**  
  If your objectives have known “worst acceptable” bounds or a natural operating region, set a custom reference point to improve interpretability and stability of HV.

- **When to customize weights:**  
  If stakeholders can articulate preferences, supply those weight vectors (each row sums to 1). Otherwise, use the defaults to summarize global performance across preferences.

- **How to interpret bounds (thresholds):**  
  If you set **max acceptable** values for HV, scalarized, or ε-regret, the UI will flag results:
  - ✅ when metric ≤ bound
  - ❌ otherwise

---

## Computational notes

- **`pareto_mask` cost:** dominance is checked between every pair of rows using `(n, n, m)` boolean
  arrays, so time and memory grow with `n² · m`. A few thousand rows is fine; for two objectives,
  `n = 20,000` needs on the order of 2 GB. Filter or subsample very large candidate sets first.

---

## API recap

```python
from multioutreg.metrics import (
    pareto_mask,
    hypervolume,
    hypervolume_regret,
    scalarized_regret,
    epsilon_regret,
    front_hypervolume_regret,
    RegretTracker,
)

# Exact hypervolume of a front, any number of objectives
hv = hypervolume(Y_true, reference_point=ref, minimize=True)

# Decision hypervolume regret: true values at the designs the model calls Pareto-optimal.
# Y_true and Y_pred are the same candidates, row for row.
r_hv = hypervolume_regret(Y_true, Y_pred, reference_point=None, minimize=True)

# Front fidelity: predicted front versus true front (clipped at 0)
r_front = front_hypervolume_regret(Y_true, Y_pred, reference_point=ref)

# Scalarized decision regret across weights (1D/2D/scalar accepted; rows auto-normalized)
r_sc = scalarized_regret(Y_true, Y_pred, weights=[[0.5,0.5],[0.2,0.8]], reduce="mean")

# Additive epsilon indicator of the picks
r_eps = epsilon_regret(Y_true, Y_pred, minimize=True)

# Iterative tracking (records the three decision regrets)
tracker = RegretTracker(minimize=True, n_weight_samples=16)
stats = tracker.step(Y_true_batch, Y_pred_batch)
```

**Shapes:** `Y_*` are `(n, m)` or `(n,)` (treated as `(n,1)`); the decision functions need `Y_true` and `Y_pred` of the same shape.  
Weights can be `(k, m)`, `(m,)`, `(k,)` (if `m==1`), or scalar (if `m==1`).

---

## Joint predictive scores

The regrets above score *where a model's predictions land*. `energy_score` and `variogram_score`
score its *joint predictive distribution*: whether the uncertainty it reports for several
outputs together, including how they co-vary, matches what happens. Per-output metrics
cannot see this. Both are proper scores (lower is better) computed from samples of shape
`(n_rows, n_samples, n_outputs)`.

```python
from multioutreg.metrics import energy_score, variogram_score, gaussian_samples

mean, cov = model.predict(X_test, return_cov=True)   # or return_std=True for independent outputs
S = gaussian_samples(mean, cov, n_samples=500, seed=0)
es = energy_score(Y_test, S)          # multivariate CRPS (Gneiting & Raftery 2007)
vs = variogram_score(Y_test, S, p=0.5)  # sensitive to wrong correlation (Scheuerer & Hamill 2015)
```

- **Energy score** reduces to the CRPS for one output. It is dominated by errors in the means,
  so it separates correct from wrong dependence only weakly.
- **Variogram score** compares pairwise output differences, so it reacts strongly to a wrong
  correlation and ignores a shift shared by all outputs. Report it with a marginal score.
- Both use distances across outputs, so standardize outputs on very different scales first.
- `energy_score` builds an `(n, S, S, m)` array; keep `n_samples` to a few hundred for large `n`.

Surrogates that return a covariance: `GaussianCopulaSurrogate`, `PCAOutputSurrogate` and
`MultiTaskGPSurrogate`. The `dtlz2` benchmark (`multioutreg.active.benchmarks`, front from
`dtlz_front`) has strongly dependent outputs for testing them.

---

## Defaults summary

- **HV reference**: data-dependent worst + **10%** span (per objective) if not provided, from `Y_true` for decision regret and from both arrays for `front_hypervolume_regret`.  
- **Scalarized weights**: required by `scalarized_regret`; `RegretTracker` samples **16** Dirichlet(1) vectors and the GUI's Auto mode samples **8**; **1.0** when `m==1`.  
- **ε-indicator**: parameter-free, but in objective units; standardize objectives to compare across them.  
- **Minimization**: by default, all objectives are minimized (pass `minimize=[...]` to mix min/max).

These defaults are chosen to be **safe, scale-aware, and preference-agnostic**, giving meaningful numbers without extra configuration, while still letting you override them when domain knowledge is available.
