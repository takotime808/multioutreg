# Multi-Objective Regret in `multioutreg.metrics`

This module provides three complementary regret metrics for multi-objective model evaluation and iterative optimization:

- **Hypervolume regret**: `HV(true_front) − HV(pred_front)`, with `hypervolume(Y)` exact in any number of objectives  
- **Scalarized regret**: average (or distribution) of `min_x w·Y_true(x) − min_x w·Y_pred(x)` across weight vectors `w`  
- **ε-indicator regret**: additive dominance gap needed so the predicted front weakly dominates the true front

All metrics are formulated under a **minimization** convention (flip signs for maximization or pass `minimize=[...]`).

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

The default is computed from **both** `Y_true` and `Y_pred`, so it moves with the prediction:
fix `reference_point` when comparing several models, or their regrets are on different scales.
For objectives with `minimize=False`, give the reference point's coordinate negated (the
worst value in your own units, times −1).

**Exact in any dimension:** `hypervolume(Y, reference_point)` and `hypervolume_regret` are
exact for any number of objectives. 1-D and 2-D use closed forms; for 3+ objectives the front
is sliced along the last objective and the slab volumes are summed recursively down to the
2-D sweep. With `pymoo` installed (`pip install multioutreg[moo]`) its faster exact algorithm
is used for 3+ objectives; the values agree. The pure-NumPy path takes well under two seconds
for fronts of hundreds of points with 3–4 objectives; for 5+ objectives with large fronts,
install `pymoo`.

**Front fidelity versus decision regret:** `hypervolume_regret(Y_true, Y_pred)` compares the
*predicted* front with the true one and is clipped at zero, so an optimistic model can score
0. To score the designs a model would pick, pass the true values at its picks:
`hypervolume_regret(Y_true, Y_true[pareto_mask(Y_pred)])`. Swapping the arguments,
`hypervolume_regret(Y_pred, Y_true)`, is positive when the prediction is optimistic.
`examples/hypervolume_regret.ipynb` walks through all of this.

### 2) Scalarized regret — preference-aware loss
Many real-world decisions are made with some (possibly unknown) preference over objectives. Weighted sums `w·Y` are a simple and widely used scalarization. By comparing `min_x w·Y_true` vs. `min_x w·Y_pred` over **multiple** weights, we summarize how much utility a decision-maker might lose when using the model’s predicted front instead of the true front.

**Default choice:**  
`scalarized_regret` requires `weights`; each row is clipped to non-negative values and normalized to sum to 1. The callers that choose weights for you sample them uniformly over the simplex (Dirichlet(1)), which gives a spread of preferences without favoring any one objective:
- `RegretTracker` samples **16** weight vectors (`n_weight_samples`) on its first `step`.
- The GUI's **Auto** mode samples **8** weight vectors; its **Uniform** mode uses the single vector `1/m`.

When the number of objectives is 1, a scalar weight of `1.0` is used.

### 3) ε-indicator regret — dominance-based tightness
The additive ε-indicator reports how far (additively, per objective) the predicted front must be shifted so it **weakly dominates** the true front. It is a crisp, dominance-centric notion and complements HV and scalarized regret: ε=0 means the predicted front is already at least as good everywhere.

**Default choice:**  
ε has no additional parameters. We compute the indicator between the **non-dominated** subsets of the two fronts.

**Definition.** With `P` the predicted front and `T` the true front (both non-dominated, after
flipping any `minimize=False` objectives to minimization):

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

**Worked examples** (two objectives, both minimized, `T = [[0, 1], [1, 0]]`):

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
- **Optimism scores 0.** As with HV regret, a model that over-promises gets ε = 0. To score the
  designs a model would actually pick, pass the true values at its picks:
  `epsilon_regret(Y_true, Y_true[pareto_mask(Y_pred)])`.
- **Empty fronts** return `inf` (e.g. an empty `Y_pred`).
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
    RegretTracker,
)

# Exact hypervolume of a front, any number of objectives
hv = hypervolume(Y_true, reference_point=ref, minimize=True)

# Hypervolume regret (optionally pass a reference point)
r_hv = hypervolume_regret(Y_true, Y_pred, reference_point=None, minimize=True)

# Decision regret: true values at the designs the model calls Pareto-optimal
r_dec = hypervolume_regret(Y_true, Y_true[pareto_mask(Y_pred)], reference_point=ref)

# Scalarized regret across weights (1D/2D/scalar accepted; rows auto-normalized)
r_sc = scalarized_regret(Y_true, Y_pred, weights=[[0.5,0.5],[0.2,0.8]], reduce="mean")

# Additive epsilon indicator regret
r_eps = epsilon_regret(Y_true, Y_pred, minimize=True)

# Iterative tracking
tracker = RegretTracker(minimize=True, n_weight_samples=16)
stats = tracker.step(Y_true_batch, Y_pred_batch)
```

**Shapes:** `Y_*` are `(n, m)` or `(n,)` (treated as `(n,1)`).  
Weights can be `(k, m)`, `(m,)`, `(k,)` (if `m==1`), or scalar (if `m==1`).

---

## Defaults summary

- **HV reference**: data-dependent worst + **10%** span (per objective) if not provided.  
- **Scalarized weights**: required by `scalarized_regret`; `RegretTracker` samples **16** Dirichlet(1) vectors and the GUI's Auto mode samples **8**; **1.0** when `m==1`.  
- **ε-indicator**: parameter-free, but in objective units; standardize objectives to compare across them.  
- **Minimization**: by default, all objectives are minimized (pass `minimize=[...]` to mix min/max).

These defaults are chosen to be **safe, scale-aware, and preference-agnostic**, giving meaningful numbers without extra configuration, while still letting you override them when domain knowledge is available.
