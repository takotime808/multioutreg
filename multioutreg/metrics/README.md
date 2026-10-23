# Multi-Objective Regret in `multioutreg.metrics`

This module provides three complementary regret metrics for multi-objective model evaluation and iterative optimization:

- **Hypervolume regret**: `HV(true_front) − HV(pred_front)`  
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

### 2) Scalarized regret — preference-aware loss
Many real-world decisions are made with some (possibly unknown) preference over objectives. Weighted sums `w·Y` are a simple and widely used scalarization. By comparing `min_x w·Y_true` vs. `min_x w·Y_pred` over **multiple** weights, we summarize how much utility a decision-maker might lose when using the model’s predicted front instead of the true front.

**Default choice:**  
If weights aren’t supplied, we sample **8 Dirichlet** weight vectors uniformly over the simplex. This provides a small but diverse spread of preferences without biasing toward any one objective. When the number of objectives is 1, a scalar weight of `1.0` is used.

### 3) ε-indicator regret — dominance-based tightness
The additive ε-indicator reports how far (additively, per objective) the predicted front must be shifted so it **weakly dominates** the true front. It is a crisp, dominance-centric notion and complements HV and scalarized regret: ε=0 means the predicted front is already at least as good everywhere.

**Default choice:**  
ε has no additional parameters. We compute the indicator between the **non-dominated** subsets of the two fronts.

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

## API recap

```python
from multioutreg.metrics import (
    pareto_mask,
    hypervolume_regret,
    scalarized_regret,
    epsilon_regret,
    RegretTracker,
)

# Hypervolume regret (optionally pass a reference point)
r_hv = hypervolume_regret(Y_true, Y_pred, reference_point=None, minimize=True)

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
- **Scalarized weights**: **Dirichlet(1)** sample of **8** weight vectors when `m>1`; **1.0** when `m==1`.  
- **ε-indicator**: parameter-free.  
- **Minimization**: by default, all objectives are minimized (pass `minimize=[...]` to mix min/max).

These defaults are chosen to be **safe, scale-aware, and preference-agnostic**, giving meaningful numbers without extra configuration, while still letting you override them when domain knowledge is available.
