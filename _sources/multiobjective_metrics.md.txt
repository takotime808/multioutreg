# Multi-Objective Metrics

When a surrogate is used to explore trade-offs between outputs (lower drag *and*
higher lift, cheaper *and* stronger), per-output R² is not enough. What matters is
which designs the model would pick as Pareto-optimal, and how good those designs
really are. {mod}`multioutreg.metrics` scores that with three **decision regrets**.
The model's *picks* are the rows on the Pareto front of its predictions; each regret
evaluates the **true** objectives at those picks against the true Pareto front. All
three are 0 when the picks contain the true front, are never negative, are in the
units of the objectives, and assume every objective is **minimized** (pass
`minimize=[...]` to mix minimized and maximized columns).

| Metric | Question it answers | Sensitive to |
|---|---|---|
| Hypervolume regret | How much of the trade-off space do the picks lose? | The whole front, on average |
| Scalarized regret | How much worse is the design the model prefers, for a given preference? | Typical preferences between objectives |
| ε-regret | What is the largest shortfall anywhere on the front? | The single worst-covered region |

Each has a `front_*` counterpart that compares the *predicted* front with the true one
instead: how well the predictions reproduce the front, rather than what acting on them
costs.

## A short example

Two objectives on a ZDT1-like front, and predictions with a little noise:

```python
import numpy as np
from multioutreg.metrics import (
    epsilon_regret, front_epsilon_regret, front_hypervolume_regret,
    front_scalarized_regret, hypervolume, hypervolume_regret, scalarized_regret,
)

rng = np.random.default_rng(0)
f1 = rng.uniform(size=200)
Y_true = np.c_[f1, 1 - np.sqrt(f1) + 0.1 * rng.uniform(size=200)]
Y_pred = Y_true + rng.normal(0, 0.05, size=Y_true.shape)
ref = np.array([1.1, 1.2])                       # worse than every point
W = rng.dirichlet(np.ones(2), size=8)            # 8 preference weightings

hypervolume(Y_true, reference_point=ref)                  # 0.9565

# Decision regret: what acting on the model's picks costs
hypervolume_regret(Y_true, Y_pred, reference_point=ref)   # 0.0421
scalarized_regret(Y_true, Y_pred, weights=W)              # 0.0336
epsilon_regret(Y_true, Y_pred)                            # 0.0634

# Front fidelity: how closely the predicted front matches the true one
front_hypervolume_regret(Y_true, Y_pred, reference_point=ref)  # 0.0
front_scalarized_regret(Y_true, Y_pred, weights=W)             # 0.0546
front_epsilon_regret(Y_true, Y_pred)                           # 0.0044
```

The two views disagree, and the decision regret is the one to trust when the model
will choose designs. Noise pushes some predicted points past the true front, so the
predicted front covers at least as much volume as the true one and the clipped
`front_hypervolume_regret` is **0**. Acting on those predictions still loses
hypervolume, because some of the rows the model calls Pareto-optimal are not. A model
cannot lower decision regret by over-promising: shifting every prediction by a
constant leaves its picks, and the regret, unchanged.

## Definitions

Write {math}`y_k` and {math}`\hat{y}_k` for the true and predicted objectives of row
{math}`k` (same rows in both), {math}`T` for the true Pareto front, and
{math}`S = \{\, y_k : \hat{y}_k \text{ is on the Pareto front of the predictions} \,\}`
for the true values at the model's picks. {math}`m` is the number of objectives and
{math}`r` the hypervolume reference point.

**Hypervolume** is the volume dominated by a set of points and bounded by {math}`r`:

```{math}
\mathrm{HV}(F; r) = \lambda\Big(\bigcup_{f \in F} [f_1, r_1] \times \dots \times [f_m, r_m]\Big)
```

where {math}`\lambda` is the Lebesgue measure. It is computed exactly for any
{math}`m`. When `reference_point` is not given it is set, per objective, to the worst
true value plus 10% of the range (`front_hypervolume_regret` uses both arrays); fix it
yourself when comparing several models.

```{math}
R_{\mathrm{HV}} = \mathrm{HV}(T; r) - \mathrm{HV}(S; r)
```

**Scalarized regret** asks, for each weight vector {math}`w \in W` (rows clipped to
non-negative and normalized to sum to 1), how much worse the row the model prefers is
than the true best:

```{math}
r(w) = w^\top y_{\hat{k}(w)} - \min_k w^\top y_k,
\qquad \hat{k}(w) = \arg\min_k w^\top \hat{y}_k,
\qquad R_{\mathrm{scal}} = \frac{1}{|W|} \sum_{w \in W} r(w)
```

`reduce` picks the summary (`"mean"`, `"median"`, `"max"`, `"dict"` for several,
`"none"` for every {math}`r(w)`).

**Additive ε-indicator regret** is the smallest amount every objective of the picks
would have to improve for them to weakly dominate the true front:

```{math}
R_{\varepsilon} = \max_{t \in T}\ \min_{s \in S}\ \max_{i = 1..m}\ (s_i - t_i)
```

{math}`R_{\mathrm{HV}}` averages over the whole front; {math}`R_{\varepsilon}` is a
worst case, so one badly missed corner sets it even when the rest is perfect.

**Front versions.** `front_hypervolume_regret` and `front_epsilon_regret` use the
predicted front {math}`P` in place of {math}`S` and clip at 0;
`front_scalarized_regret` uses {math}`r(w) = \min_k w^\top y_k - \min_k w^\top \hat{y}_k`,
which is negative for pessimistic predictions and positive for optimistic ones. They
accept arrays with different numbers of rows.

**Ties.** Identical predicted rows are mutually non-dominated, so a model that predicts
the same vector everywhere picks *every* row and scores 0 hypervolume and ε regret.
Check `pareto_mask(Y_pred).sum()` alongside the regret when predictions can tie.

## In the app and the report

The Auto-Detect page computes the three decision regrets on the test split, lets you set
a maximum for each, and the report shows them with **PASS** / **FAIL** in section 2 ("Multi-Objective
Regret"). A failed bound also appears as a finding and a next step on the report's front
page; see {doc}`reading_the_report`.

## Practical notes and defaults

```{include} ../multioutreg/metrics/README.md
:start-after: "## Practical guidance"
```
