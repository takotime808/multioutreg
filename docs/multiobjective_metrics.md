# Multi-Objective Metrics

When a surrogate is used to explore trade-offs between outputs (lower drag *and*
higher lift, cheaper *and* stronger), per-output R² is not enough: what matters is
whether the **Pareto front** the model predicts matches the true one.
{mod}`multioutreg.metrics` scores that with three regrets. All three are 0 for a
perfect front, are in the units of the objectives, and assume every objective is
**minimized** (pass `minimize=[...]` to mix minimized and maximized columns).

| Metric | Question it answers | Sensitive to |
|---|---|---|
| Hypervolume regret | How much of the trade-off space does the predicted front lose? | The whole front, on average |
| Scalarized regret | How much worse is the best design for a given preference? | Typical preferences between objectives |
| ε-regret | What is the largest shortfall anywhere on the front? | The single worst-covered region |

## A short example

Two objectives on a ZDT1-like front, and predictions with a little noise:

```python
import numpy as np
from multioutreg.metrics import (
    epsilon_regret, hypervolume, hypervolume_regret, pareto_mask, scalarized_regret,
)

rng = np.random.default_rng(0)
f1 = rng.uniform(size=200)
Y_true = np.c_[f1, 1 - np.sqrt(f1) + 0.1 * rng.uniform(size=200)]
Y_pred = Y_true + rng.normal(0, 0.05, size=Y_true.shape)
ref = np.array([1.1, 1.2])                     # worse than every point

hypervolume(Y_true, reference_point=ref)                        # 0.9565
hypervolume_regret(Y_true, Y_pred, reference_point=ref)         # 0.0
scalarized_regret(Y_true, Y_pred,
                  weights=rng.dirichlet(np.ones(2), size=8))    # 0.0546
epsilon_regret(Y_true, Y_pred)                                  # 0.0044

# Score the designs the model would pick, not the front it promises
picked = Y_true[pareto_mask(Y_pred)]
hypervolume_regret(Y_true, picked, reference_point=ref)         # 0.0421
epsilon_regret(Y_true, picked)                                  # 0.0634
```

The predicted front scores a hypervolume regret of **0**: noise pushes some predicted
points past the true front, the over-optimistic front covers at least as much volume,
and the regret is clipped at zero. Scoring the *true* values at the designs the model
calls Pareto-optimal (the last two lines) shows what using the model would cost.
Use that "decision regret" when the model will choose designs.

## Definitions

Write {math}`T` for the true Pareto front and {math}`P` for the predicted one (the
non-dominated rows of `Y_true` and `Y_pred`), {math}`m` for the number of objectives
and {math}`r` for the hypervolume reference point.

**Hypervolume** is the volume dominated by a front and bounded by {math}`r`:

```{math}
\mathrm{HV}(F; r) = \lambda\Big(\bigcup_{f \in F} [f_1, r_1] \times \dots \times [f_m, r_m]\Big)
```

where {math}`\lambda` is the Lebesgue measure. It is computed exactly for any
{math}`m`. When `reference_point` is not given it is set, per objective, to the worst
value in `Y_true` and `Y_pred` plus 10% of the range; fix it yourself when comparing
several models, or their regrets are on different scales.

```{math}
R_{\mathrm{HV}} = \max\big(0,\ \mathrm{HV}(T; r) - \mathrm{HV}(P; r)\big)
```

**Scalarized regret** compares the best weighted sum the true and predicted values
reach under each weight vector {math}`w \in W` (rows clipped to non-negative and
normalized to sum to 1), over the {math}`n` evaluated rows:

```{math}
r(w) = \min_{k} w^\top y_k - \min_{k} w^\top \hat{y}_k,
\qquad
R_{\mathrm{scal}} = \frac{1}{|W|} \sum_{w \in W} r(w)
```

It is not clipped: positive when the predictions promise a better optimum than the
truth delivers, negative when they are pessimistic. `reduce` picks the summary
(`"mean"`, `"median"`, `"max"`, `"dict"` for several, `"none"` for every {math}`r(w)`).

**Additive ε-indicator regret** is the smallest amount to subtract from every
objective of every predicted point so that each true point is weakly dominated:

```{math}
R_{\varepsilon} = \max\Big(0,\ \max_{t \in T}\ \min_{p \in P}\ \max_{i = 1..m}\ (p_i - t_i)\Big)
```

{math}`R_{\mathrm{HV}}` averages over the whole front; {math}`R_{\varepsilon}` is a
worst case, so one badly missed corner sets it even when the rest is perfect.

## In the app and the report

The Auto-Detect page computes all three on the test split, lets you set a maximum for
each, and the report shows them with **PASS** / **FAIL** in section 2 ("Multi-Objective
Regret"). A failed bound also appears as a finding and a next step on the report's front
page; see {doc}`reading_the_report`.

## Practical notes and defaults

```{include} ../multioutreg/metrics/README.md
:start-after: "## Practical guidance"
```
