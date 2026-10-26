# Surrogate Hyperparameters

This page covers the hyperparameters of the kernel-based (Gaussian-process family) surrogates:
what each one does, the defaults, which ones are learned during `fit`, and which ones
`AutoDetectMultiOutputRegressor` and the Grid Search GUI search over. For the full list of
surrogates see [Integrated Surrogates](integrated_surrogates.md).

---

## Kernel hyperparameters in one paragraph

A GP models each output as a smooth function whose covariance between two inputs `x`, `x'` is
set by a kernel. For the squared-exponential (RBF) kernel used by default:

```
k(x, x') = σ² · exp(−‖x − x'‖² / (2 ℓ²))  +  σ_n² · [x = x']
```

| Symbol | Name | sklearn object | What it controls |
|---|---|---|---|
| `ℓ` | **length scale** | `RBF(length_scale=…)` / `Matern(length_scale=…)` | How far apart two inputs can be and still be strongly correlated. Small `ℓ` gives a wiggly function that only trusts nearby data; large `ℓ` gives a smooth, nearly linear one. Uncertainty grows roughly `ℓ` away from the training data. |
| `σ²` | amplitude / signal variance | `ConstantKernel` | Vertical scale of the function, i.e. how far from the mean predictions can swing. |
| `σ_n²` | noise level | `WhiteKernel(noise_level=…)` | Observation noise. With no noise term the GP has to pass through every training point, which forces `ℓ` to collapse on noisy data. |
| `ν` | Matérn smoothness | `Matern(nu=…)` | Differentiability. `ν = 2.5` (twice differentiable) is often a better prior than RBF (infinitely differentiable) for physical simulations. |

**ARD** (automatic relevance determination) gives each input feature its own length scale
`ℓ_j`. A feature whose `ℓ_j` grows very large barely affects the output, so the fitted length
scales double as a feature-relevance ranking.

---

## Defaults by surrogate

| Surrogate | Kernel | Hyperparameters | Learned in `fit`? | Input units |
|---|---|---|---|---|
| `GaussianProcessSurrogate` | `default_gp_kernel()` = `ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-2)`, `normalize_y=True` | `ℓ`, `σ²`, `σ_n²` | Yes, by maximum likelihood (L-BFGS-B) | Standardized (`scale_inputs=True`) |
| `ARDGPSurrogate` | `ConstantKernel(1.0) * RBF(length_scale=ones(p)) + WhiteKernel(1e-2)`, `normalize_y=True` | One `ℓ_j` per feature, `σ²`, `σ_n²`; `alpha=1e-6`, `n_restarts_optimizer=0` | Yes, by maximum likelihood | Standardized |
| `RFFGPSurrogate` | Random Fourier features + `BayesianRidge` | `length_scale=1.0`, `kernel="rbf"` (`"matern32"`, `"matern52"`), `n_components=500` | **No.** `length_scale` is fixed; only the linear weights are fit | **Raw** |
| `NystroemGPSurrogate` | `sklearn.kernel_approximation.Nystroem` + `BayesianRidge` | `gamma=None`, `kernel="rbf"`, `n_components=100` | **No.** `gamma` is fixed | **Raw** |
| `SparseGPSurrogate` | GPyTorch SGPR, `ScaleKernel(RBFKernel())` with inducing points | `n_inducing=50`, `max_iter=100`, `learning_rate=0.1` | Yes: `ℓ`, amplitude and noise by Adam on the marginal likelihood | Standardized internally |
| `GPXSurrogate` (SMT/egobox) | `corr="squar_exp"`, `poly="constant"` | `corr`, `poly`, `n_start=10` | Yes: SMT's `theta` (one inverse length scale per feature) by multistart MLE | As passed |
| `KPLSSurrogate` (SMT) | Kriging with partial-least-squares kernel | `n_comp=2`, `corr="squar_exp"`, `poly="constant"` | Yes: one `theta` per PLS component | As passed |

Notes:

- **Learned length scales start from `1.0`** and are bounded to sklearn's default range
  `(1e-5, 1e5)`. Because inputs are standardized first, `ℓ = 1` means "about one standard
  deviation of that feature", a sensible starting point whatever the raw units.
- **Fixed length scales are in raw input units.** For `RFFGPSurrogate`, `ℓ = 1.0` is far too
  small if a feature spans `0–1000` and far too large if it spans `0–0.01`. Standardize `X`
  yourself (e.g. `make_pipeline(StandardScaler(), …)`) or pick `length_scale` to match the
  feature spacing.
- **`gamma` is an inverse length scale.** sklearn's RBF is `exp(−γ‖x − x'‖²)`, so
  `γ = 1 / (2ℓ²)` and `ℓ = 1 / √(2γ)`. `gamma=None` means `γ = 1 / n_features`.
- **`alpha`** on the sklearn GPs is extra jitter on the kernel diagonal for numerical stability,
  added on top of the learned `WhiteKernel` noise. Raise it (e.g. `1e-2`) if fitting warns about
  ill-conditioned matrices.
- **`normalize_y=True`** standardizes each output, so the amplitude `σ²` starts near the right
  scale whatever the units of `Y`.

---

## Inspecting fitted length scales

`GaussianProcessSurrogate` wraps one fitted pipeline per output in `model.estimators_`, and the
last pipeline step holds the optimized kernel:

```python
from multioutreg.surrogates.gp_sklearn import GaussianProcessSurrogate

gp = GaussianProcessSurrogate().fit(X, Y)
for j, est in enumerate(gp.model.estimators_):
    print(j, est[-1].kernel_)
# 0 2.24**2 * RBF(length_scale=1.75) + WhiteKernel(noise_level=1e-05)
```

For ARD, each output's fitted GP is `estimators_[j]._gpr`:

```python
from multioutreg.surrogates.ard_gp_sklearn import ARDGPSurrogate

ard = ARDGPSurrogate().fit(X, Y)
for j, est in enumerate(ard.estimators_):
    print(j, est._gpr.kernel_.k1.k2.length_scale)   # one ℓ per feature
# 0 [1.52e+00 1.00e+05]   -> output 0 ignores feature 1
# 1 [1.00e+05 4.52e+01]   -> output 1 ignores feature 0
```

How to read the fitted values:

| Fitted value | Likely meaning | What to do |
|---|---|---|
| `ℓ` at the upper bound (`1e5`) | The output does not depend on that feature (ARD), or the response is essentially linear in it | Fine for ARD; it works as feature selection. For a scalar-ℓ GP, check whether a linear model is enough. |
| `ℓ` at the lower bound (`1e-5`) or much smaller than the spacing between training points | The GP is interpolating noise, or the data has duplicate `X` with different `Y` | Make sure a `WhiteKernel` is present, raise `alpha`, or add data. |
| `noise_level` at its lower bound (`1e-5`) | The data is effectively noise-free (common for deterministic simulators) | Nothing; this is expected. |
| `noise_level` comparable to `Var(Y)` | The kernel explains little of the signal | Try `matern_gp_kernel()`, ARD, or another surrogate family. |

The likelihood surface can have several optima. If fits vary between runs, pass
`n_restarts_optimizer=5` (sklearn GPs) or raise `n_start` (GPX).

---

## Overriding the defaults

Any `GaussianProcessRegressor` keyword passed to `GaussianProcessSurrogate` overrides the
default:

```python
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from multioutreg.surrogates.gp_sklearn import GaussianProcessSurrogate, matern_gp_kernel

# Matérn 5/2 instead of RBF
gp = GaussianProcessSurrogate(kernel=matern_gp_kernel())

# Your own initial length scale and bounds (in standardized units)
kernel = (ConstantKernel(1.0) * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e2))
          + WhiteKernel(1e-3, noise_level_bounds=(1e-8, 1e-1)))
gp = GaussianProcessSurrogate(kernel=kernel, n_restarts_optimizer=5)

# Fix the length scale (no optimization of ℓ)
gp = GaussianProcessSurrogate(kernel=ConstantKernel(1.0) * RBF(0.3, "fixed") + WhiteKernel(1e-2))
```

For the approximate GPs the length scale is an ordinary constructor argument:

```python
from multioutreg.surrogates.rfgp_sklearn import RFFGPSurrogate

rff = RFFGPSurrogate(length_scale=0.5, kernel="matern52", n_components=1000, random_state=0)
```

---

## What the model search tries

`AutoDetectMultiOutputRegressor` (`multioutreg/model_selection/auto_detect.py`) and the Grid
Search GUI (`multioutreg/gui/grid_search_core.py`) run `GridSearchCV` over these grids for the
GP-family models. Exact GPs search over the **kernel**, and their length scales are then fitted
by maximum likelihood inside each CV fold. The approximate GPs search over the **length scale
itself**, because they cannot learn it.

| Model key | Grid |
|---|---|
| `gp` / `gpr` | `kernel ∈ {default_gp_kernel(), matern_gp_kernel()}` |
| `ard_gp` | `alpha ∈ {1e-6, 1e-2}` |
| `rfgp` | `n_components ∈ {100, 500}`, `length_scale ∈ {0.1, 1.0, 10.0}`, `kernel ∈ {rbf, matern52}` |
| `nystroem_gp` / `sgp` | `n_components ∈ {50, 200}`, `gamma ∈ {None, 0.1, 1.0}` |
| `sgpr` (needs `gpytorch`) | `n_inducing ∈ {30, 50, 100}` in AutoDetect (`{30, 50}` in the GUI), `learning_rate ∈ {0.05, 0.1}` |
| `gpx` (needs `smt[gpx]`) | `corr ∈ {squar_exp, matern52}`, `poly ∈ {constant, linear}` |
| `kpls` (needs `smt`) | `n_comp ∈ {2, 4}`, `corr ∈ {squar_exp, matern52}` |
| `svr` | `C ∈ {1, 10}`, `gamma ∈ {scale, auto}` (an RBF kernel too) |

The `rfgp` and `nystroem_gp` grids are in raw input units (see the notes above). If your
features are far from unit scale, none of the three length scales may fit; standardize `X`
before the search.

---

## Not to be confused with: the active-learning length scale

`select_batch` in `multioutreg/active/batch.py` also takes a `length_scale`. It is **not** a
surrogate hyperparameter. It sets the radius of the distance penalty that spreads a batch of
picks apart, measured in the unit cube, and defaults to `n ** (−1/d)` (the typical spacing of a
space-filling design of `n` points in `d` dimensions). See [Active Learning](active_learning.md).
