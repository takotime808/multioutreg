# Integrated Surrogate Models

Reference table for every surrogate registered in `multioutreg`.  All surrogates
expose a `predict(X, return_std=True)` interface and are compatible with
`ConformalMixin` for distribution-free prediction intervals.

## Legend

| Symbol | Meaning |
|--------|---------|
| ✓ | Integrated / supported |
| ✗ | Not integrated |
| ~ | Conditional on optional dependency |
| Per-output | One estimator per output column (MultiOutputRegressor or internal list) |
| Joint | Single model predicts all outputs simultaneously (`_multi_output = True`) |
| Wrapper | Composes around another surrogate; not a standalone model |
| AD | Skipped by Auto-Detect GUI "Skip computationally expensive models" checkbox |
| GS | Skipped by Grid-Search GUI "Skip computationally expensive models" checkbox |
| — | Not skipped by either GUI |
| N/A | Not part of the automatic grid search (opt-in or wrapper) |

## Surrogate Table

| Surrogate | Backing Library | Multi-Output Strategy | Multi-output via | Multi-Fidelity | Multi-fidelity via | Auto-Detect CLI | Grid-Search CLI | Auto-Detect GUI | Grid-Search GUI | Train Complexity | Optional Dep | Screening Key | Best Use Case | Skipped (expensive) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **LinearRegressionSurrogate** | scikit-learn `LinearRegression` | Per-output (MultiOutputRegressor) | `MultiOutputRegressor` | ✗ | `MultiFidelitySurrogate` wrapper | ✓ | ✗ | ✓ | ✗ | O(np²) | — | `linear` — always | Cheap interpretable baseline; well-conditioned linear data | — |
| **GaussianProcessSurrogate** | scikit-learn `GaussianProcessRegressor` | Per-output (MultiOutputRegressor) | `MultiOutputRegressor` | ✗ | `MultiFidelitySurrogate` wrapper | ✓ | ✓ | ✓ | ✓ | O(n³) | — | `gp` — N < 300 | Gold-standard calibrated uncertainty on small datasets (n ≤ 300) | AD, GS |
| **ARDGPSurrogate** | scikit-learn `GaussianProcessRegressor` + ARD `RBF` | Per-output (internal estimator list) | Internal `estimators_` list | ✗ | `MultiFidelitySurrogate` wrapper | ✓ | ✓ | ✓ | ✓ | O(n³) | — | `ard_gp` — N < 300 | GP with per-feature length scales; automatic relevance determination / soft feature selection via MLE | GS |
| **GPXSurrogate** | SMT/egobox Rust Kriging | Per-output (internal estimator list) | Internal `estimators_` list | ✗ | `MultiFidelitySurrogate` wrapper | ~ | ~ | ~ | ~ | O(n³) | `pip install smt[gpx]` | `gpx` — N < 300 | Drop-in GP replacement; 10–100× faster than sklearn GPR at the same O(n³) asymptotic cost | GS |
| **KPLSSurrogate** | SMT `KPLS` (Kriging + Partial Least Squares) | Per-output (internal estimator list) | Internal `estimators_` list | ✗ | `MultiFidelitySurrogate` wrapper | ~ | ~ | ~ | ~ | O(n³) | `pip install smt` | `kpls` — N < 300 | High-dimensional inputs (p ≫ n); PLS projection reduces effective input dimensionality before Kriging | GS |
| **RFFGPSurrogate** | Custom RFF + scikit-learn `BayesianRidge` | Per-output (internal estimator list) | Internal `estimators_` list | ✗ | `MultiFidelitySurrogate` wrapper | ✓ | ✓ | ✓ | ✓ | O(D·n) | — | `rfgp` — always | Scalable GP approximation via Bochner spectral sampling; supports RBF, Matérn 3/2, Matérn 5/2 kernels | — |
| **NystroemGPSurrogate** | scikit-learn `Nystroem` + `BayesianRidge` | Per-output (internal estimator list) | Internal `estimators_` list | ✗ | `MultiFidelitySurrogate` wrapper | ✓ | ✓ | ✓ | ✓ | O(m²n + m³) | — | `sgp` — always | Data-adaptive landmark GP approximation; superior to RFF when m ≪ n and the data has exploitable structure | — |
| **BayesianRidgeSurrogate** | scikit-learn `BayesianRidge` | Per-output (MultiOutputRegressor) | `MultiOutputRegressor` | ✗ | `MultiFidelitySurrogate` wrapper | ✓ | ✓ | ✓ | ✓ | O(p³) | — | `bayesian_ridge` — always | Analytic Bayesian posterior over linear weights; fast and calibrated for low-dimensional linear problems | — |
| **PolynomialBayesianRidgeSurrogate** | scikit-learn `PolynomialFeatures` + `BayesianRidge` | Per-output (internal estimator list) | Internal `estimators_` list | ✗ | `MultiFidelitySurrogate` wrapper | ✓ | ✓ | ✓ | ✓ | O(C(p+d,d)³) | — | `pbr` — p < 20 | Analytic posterior over polynomial function class; captures nonlinearity without sampling | — |
| **RandomForestSurrogate** | scikit-learn `RandomForestRegressor` | Per-output (MultiOutputRegressor) | `MultiOutputRegressor` | ✗ | `MultiFidelitySurrogate` wrapper | ✓ | ✓ | ✓ | ✓ | O(n·p·log n·T) | — | `rf` — nonlinear data | Robust nonlinear baseline; uncertainty from per-tree variance; handles mixed feature types | — |
| **ExtraTreesRegressorSurrogate** | scikit-learn `ExtraTreesRegressor` | Per-output (MultiOutputRegressor) | `MultiOutputRegressor` | ✗ | `MultiFidelitySurrogate` wrapper | ✓ | ✗ | ✓ | ✓ | O(n·p·T) | — | `et` — nonlinear data | Faster than RF via random split thresholds; lower variance than single DT; good when RF overfits | — |
| **GradientBoostingSurrogate** | scikit-learn `GradientBoostingRegressor` | Per-output (MultiOutputRegressor) | `MultiOutputRegressor` | ✗ | `MultiFidelitySurrogate` wrapper | ✓ | ✓ | ✓ | ✓ | O(n·p·T) | — | `gb` — nonlinear data | Strong sequential error-correction; often top performer on tabular engineering data | — |
| **NGBoostSurrogate** | ngboost `NGBRegressor` | Per-output (MultiOutputRegressor) | `MultiOutputRegressor` | ✗ | `MultiFidelitySurrogate` wrapper | ~ | ✗ | ~ | ~ | O(n·p·T) | `pip install ngboost` | `ngboost` — nonlinear data | Native probabilistic boosting; returns full predictive distribution without bootstrap tricks | AD, GS |
| **HistGradientBoostingSurrogate** | scikit-learn `HistGradientBoostingRegressor` | Per-output (MultiOutputRegressor) | `MultiOutputRegressor` | ✗ | `MultiFidelitySurrogate` wrapper | ✓ | ✓ | ✓ | ✓ | O(n·bins·T) | — | `hgb` — nonlinear data | 5–10× faster than GradientBoostingSurrogate via histogram binning; preferred for n > 5 000; no extra dependencies | — |
| **LightGBMSurrogate** | `lightgbm.LGBMRegressor` | Per-output (MultiOutputRegressor) | `MultiOutputRegressor` | ✗ | `MultiFidelitySurrogate` wrapper | ~ | ~ | ~ | ~ | O(n·T·leaves) | `pip install lightgbm` | `lgbm` — nonlinear data | 10–20× faster than sklearn GradientBoosting; leaf-wise growth; GPU support via `device="gpu"` | — |
| **XGBoostSurrogate** | `xgboost.XGBRegressor` | Per-output (MultiOutputRegressor) | `MultiOutputRegressor` | ✗ | `MultiFidelitySurrogate` wrapper | ~ | ~ | ~ | ~ | O(n·T·depth) | `pip install xgboost` | `xgb` — nonlinear data | Engineering competition standard; native missing-value handling; GPU support via `device="cuda"` | — |
| **SVRSurrogate** | scikit-learn `SVR` | Per-output (MultiOutputRegressor) | `MultiOutputRegressor` | ✗ | `MultiFidelitySurrogate` wrapper | ✓ | ✗ | ✓ | ✓ | O(n²p) | — | `svr` — N < 2000 | Robust kernel regression with support vector margin; smooth response surfaces for n < 2000 | — |
| **KNeighborsSurrogate** | scikit-learn `KNeighborsRegressor` | Per-output (MultiOutputRegressor) | `MultiOutputRegressor` | ✗ | `MultiFidelitySurrogate` wrapper | ✓ | ✓ | ✓ | ✓ | O(1) train / O(np) predict | — | `knn` — N/p > 10 | Zero-training-cost instance-based fallback; interpretable neighborhood averages | — |
| **DecisionTreeRegressorSurrogate** | scikit-learn `DecisionTreeRegressor` | Per-output (MultiOutputRegressor) | `MultiOutputRegressor` | ✗ | `MultiFidelitySurrogate` wrapper | ✓ | ✗ | ✓ | ✓ | O(n·p·log n) | — | `dt` — always | Interpretable piecewise-constant model; cheap nonlinear baseline; useful for rule extraction | — |
| **ConformalPredictionNetworkSurrogate** | scikit-learn `MLPRegressor` + split-conformal calibration | Per-output (MultiOutputRegressor) | `MultiOutputRegressor` | ✗ | `MultiFidelitySurrogate` wrapper | ✓ | ✗ | ✓ | ✓ | O(n·layers·epochs) | — | `mlp`/`cpn` — N > 100, nonlinear | Neural network with distribution-free coverage guarantees; preferred when interval coverage matters more than Bayesian calibration | AD, GS |
| **BNNSurrogate** | PyTorch fully-connected network + MC Dropout | Joint (`_multi_output = True`) | Native joint architecture | ✗ | `MultiFidelitySurrogate` wrapper | ✗ | ✗ | ✗ | ✗ | O(n·epochs·layers) | `pip install torch` | _(not screened)_ | Deep nonlinear model; joint output prediction captures inter-output correlations; best for n > 500 and complex response surfaces | N/A (opt-in) |
| **MixtureOfExpertsSurrogate** | Custom hard-EM gating network + K expert surrogates | Joint (`_multi_output = True`) | Native joint architecture | ✗ | `MultiFidelitySurrogate` wrapper | ✗ | ✓ (via `--use-moe`) | ✗ | ✓ (via checkbox) | O(K × expert\_cost) | — | _(not screened)_ | Heterogeneous response surfaces where different input regions are governed by qualitatively different model families | N/A (opt-in) |
| **MultiFidelitySurrogate** | Composes around any `BaseSurrogate` | Wrapper (per wrapped surrogate) | Delegates to wrapped surrogate | ✓ | Native (is the wrapper) | ✗ | ✗ | ✗ | ✓ (as `mfs_lr`) | _(wrapped surrogate)_ | _(wrapped surrogate)_ | _(not screened)_ | Multi-fidelity datasets (e.g. coarse + fine simulation); maintains one surrogate instance per fidelity level — **no cross-fidelity coupling** | N/A (wrapper) |
| **StackedVFMSurrogate** | Composes around any surrogate(s) (default: `RandomForestSurrogate`) | Joint (`_multi_output = True`) | Delegates to per-level surrogates; level k input = `[X_k \| f₀(X_k) \| f₁(...)]` | ✓ | Native (recursive feature augmentation) | ✗ | ✗ | ✗ | ✗ | _(sum of per-level surrogate costs)_ | — | _(not screened)_ | Nonlinear multi-fidelity with N ≥ 2 levels; each level corrects the previous via augmented features; any surrogate mix per level; `augment_with_std=True` pipes uncertainty as features (Perdikaris et al. 2017) | N/A (opt-in) |
| **AdditiveCorrectionVFM** | Composes around any two surrogates (default: `RandomForestSurrogate` lo, `GaussianProcessSurrogate` delta) | Joint (`_multi_output = True`) | Two-surrogate composition: `f_hi = f_lo + δ` | ✓ | Native (additive correction) | ✗ | ✗ | ✗ | ✗ | _(lo cost + delta cost)_ | — | _(not screened)_ | Two-level additive correction (Kennedy–O'Hagan AR1); learns residual `δ = Y_hi − f_lo(X_hi)`; uncertainty combined in quadrature `σ_hi = √(σ_lo² + σ_δ²)`; `predict_components()` for diagnostics | N/A (opt-in) |

## Notes

### Multi-Output Strategies in Detail

**Per-output (MultiOutputRegressor)** — `BaseSurrogate.__init__` wraps the sklearn estimator in
`sklearn.multioutput.MultiOutputRegressor`. One clone is fit independently per output column.
Output covariances are not modeled.

**Per-output (internal estimator list)** — Surrogates that maintain their own `self.estimators_`
list: `RFFGPSurrogate`, `NystroemGPSurrogate`, `GPXSurrogate`, `KPLSSurrogate`, `ARDGPSurrogate`,
`PolynomialBayesianRidgeSurrogate`. They inherit from `ConformalMixin` rather than `BaseSurrogate`
and manually stack per-column predictions. Functionally equivalent to `MultiOutputRegressor` but
necessary when the underlying API (e.g. SMT) does not conform to the sklearn estimator interface,
or when per-output hyperparameters (e.g. per-output random seeds) are required.

**Joint (`_multi_output = True`)** — `BNNSurrogate`, `MixtureOfExpertsSurrogate`,
`StackedVFMSurrogate`, and `AdditiveCorrectionVFM` predict all outputs simultaneously from a
shared architecture (or composition). Inter-output correlations can be captured.
These cannot be wrapped in `MultiOutputRegressor` and are not included in the standard AutoDetect
grid search (they require a separate evaluation path).

### Computational Complexity Notation

| Symbol | Meaning |
|--------|---------|
| n | Number of training samples |
| p | Number of input features |
| D | Number of random Fourier features (RFF) |
| m | Number of Nyström landmark points |
| T | Number of trees (RF, ET) or boosting rounds (GB, NGBoost) |
| K | Number of MoE experts |
| d | Polynomial degree (PBR) |

### Statistical Pre-Screening Gates (`pre_screen=True`)

When `AutoDetectMultiOutputRegressor.with_vendored_surrogates(pre_screen=True)` is used,
`ModelScreener` runs inexpensive statistical tests before fitting and gates expensive or
unlikely-to-help models per output column:

| Gate | Condition | Models Gated |
|------|-----------|-------------|
| `gp_feasible` | N < 300 | `gp`, `gpx`, `ard_gp`, `kpls` |
| `svr_feasible` | N < 2000 | `svr` |
| `mlp_feasible` | N > 100 | `mlp`, `cpn` |
| `pbr_feasible` | p < 20 | `pbr` |
| `knn_ratio_ok` | N/p > 10 | `knn` |
| Nonlinear detected | Ramsey RESET or RF–LR R² gain > 0.05 | `rf`, `et`, `gb`, `mlp`, `hgb`, `lgbm`, `xgb` |
| Always eligible | — | `linear`, `dt`, `bayesian_ridge`, `rfgp`, `sgp` |

### Optional Dependencies

| Package | Required By | Install |
|---------|------------|---------|
| `torch` | `BNNSurrogate` | `pip install torch` |
| `ngboost` | `NGBoostSurrogate` | `pip install ngboost` |
| `smt[gpx]` | `GPXSurrogate` | `pip install smt[gpx]` |
| `smt` | `KPLSSurrogate` | `pip install smt` |
| `lightgbm` | `LightGBMSurrogate` | `pip install lightgbm` |
| `xgboost` | `XGBoostSurrogate` | `pip install xgboost` |

All other surrogates (including `HistGradientBoostingSurrogate`) depend only on `numpy` and `scikit-learn`.
