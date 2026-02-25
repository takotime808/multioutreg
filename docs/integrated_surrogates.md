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
| **PolynomialBayesianRidgeSurrogate** | scikit-learn `PolynomialFeatures` + `BayesianRidge` | Per-output (MultiOutputRegressor) | `MultiOutputRegressor` (iterates `estimators_` for std) | ✗ | `MultiFidelitySurrogate` wrapper | ✓ | ✓ | ✓ | ✓ | O(C(p+d,d)³) | — | `pbr` — p < 20 | Analytic posterior over polynomial function class; captures nonlinearity without sampling | — |
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
| **BNNSurrogate** | PyTorch fully-connected network + MC Dropout | Joint (`_multi_output = True`) | Native joint architecture | ✗ | `MultiFidelitySurrogate` wrapper | ✗ | ✗ | ✓ (via checkbox) | ✗ | O(n·epochs·layers) | `pip install torch` | _(not screened)_ | Deep nonlinear model; joint output prediction captures inter-output correlations; best for n > 500 and complex response surfaces | N/A (opt-in) |
| **MixtureOfExpertsSurrogate** | Custom hard-EM gating network + K expert surrogates | Joint (`_multi_output = True`) | Native joint architecture | ✗ | `MultiFidelitySurrogate` wrapper | ✗ | ✓ (via `--use-moe`) | ✓ (via checkbox) | ✓ (via checkbox) | O(K × expert\_cost) | — | _(not screened)_ | Heterogeneous response surfaces where different input regions are governed by qualitatively different model families | N/A (opt-in) |
| **MultiFidelitySurrogate** | Composes around any `BaseSurrogate` | Wrapper (per wrapped surrogate) | Delegates to wrapped surrogate | ✓ | Native (is the wrapper) | ✗ | ✗ | ✗ | ✓ (as `mfs_lr`) | _(wrapped surrogate)_ | _(wrapped surrogate)_ | _(not screened)_ | Multi-fidelity datasets (e.g. coarse + fine simulation); maintains one surrogate instance per fidelity level — **no cross-fidelity coupling** | N/A (wrapper) |
| **StackedVFMSurrogate** | Composes around any surrogate(s) (default: `RandomForestSurrogate`) | Joint (`_multi_output = True`) | Delegates to per-level surrogates; level k input = `[X_k \| f₀(X_k) \| f₁(...)]` | ✓ | Native (recursive feature augmentation) | ✗ | ✗ | ✗ | ✗ | _(sum of per-level surrogate costs)_ | — | _(not screened)_ | Nonlinear multi-fidelity with N ≥ 2 levels; each level corrects the previous via augmented features; any surrogate mix per level; `augment_with_std=True` pipes uncertainty as features (Perdikaris et al. 2017) | N/A (opt-in) |
| **AdditiveCorrectionVFM** | Composes around any two surrogates (default: `RandomForestSurrogate` lo, `GaussianProcessSurrogate` delta) | Joint (`_multi_output = True`) | Two-surrogate composition: `f_hi = f_lo + δ` | ✓ | Native (additive correction) | ✗ | ✗ | ✗ | ✗ | _(lo cost + delta cost)_ | — | _(not screened)_ | Two-level additive correction (Kennedy–O'Hagan AR1); learns residual `δ = Y_hi − f_lo(X_hi)`; uncertainty combined in quadrature `σ_hi = √(σ_lo² + σ_δ²)`; `predict_components()` for diagnostics | N/A (opt-in) |
| **ElasticNetSurrogate** | scikit-learn `ElasticNet` | Per-output (MultiOutputRegressor) | `MultiOutputRegressor` | ✗ | `MultiFidelitySurrogate` wrapper | ✓ | ✓ | ✓ | ✓ | O(n·p) coordinate descent | — | `elastic_net` — always | Sparse linear regression (L1 + L2); zeros irrelevant features; preferred over Lasso when inputs are correlated | — |
| **LassoSurrogate** | scikit-learn `Lasso` | Per-output (MultiOutputRegressor) | `MultiOutputRegressor` | ✗ | `MultiFidelitySurrogate` wrapper | ✓ | ✓ | ✓ | ✓ | O(n·p) coordinate descent | — | `lasso` — always | Pure L1 sparse regression; hard feature zeroing; interpretable feature selection for p >> n | — |
| **QuantileRegressionSurrogate** | scikit-learn `QuantileRegressor` | Per-output (internal model list per quantile) | Three `QuantileRegressor` per output (q=α/2, 0.5, 1-α/2) | ✗ | `MultiFidelitySurrogate` wrapper | ✓ | ✓ | ✓ | ✓ | O(n·p) LP solver | — | `quantile` — always | Heteroscedastic asymmetric prediction intervals; robust to outliers; no normality assumption; use `predict_intervals()` for raw (lower, upper) bounds | — |
| **CatBoostSurrogate** | `catboost.CatBoostRegressor` | Per-output (MultiOutputRegressor) | `MultiOutputRegressor` | ✗ | `MultiFidelitySurrogate` wrapper | ~ | ~ | ~ | ~ | O(n·T·depth) | `pip install catboost` | `catboost` — nonlinear data | Only tree model with native aleatoric + epistemic decomposition (virtual ensembles via `RMSEWithUncertainty`); handles categoricals natively; strong on small tabular data | — |
| **DeepEnsembleSurrogate** | PyTorch feed-forward networks × N | Joint (`_multi_output = True`) | Native joint architecture | ✗ | `MultiFidelitySurrogate` wrapper | ✗ | ✗ | ✓ (via checkbox) | ✗ | O(N × n·epochs·layers) | `pip install torch` | _(not screened)_ | Empirical gold-standard for calibrated predictive uncertainty (Lakshminarayanan 2017); ensemble disagreement = epistemic std; no variational inference; outperforms BNN on calibration benchmarks | N/A (opt-in) |
| **SparseGPSurrogate** | GPyTorch SGPR with inducing points | Per-output (internal estimator list) | Internal `estimators_` list | ✗ | `MultiFidelitySurrogate` wrapper | ~ | ~ | ~ | ~ | O(n·m²) where m = n_inducing | `pip install gpytorch` | `sgpr` — N < 10000 | Scalable GP-quality posterior in the n = 300–10 000 regime; exact GP with Nyström inducing-point approximation (Titsias 2009); bridges the gap between O(n³) exact GP and kernel approximations | — |

## Notes

### Multi-Output Strategies in Detail

**Per-output (MultiOutputRegressor)** — `BaseSurrogate.__init__` wraps the sklearn estimator in
`sklearn.multioutput.MultiOutputRegressor`. One clone is fit independently per output column.
Output covariances are not modeled.

**Per-output (internal estimator list)** — Surrogates that maintain their own `self.estimators_`
list: `RFFGPSurrogate`, `NystroemGPSurrogate`, `GPXSurrogate`, `KPLSSurrogate`, `ARDGPSurrogate`.
They inherit from `ConformalMixin` rather than `BaseSurrogate` and manually stack per-column
predictions. Functionally equivalent to `MultiOutputRegressor` but necessary when the underlying
API (e.g. SMT) does not conform to the sklearn estimator interface, or when per-output
hyperparameters (e.g. per-output random seeds) are required.

`PolynomialBayesianRidgeSurrogate` inherits from `BaseSurrogate` (and therefore uses
`MultiOutputRegressor` internally), but overrides `predict` to iterate `model.estimators_`
directly when `return_std=True` in order to collect per-output posterior standard deviations.

**Joint (`_multi_output = True`)** — `BNNSurrogate`, `DeepEnsembleSurrogate`,
`MixtureOfExpertsSurrogate`, `StackedVFMSurrogate`, and `AdditiveCorrectionVFM` predict all
outputs simultaneously from a shared architecture (or composition). Inter-output correlations
can be captured. These cannot be wrapped in `MultiOutputRegressor` and are not included in the
standard AutoDetect grid search (they require a separate evaluation path via
`register_multi_output_candidates`).

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
| Always eligible | — | `linear`, `dt`, `bayesian_ridge`, `rfgp`, `nystroem_gp`, `elastic_net`, `lasso`, `quantile` |

### Optional Dependencies

| Package | Required By | Install |
|---------|------------|---------|
| `torch` | `BNNSurrogate`, `DeepEnsembleSurrogate` | `pip install torch` |
| `ngboost` | `NGBoostSurrogate` | `pip install ngboost` |
| `smt[gpx]` | `GPXSurrogate` | `pip install smt[gpx]` |
| `smt` | `KPLSSurrogate` | `pip install smt` |
| `lightgbm` | `LightGBMSurrogate` | `pip install lightgbm` |
| `xgboost` | `XGBoostSurrogate` | `pip install xgboost` |
| `catboost` | `CatBoostSurrogate` | `pip install catboost` |
| `gpytorch` | `SparseGPSurrogate` | `pip install gpytorch` |

All other surrogates (including `HistGradientBoostingSurrogate`, `ElasticNetSurrogate`, `LassoSurrogate`, `QuantileRegressionSurrogate`) depend only on `numpy` and `scikit-learn`.

---

## Time Series Forecasters (`multioutreg.time_series`)

These classes share a common `fit(y) → predict(prediction_length, quantiles) → ForecastResult`
contract.  `ForecastResult.quantiles` has shape `[n_series, n_quantiles, horizon]`.
They are **not** surrogate models and are **not** part of the regression grid search.

### Legend (TS)

| Symbol | Meaning |
|--------|---------|
| split-conformal | Calibrated via held-out residuals; coverage ≈ 1−α without distributional assumptions |
| Gaussian | `mu ± z_q × σ`; assumes normally-distributed forecast errors |
| linear interp | Lower/upper bounds linearly interpolated from (lo, median, hi) anchor points |
| propagated | Recursive uncertainty accumulation across multi-step horizon |

### Forecaster Table

| Class | Backing Library | Uncertainty Method | Fits to 1D Series | Multi-Series | Optional Dep | GUI Tab | Best Use Case |
|---|---|---|---|---|---|---|---|
| **ChronosForecaster** | Amazon Chronos (HuggingFace Transformers + PyTorch) | Quantile regression (internal model) | ✓ (also dict of arrays) | ✓ | `pip install chronos-forecasting torch` | Tab 1 | Zero-shot probabilistic forecasting; no training required; strong on data-sparse or novel series |
| **ProphetForecaster** | Meta Prophet (Stan/cmdstanpy) | linear interp of `yhat_lower`/`yhat`/`yhat_upper` | ✓ | ✗ (one series per fit) | `pip install prophet` | Tab 1 | Long series with strong trend + seasonality (daily/weekly/yearly); interpretable decomposition; handles holidays |
| **NeuralForecaster** | Nixtla NeuralForecast (PyTorch Lightning) | split-conformal (`val_size`) or point-only if insufficient data | ✓ | ✗ (one series per fit) | `pip install neuralforecast` | Tab 1 | Deep pattern learning on 100+ observations; N-BEATS (interpretable basis expansion) or N-HiTS (hierarchical interpolation) |
| **LagFeatureForecaster** | Any `BaseSurrogate` or sklearn estimator | split-conformal fallback (all estimators) or `return_std` (GP-family) | ✓ | ✗ | _(surrogate dep)_ | Tab 3 | Bridge all 35+ surrogates to time series; uncertainty via residual conformal calibration on held-out lag windows |
| **AutoSurrogateForecaster** | `AutoDetectMultiOutputRegressor` (all vendored surrogates) | split-conformal (wraps best selected surrogate) | ✓ | ✗ | _(surrogate deps)_ | Tab 3 (Auto) | Automatic surrogate selection for TS; runs the regression grid search on lag-feature matrices; best when series length ≥ 50 × n_lags |

### Walk-Forward Cross-Validation

| Class / Function | Description |
|---|---|
| `WalkForwardCV` | Expanding- or rolling-window walk-forward evaluator. Forecaster contract: `fit(train_series)`, `predict(horizon, quantiles) → ForecastResult`. Reports SMAPE, MASE, WQL per fold. |
| `walk_forward_splits(n, min_train, horizon, step, max_train)` | Generator of `(train_idx, test_idx)` pairs; no data leakage. |
| `TimeSeriesSplitWrapper` | sklearn `BaseCrossValidator` adapter for `cross_val_score` / `GridSearchCV`. |
| `TSFoldResult` | Dataclass: `fold_idx`, `train_size`, `test_size`, `y_true`, `y_pred`, `quantiles` `(Q, H)`, `q_levels`, `smape`, `mase`, `wql`. |

### Uncertainty Utilities (`multioutreg.time_series.uncertainty`)

| Function | Signature | Description |
|---|---|---|
| `gaussian_quantiles` | `(mean, std, quantiles) → (Q, H)` | Converts Gaussian `(mu, sigma)` to arbitrary quantile levels via `erfinv`. |
| `conformal_interval_from_residuals` | `(point_pred, cal_residuals, alpha) → (lower, upper)` | Split-conformal interval: `point_pred ± ceil((n+1)(1−α)/n)`-quantile of `|residuals|`. |
| `propagate_uncertainty_recursive` | `(single_step_std, horizon, correlation) → (H,)` | AR(1) uncertainty propagation: `σ_h² = σ₁²(1−ρ^{2h})/(1−ρ²)`; bounded variance for |ρ| < 1. |

### Optional Dependencies (TS)

| Package | Required By | Install |
|---------|------------|---------|
| `chronos-forecasting`, `torch` | `ChronosForecaster` | `pip install multioutreg[ts]` |
| `prophet` | `ProphetForecaster` | `pip install multioutreg[prophet]` |
| `neuralforecast` | `NeuralForecaster` | `pip install multioutreg[neuralforecast]` |
| `torch` | `LagFeatureForecaster` with `BNNSurrogate` or `DeepEnsembleSurrogate` | `pip install torch` |
