# Multi-Fidelity Surrogate Models page

The Streamlit page `multioutreg/gui/pages/02_Multi_Fidelity_Surrogate_Models.py` fits
multi-fidelity surrogates to a single CSV, picks the best configuration by grid search, and
produces a downloadable HTML report (`mf_model_report.html`).

```bash
.venv/bin/streamlit run multioutreg/gui/Grid_Search_Surrogate_Models.py
# then choose "Multi Fidelity Surrogate Models" in the sidebar
```

An example dataset (`x1, x2, fidelity_level, y1, y2`, three levels) can be downloaded from the
**Other Dataset Examples** page.

## Input

One CSV with a **fidelity level column** plus input and output columns. Each unique value of the
fidelity column is one level. In the form you choose:

| Field | Meaning |
|-------|---------|
| Fidelity level column | Column whose values identify the levels (e.g. `0, 1, 2` or `lo, hi`). |
| Input / output columns | Features and targets. The fidelity column cannot be either. |
| Fidelity levels — ordered lowest → highest | At least 2. The **last** entry is the highest fidelity; everything is evaluated on it. |
| Skip computationally expensive models | Leaves out the sklearn GP (O(n³)). |
| Fix random seed / Random seed | See [Reproducibility](#reproducibility). On by default (seed 0). |
| Compute conformal prediction intervals | Adds split-conformal intervals at the chosen alpha. |

Missing values: target rows with missing values are always dropped; inputs can be dropped or
KNN-imputed, with the imputer fitted on each level's training split only.

## What the page does

1. **Split.** Each level is split 75/25 into train/test. Only the **highest-fidelity test set** is
   used for reported metrics. With conformal on, 20% of the highest-fidelity training split is
   held out for calibration and never used for fitting.
2. **Grid search.** Every combination of multi-fidelity type and base surrogate is fitted:

   | Type | Candidate name | How levels are used |
   |------|----------------|---------------------|
   | `StackedVFMSurrogate` | `StackedVFM+<base>` (`+augStd` also feeds lower-level std) | Each level's model gets the lower levels' predictions as extra inputs. Any number of levels. |
   | `AdditiveCorrectionVFM` | `AdditiveVFM+<base>` | `f_hi = f_lo + δ`. Only offered with exactly 2 levels. |
   | `MultiFidelitySurrogate` | `MultiFidelity+<base>` | One independent model per level. **No coupling**: the highest-level prediction ignores the lower levels. |

   Base surrogates: RF, HGB, Linear, ElasticNet, Lasso, GP (unless skipped), plus LightGBM,
   XGBoost, CatBoost and SparseGP when their packages are installed.
3. **Selection.** Candidates are fitted on the training data minus a 20% validation split of the
   highest-fidelity training data and ranked by validation MSE. The test set is not used.
4. **Refit and evaluate.** The winner is refit on all training data and scored once on the
   highest-fidelity test set.
5. **Comparison models.** The winner's base surrogate is also fitted on each level alone (used
   for the Fidelity Levels table; the highest level is the HF-only baseline). For an independent
   (`MultiFidelity+…`) winner the fitted per-level models are reused, so its baseline comparison
   is exact.

## Reading the report

| Section | What it shows | How to read it |
|---------|---------------|----------------|
| Summary | Auto-generated findings from the metrics below. | Start here. |
| Fidelity Levels | Per level and output: train/test n, output mean ± std, and **r** / **R²** of a model trained on that level alone, scored on the highest-fidelity test outputs. | High r with lower R² = informative but biased (multi-fidelity can correct this). Low r = the level carries little usable information. |
| Regression Metrics | R², RMSE, MAE and mean predicted std per output (highest-fidelity test set). | |
| Multi-Fidelity vs HF-only Baseline | Same base surrogate trained only on highest-fidelity data. | Negative RMSE change = the lower levels helped. |
| Model Selection | Every candidate ranked by validation MSE; winner highlighted; failures with their error. | A small gap to the runner-up means the choice is not robust. |
| Uncertainty Metrics / Coverage Plot | Sharpness (mean std), empirical coverage of the 95% Gaussian interval, miscoverage, Gaussian NLL (omitted when any std is 0). | Coverage below nominal = overconfident std; above = conservative. |
| Conformal Prediction Intervals | Split-conformal intervals and Gaussian-vs-conformal coverage curves (calibration residuals, not test residuals). | Coverage ≥ 1−α holds on average when calibration and test data are exchangeable. |
| Predictions vs True | Ordered intervals and per-output scatter with ±std error bars. | |
| SHAP / PDP | Explanations of the **highest-fidelity** predictions, with input column names. | |
| Sampling & Input Space | Sampling method inferred from the full highest-fidelity input design (Monte Carlo test from `multioutreg.sampling.detect`), and a scatter (2 inputs) or UMAP (> 2 inputs) of that design. | A statistical statement, not proof; see Appendix B of `AUDIT_CHECKLIST.md` for limits. |
| Additional Diagnostics | Residuals (observed − predicted) vs predicted with a trend band; error histograms (predicted − observed). | A trend or offset from zero = systematic bias. Note the opposite sign conventions. |
| Appendix | Per-level training sizes, test size, selection/calibration scheme, seed. | |

### Interpretation caveats

- **An independent winner means multi-fidelity did not help.** `MultiFidelity+<base>` uses only
  highest-fidelity data for its highest-level prediction, so if it wins, no coupled model beat
  high-fidelity data alone on validation. The Summary says so explicitly.
- **Small test sets are noisy.** The Summary warns below 30 highest-fidelity test points. The
  MF-vs-baseline difference on such sets can be within run-to-run noise.
- **Uncertainty can be badly miscalibrated** (e.g. SparseGP std collapsing to its floor on
  noiseless data). Prefer the conformal intervals when the coverage plot falls below the diagonal.

## Reproducibility

With **Fix random seed** checked (the default), one seed drives:

- the train/test, validation and calibration splits;
- every base surrogate with randomness, via its own seed argument (`random_state` for RF, HGB,
  GP, ElasticNet/Lasso, LightGBM, XGBoost and SparseGP; `random_seed` for CatBoost; Linear is
  deterministic). Mapping: `_SEED_KWARG` / `_seeded` in the page module;
- the bootstrap band of the residual plot.

Two runs with the same data, settings and seed produce a byte-identical report. Uncheck the box
to see run-to-run variation (the report then records the seed as "not fixed"). Without a seed,
`SparseGPSurrogate` in particular gives a different model on every fit.
