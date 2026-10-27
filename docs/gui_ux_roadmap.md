# Improving the GUI

A plan for making the Streamlit GUI (`multioutreg/gui/`) faster to get started
with, easier to read, and more worth coming back to. The grid search page,
`Grid_Search_Surrogate_Models.py`, is where most users start, so most items
begin there. See [Multi-Fidelity GUI](multi_fidelity_gui.md) for the
multi-fidelity page.

*Last reviewed 2026-10-26, against `main` at #45 plus the `fix-wheel-packages` branch.*

## Status at a Glance

| # | Item | Status |
|---|---|---|
| 0 | Remove the duplicate pipeline run on the grid search page | Done (#43) |
| 1 | Keep results on screen across reruns | Done (#43) |
| 2 | Show progress during a run | Done (#43) |
| 3 | Let people try the app without their own data | Sample data done (#45), walkthrough and background pair plot done; input checks open |
| 4 | Show results in the app, not only in the downloaded report | Open |
| 5 | Navigation and page structure | Open |
| 6 | Engagement features | Open; hosted demo already exists |
| 7 | Packaging: stray top-level folders in the wheel | Fixed (`fix-wheel-packages`) |

## Done

### 0. Duplicate pipeline run (#43)

A merge left two copies of the pipeline on the grid search page. The second
copy applied PCA a second time, picked models and MoE by test-set MSE, and
replaced the correct validation-split result, while roughly doubling the
runtime. It was removed. Models are now chosen only by `select_per_target` on a
validation split of the training data, and the report's `model_type` names the
model that was actually chosen.

### 1. Results survive reruns (#43)

Results used to be drawn only on the run that followed **Run Grid Search**, so
any later click (including **Download HTML Report**) wiped them. A run now saves
its results and messages to `st.session_state["gs_results"]`, and the page draws
them on every rerun. A caption names the inputs and outputs of the run being
shown, and the results are cleared when a different file is loaded. CSV parsing
is cached with `st.cache_data`.

### 2. Progress feedback (#43)

`select_per_target` takes an optional `on_progress(done, total, label)`
callback. The page shows an `st.status` box that steps through preparing data,
PCA, pre-screening, a per-candidate progress bar, metrics and the report, and
ends as "Grid search complete in Ns", or in an error state if the run fails.

### 3a. Sample data (#45)

When nothing is uploaded, a **Try with sample data** button loads
`multioutreg/gui/data/example_dataset_3_outputs.csv`, a synthetic dataset with
300 rows, 5 inputs and 3 outputs. The file is shipped as package data, so the
button also works from a pip install. Inputs and outputs are pre-selected, so
one click on **Run Grid Search** runs it. A run with all default models takes
about 70 seconds and reaches R² of 0.98 or higher on every output. An upload
always replaces the sample.

### 3b. Sample data walkthrough and background plots (`sample-data-experience`)

With the sample loaded, the page now:

- says what the data is and what to look for (each output is driven mainly by
  one input), offers **Download sample CSV** to show the expected format, a
  **Stop using sample data** button, and a prefilled project description;
- draws a **pair plot** of the selected inputs and outputs, right away for the
  sample (its columns are known) and on **Run Grid Search** for an upload. It
  is the "fast" grid from [Pair Plots in Reports](pair_plot_report.md):
  scatter / 2-D histogram / histogram, blue input labels, orange output labels,
  at most 1,000 rows and 10 variables (then outputs vs inputs only, then
  skipped with a note). See `multioutreg.figures.doe_plots.pair_plot_b64`.

Slow plots no longer hold up the run. `multioutreg/gui/background.py` runs them
in worker processes (`spawn`, one worker per pool):

| Job | Pool | Started | Shown |
|---|---|---|---|
| Pair plot (about 7 s for the sample's 8 variables on a 2014 laptop) | `plots` | Sample load, or Run | As soon as it is ready, including mid-run (the progress callback checks it) |
| HTML report (prediction, SHAP, PDP, UMAP, ... plots) | `report` | After the test metrics | Metrics and tables show at once; the download button appears when the report is done |

The `report` pool imports the report modules (UMAP takes up to a minute) when
its worker starts, and the page starts it as soon as data is loaded, so that
cost overlaps with choosing columns. The page waits for unfinished jobs only at
the end of the script, updating a message every second, so a click still
interrupts it; the jobs keep running and the next rerun picks them up. A job
whose arguments cannot be pickled (or whose pool dies) runs inline instead.

Processes rather than threads, because pyplot's "current figure" is global
(`safe_plot_b64` saves whatever figure is current) and plotting holds the GIL.
Each worker costs memory (see [Parallelization](parallel_computing.md)), which
is why each pool has a single worker. Set `MOR_BACKGROUND_JOBS=0` to run
everything inline, e.g. on a host with very little memory; the test suite does
this in `tests/conftest.py`.

### 7. Stray top-level packages in the wheel (`fix-wheel-packages`)

`where = [".", "mor_cli", "multioutreg"]` in
`[tool.setuptools.packages.find]` made each entry a separate search root, so
the wheel installed 15 top-level packages instead of 2. Every subpackage of
`multioutreg` was installed a second time under a bare name (`active`,
`conformal`, `figures`, `gui`, `metrics`, `model_selection`, `performance`,
`report`, `sampling`, `surrogates`, `time_series`, `utils`), and the whole test
suite shipped as `tests`. Generic names like `utils` and `tests` clash with
other packages that make the same mistake, and `surrogates.X` and
`multioutreg.surrogates.X` were two separate classes.

The fix is `where = ["."]` with `include = ["multioutreg*", "mor_cli*"]`. The
wheel now contains only `multioutreg` and `mor_cli` (794K down to 337K), with
the same 142 files under `multioutreg/` and both package-data files. In a fresh
non-editable install, the stray names no longer import, the `multioutreg` CLI
runs, and the grid search page runs the sample data end to end.

Editable installs made before the fix keep the old mapping until they are
reinstalled (`uv sync`, or `uv pip install --no-deps -e .`).

## Open

Effort: **Low** is under a day, **Medium** is one to three days, **High** is
more than that.

### 3b. Check inputs before a run (Low)

**Problem:** bad column choices only show up as a stack trace or as fits that
fail silently. Clicking **Run Grid Search** with no columns chosen now gives a
warning, but nothing else is checked.

**Plan:**
1. Block a column chosen as both an input and an output.
2. Flag non-numeric columns among the inputs and outputs, and say which ones.
3. Warn when there are too few rows for the train/test split, the 20%
   validation split and the optional conformal calibration split.
4. Show the problems inside the form, before any fitting starts.

**Done when:** each case has an `AppTest` test in
`tests/unit/gui/test_Grid_Search_app.py` that shows the message and does not
run the grid search.

### 4. Results in the app (Medium)

**Problem:** the page shows the best configuration as raw JSON and the metrics
as plain tables. The prediction, uncertainty, SHAP, PDP and UMAP plots exist
only inside the downloaded HTML report, as static images.

**Plan:**
1. Put results in tabs: **Summary / Predictions / Uncertainty / Explainability /
   Report**.
2. Summary: an `st.metric` card per output (R², RMSE) and a readable table of
   the chosen model per output, in place of `st.json`.
3. Predictions and Uncertainty: interactive Plotly parity plots and ordered
   interval plots, drawn from the arrays already saved in `gs_results`.
4. Explainability: the existing SHAP and PDP figures, generated once and saved
   in `gs_results` so reruns stay fast.
5. Report: an inline preview with `st.components.v1.html`, next to the
   download button.

**Done when:** every figure in the report also appears in a tab, switching tabs
does not refit anything, and the existing app tests still pass.

### 5. Navigation and page structure (Medium)

**Problem:**
- Two pages share the `05_` prefix (`05_Sampling_Datasets_Examples.py` and
  `05_Time_Series_Forecasting.py`).
- Five unnumbered pages overlap with pages 03 and 04: `Detect_Sampling.py`,
  `PCA_Sample_detection.py`, `lhs_detector_app.py`, `sample_infer.py` and
  `sample_infer_dashboard.py`.
- Only 4 of the 14 pages call `st.set_page_config`, so layouts differ from page
  to page.

**Plan:**
1. Switch to `st.navigation` with grouped sections: **Modeling** (grid search,
   Auto-Detect, multi-fidelity), **Sampling & DOE**, **Time Series** and
   **Examples**.
2. Merge the sampling-detection pages into one page, with tabs for the
   detection method, the PCA view, LHS detection and bias alerts.
3. Set the page config once, in the entry point.
4. Add a short landing page that explains the workflow and links to each
   section.

**Open question:** which legacy pages are still in use? This needs an answer
before any page is merged or removed.

**Done when:** every page is reachable from the grouped navigation, no
functionality is lost in the merge, and the page tests under
`tests/unit/gui/pages/` pass.

### 6. Engagement features (Medium)

**Problem:** 11 pages have their own uploaders (13 in total), so moving between
pages means uploading the same file again. Nothing on a results page suggests
what to try next.

**Plan:**
1. Upload once: keep the current dataset in `st.session_state`, show it in the
   sidebar, and let every page use it.
2. "Next step" links after each result. The time series check on the grid
   search page already uses `st.page_link` for this.
3. Export and import run settings as JSON, so a run can be repeated or shared.
4. Compare runs from the same session side by side.
5. Add a theme in `.streamlit/config.toml` (none exists yet).

A hosted demo already exists (linked from the README), so it is not part of
this item. The demo gets the sample-data button (#45) once it is redeployed
from `main`.

## Suggested Order

1. **3b**: small, and it makes the first run fail safely.
2. **4**: the largest gain for users, since the most useful output is currently
   hidden in the download.
3. **5**: after the legacy-page question is answered.
4. **6**: builds on the shared navigation from item 5.

## Testing

The GUI tests drive the real page scripts with Streamlit's `AppTest`.
`tests/unit/gui/app_harness.py` replaces the uploader with a stub (returning a
CSV path, or `None` for "nothing uploaded") and can shorten the candidate list
so a run takes seconds. Each item above should add `AppTest` coverage for its
new behaviour, and end with a quick manual check in a browser, since `AppTest`
cannot simulate a real file upload.
