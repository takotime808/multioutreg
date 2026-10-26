# Improving the GUI

A plan for making the Streamlit GUI (`multioutreg/gui/`) faster to get started
with, easier to read, and more worth coming back to. The grid search page,
`Grid_Search_Surrogate_Models.py`, is where most users start, so most items
begin there. See [Multi-Fidelity GUI](multi_fidelity_gui.md) for the
multi-fidelity page.

*Last reviewed 2026-10-26, against `main` at #43 plus the `gui-sample-data` branch.*

## Status at a Glance

| # | Item | Status |
|---|---|---|
| 0 | Remove the duplicate pipeline run on the grid search page | Done (#43) |
| 1 | Keep results on screen across reruns | Done (#43) |
| 2 | Show progress during a run | Done (#43) |
| 3 | Let people try the app without their own data | Sample data done (`gui-sample-data`); input checks open |
| 4 | Show results in the app, not only in the downloaded report | Open |
| 5 | Navigation and page structure | Open |
| 6 | Engagement features | Open; hosted demo already exists |
| 7 | Packaging: stray top-level folders in the wheel | Open |

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

### 3a. Sample data (`gui-sample-data`)

When nothing is uploaded, a **Try with sample data** button loads
`multioutreg/gui/data/example_dataset_3_outputs.csv`, a synthetic dataset with
300 rows, 5 inputs and 3 outputs. The file is shipped as package data, so the
button also works from a pip install. Inputs and outputs are pre-selected, so
one click on **Run Grid Search** runs it. A run with all default models takes
about 70 seconds and reaches R² of 0.98 or higher on every output. An upload
always replaces the sample.

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
this item. Once 3a is merged, the demo also gets the sample-data button.

### 7. Packaging: stray top-level folders in the wheel (Low)

**Problem:** the built wheel contains top-level `gui/`, `report/` and `tests/`
folders that duplicate files under `multioutreg/`. They come from
`where = [".", "mor_cli", "multioutreg"]` in `[tool.setuptools.packages.find]`
and can clash with other installed packages named `gui` or `tests`.

**Plan:** find packages from the project root only, and keep `mor_cli` as a
package of its own. Then build a wheel and check that its contents are only
`multioutreg/` and `mor_cli/`.

**Done when:** the wheel has no stray top-level folders, and
`multioutreg` (the CLI), the GUI and the sample-data button work from a fresh
non-editable install.

## Suggested Order

1. **3b**: small, and it makes the first run fail safely.
2. **7**: small, and best fixed before more package data is added.
3. **4**: the largest gain for users, since the most useful output is currently
   hidden in the download.
4. **5**: after the legacy-page question is answered.
5. **6**: builds on the shared navigation from item 5.

## Testing

The GUI tests drive the real page scripts with Streamlit's `AppTest`.
`tests/unit/gui/app_harness.py` replaces the uploader with a stub (returning a
CSV path, or `None` for "nothing uploaded") and can shorten the candidate list
so a run takes seconds. Each item above should add `AppTest` coverage for its
new behaviour, and end with a quick manual check in a browser, since `AppTest`
cannot simulate a real file upload.
