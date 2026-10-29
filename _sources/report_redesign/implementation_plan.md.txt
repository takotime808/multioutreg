# Plan: ship the redesigned report as a downloadable file

Status: **implemented** (2026-10-28), phases 1–5; the redesigned report is the default.
Design reference: `docs/_extra/report_redesign/report_redesign.html` (mockups A–D: front
page, per-output, error bars, fidelity & drivers), served with the docs at
`report_redesign/report_redesign.html`.

### What shipped

- `multioutreg/report/data.py`, `verdict.py`, `build.py`, `render.py`, with
  `templates/report_v2.html.j2`, `static/report_v2.css` and `static/report_v2.js` (no
  dependencies, inlined into every report).
- All three report callers branch on `report_style` (`"v2"` default, `"classic"`). The
  classic path is unchanged: with embedded images masked (they vary between runs even
  on the old code), its HTML is identical to the previous version for all three callers.
- GUI: a **Report format** radio on each page, and **Download report data (.json)**
  next to the HTML download (`multioutreg/gui/report_controls.py`). CLI:
  `--report-style`, plus real uncertainty metrics in place of the `dummy_metric`
  placeholder both commands passed.
- CV fold spread: `CV RMSE std` on the multi-fidelity page's rows, and
  `GridSearchResult.cv_table` (one row per config, with the outputs it won) from
  `select_per_target`. SHAP shares and PDP grids come from
  `figures.shap_multioutput.shap_importance_table` (seeded, so reports are
  reproducible) and `figures.pdp_plots.pdp_grid_data`.
- `_summary_points` moved to `multioutreg.report.verdict.summary_points` (page 02
  imports it under the old name).
- Docs: {doc}`../reading_the_report`, {doc}`../example_reports`, and
  `examples/make_example_reports.py`, which regenerates the examples in both styles.

Departures from the plan: the classic path stays in each caller rather than going
through `render_report` (that is what keeps it byte-for-byte unchanged); the "Skipped"
lines come from `build_report_data` instead of `safe_plot_b64` callers, because the v2
path no longer draws those PNGs at all; and the regret table lives in section 2.

## Goal

Users get the redesigned report (verdict and scorecard first, the seven question-led
sections, small multiples, a collapsed appendix) **as the file they download**, with the
same download buttons and CLI output they use today. The downloaded copy must:

1. Be **one self-contained `.html` file**: no network requests, no sidecar files. It opens
   from disk, from an email attachment or from a shared drive.
2. **Work without JavaScript** for the parts that carry the message (verdict, scorecard,
   findings, next steps, every table). Charts can need JS; a `<noscript>` note says so.
3. **Print or save to PDF cleanly** (all outputs shown, appendix expanded, no nav chrome).
4. Stay **small**: under 2 MB for a typical run (3–6 outputs, ≤ 500 test points).
5. Keep the **classic report available** while the new one reaches parity.

## What the mockup does that a download can't keep

| Mockup behavior | Problem in a downloaded file | Plan |
|---|---|---|
| Google Fonts (IBM Plex) | Network request; offline it falls back silently, under a privacy/firewall policy it may hang | System font stack. Optionally inline a Latin-subset Plex woff2 (~60 KB) behind a flag |
| Hash-routed views (`#front`, `#drivers`) with `hidden` sections | Only one view prints; deep sections are hidden from search and from readers without JS | One long scrolling page; the rail links to `#sec-1`…`#sec-7` anchors |
| Output switcher shows one output at a time | Prints only the selected output | Switcher is a screen-only control; `@media print` and `beforeprint` render every output stacked |
| Synthetic data generated in JS | — | Real data embedded as JSON by Python (below) |
| "Design note" boxes, notes toggle | Not report content | Dropped |

## Architecture

All three report builders (`gui/grid_search_core.py`, `gui/pages/02_Multi_Fidelity_Surrogate_Models.py`,
`report/auto_detect_report.py`) currently copy the same render code and call
`Template(text).render(...)` on `report/template.html`. The plan puts one pipeline behind them:

```
caller-specific inputs ──► build_report_data(...) ──► ReportData ──► render_report(data, style)
                                                       │                 │
                                                       ├─ verdict.py     ├─ "v2": report_v2.html.j2 + inlined css/js + JSON blob
                                                       └─ to_json()      └─ "classic": template.html (unchanged)
```

New modules in `multioutreg/report/`:

- **`data.py`**: `ReportData` dataclasses: run metadata (model, dataset name, data hash,
  seed, package version, date, CV scheme), per-output arrays (`y`, `pred`, `std`,
  residuals), metrics, uncertainty curves (nominal levels → Gaussian and conformal
  coverage), baseline rows, level rows, SHAP shares, PDP grids, CV results with fold
  spread, skipped steps with reasons, and legacy PNG figures. `to_json()` converts numpy
  to lists, NaN/inf to `null`, rounds to 4 significant figures and downsamples scatter
  data above a cap (default 2,000 points per output) so file size stays bounded.
- **`verdict.py`**: thresholds (`r2_good=0.90`, `r2_warn=0.80`, `mf_gain=-5%`, `mf_hurt=+5%`,
  coverage judged against the 95% target ± 1.96·√(p(1−p)/n_test)), `judge(output)` →
  per-criterion status, overall verdict sentence, structured findings
  (`claim, status, section, plot_key`) and next steps. `_summary_points` moves here from
  page 02 so every caller gets findings.
- **`build.py`**: `build_report_data(...)` takes the arguments the callers already pass
  (`metrics`, `uncertainty_metrics`, `y_test`, `best_pred`, `best_std`,
  `calibration_residuals`, `level_summary`, `baseline_comparison`, `grid_search_results`, …)
  and computes the derived series (coverage curves, interval widths as % of range,
  residual trend slope and its p-value, SHAP shares, PDP grids with 10th–90th ICE bands).
- **`render.py`**: `render_report(data, style="v2") -> str`. Uses a Jinja `Environment`
  with **`autoescape=True`** (today's `Template()` doesn't escape user-supplied
  `description` / `notes`). Reads `report_v2.css` and `report_v2.js` from package data and
  inlines them, so they stay normal editable files but the output is a single file.
- **`templates/report_v2.html.j2`, `static/report_v2.css`, `static/report_v2.js`**: added
  to `[tool.setuptools.package-data]` in `pyproject.toml`.

The JSON goes into `<script type="application/json" id="report-data">`. It must escape
`<`, `>` and `&` as `<`, `>`, `&` so a column name containing `</script>`
can't break out. Add a test for that.

### What renders where

- **Jinja (server-side HTML, works without JS):** run header, verdict, scorecard,
  finding-card text, next steps, level / baseline / CV / threshold tables, skipped and
  warning lines, appendix.
- **JS from the JSON blob (inline SVG):** parity, residuals with trend annotation,
  residual histogram, ordered intervals, Gaussian-vs-conformal calibration with noise band,
  coverage at 95%, interval width, MF-vs-HF dumbbell, level r-vs-R² quadrant, SHAP heatmap,
  PDP grid, CV dot plot. Port the drawing helpers (`frame`, `ticks`, `dot`, …) from the
  mockup as plain ES2017 with no dependencies (~25–30 KB unminified).
- **Kept as base64 PNG:** figures with no JS equivalent yet (UMAP sampling, PCA variance,
  SHAP beeswarm, DOE plots), in section 6 or the appendix.

## Data gaps to close

| Needed by | Gap | Fix |
|---|---|---|
| Section 7 CV dot plot | Fold spread is computed then discarded (`fold_mse` in `grid_search_core.py:494`, `fold_sse` in page 02 `:915`) | Return `CV RMSE std` alongside `CV RMSE` in each grid-search row |
| Section 5 heatmap | SHAP is only rendered as one PNG per output | Return mean \|SHAP\| per feature × output from `figures/shap_multioutput.py` |
| Section 5 PDP grid | PDPs are only rendered as PNGs | Return grid values and ICE percentiles from `figures/pdp_plots.py` |
| Run header | No data hash or package version | `hashlib.sha1` of the training frame (first 7 chars); `importlib.metadata.version("multioutreg")` |
| "Skipped" lines | Optional steps return an error string shown in red | `safe_plot_b64` callers record `{section, reason}` in `data.skipped` instead |
| Section 4 | Only page 02 has level and baseline rows | Section hidden for single-fidelity callers (as today) |

## Download and viewing

- **GUI:** keep every existing `st.download_button` and file name (`model_report.html`,
  `mf_model_report.html`, `model_report_auto.html`). Add a "Report format" control to each
  page's report options: *Redesigned (default)* / *Classic*. Optionally add an inline
  preview with `st.components.v1.html(html, height=900, scrolling=True)` under the button.
- **Extra download:** "Download report data (.json)", the same blob, so users can archive
  results or rebuild the report later without retraining.
- **CLI:** `--report-style {v2,classic}` on `mor_cli/grid_search.py` and
  `mor_cli/grid_search_auto_detect.py`; `MOR_REPORT_STYLE` env var as the default.
  `MOR_TEMPLATE_PATH` keeps overriding the classic template.
- **Inside the report:** a "Print / Save as PDF" button (`window.print()`), a light/dark
  toggle (follows the system by default; print always light), and the sticky section rail
  (hidden under 960px and in print).
- **Print CSS:** hide rail, toggles and switchers; `details[open]` forced via a
  `beforeprint` handler; `break-inside: avoid` on panels and figures; all outputs rendered.

## Phases (each one is shippable)

1. **Data model and verdict (Python only).** `data.py`, `verdict.py`, `build.py` with unit
   tests on a synthetic multi-fidelity fixture. Move `_summary_points`. Add CV fold std.
   No user-visible change.
2. **Static v2 template.** Jinja-only page: header, verdict, scorecard, findings, next
   steps, tables, skipped lines, existing PNGs placed in the new section order, appendix,
   print CSS. Behind `style="v2"`, off by default. Already an improvement on its own and
   fully downloadable.
3. **Charts.** `report_v2.js` renders the SVG figures from the JSON blob; add the SHAP
   share and PDP grid data. Replace the matching PNGs in v2.
4. **Wire up callers.** All three `generate_html_report` functions become thin adapters
   (signatures unchanged, so the CLI and existing tests keep working) that call
   `build_report_data` + `render_report`. Add the GUI format control, JSON download and
   CLI flag. Make v2 the default once the parity checklist below passes.
5. **Docs and examples.** Regenerate `docs/example_reports/` in both styles, add a Sphinx
   page explaining how to read the report and the verdict thresholds, update
   `docs/multi_fidelity_gui.md`.

## Tests

- `verdict.py`: every threshold boundary; coverage band shrinks with `n_test`; an output
  gets the worst status of its criteria.
- `render.py`:
  - output contains no `http://` / `https://` in any `src`, `href` or `@import`
    (self-contained check);
  - `#report-data` parses as JSON and round-trips `ReportData`;
  - a column named `</script><script>alert(1)</script>` is escaped in HTML and JSON;
  - size under 2 MB for a 6-output, 500-point fixture;
  - classic style output is byte-identical to today's for the same inputs.
- Existing `tests/unit/gui/pages/test_*` and `tests/unit/report/*` keep passing (adapters
  keep the signatures).
- Optional, marked `slow`: headless Chromium (Playwright) opens the file from `file://`,
  checks for zero console errors, and prints it to PDF.

Run with the usual workaround: `OMP_NUM_THREADS=1` and `import lightgbm` first.

## Parity checklist before v2 becomes the default

- [x] Every section and table in `template.html` has a home in v2 (main body or appendix)
- [x] Imputation summary, regret metrics and PCA sections are present when their inputs are
- [ ] Opens offline from disk in Chrome, Firefox and Safari with no console errors.
  *Chrome (headless) checked: every chart slot draws, no errors, on the synthetic
  fixture and all three examples. Firefox and Safari not yet checked.*
- [x] Readable with JS disabled (verdict, scorecard, tables are server-rendered; a
  `<noscript>` line says charts need JavaScript)
- [x] Prints to PDF with every output and the appendix shown (Chrome; 15 pages for the
  3-output fixture)
- [x] File size within budget on the largest example dataset (155–200 KB for the
  examples; a 6-output, 500-point fixture is under 2 MB in the tests)

## Open questions

1. Inline the Plex font (+60 KB) or use system fonts only? Recommendation: system fonts.
2. Should the verdict thresholds be user-editable in the GUI, or fixed and printed in the
   appendix? Recommendation: fixed for now, passed as a `thresholds=` argument so they can
   be exposed later.
3. Remove the classic template after one release, or keep it indefinitely?
   Recommendation: keep it for one release, then delete it along with the unused
   `generate_report*.py` / `report_template.html` scripts.
