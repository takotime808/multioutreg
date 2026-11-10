# Reading the Report

Every run of the GUI pages and of the `grid_search` / `grid_search_auto_detect`
commands produces an HTML report. The report leads with the answer, **can I use this
model, and for which outputs?**, and then gives the evidence for it, one section per
question. This page explains each section and the rules behind the verdict. See
{doc}`example_reports` for finished reports to open alongside it.

The report is **one self-contained `.html` file**: no network requests and no sidecar
files, so it opens from disk, from an email attachment or from a shared drive. The
verdict, scorecard, findings and every table work without JavaScript; the charts need it.

## The seven questions

| Section | Question | What you see |
|---|---|---|
| 1 | **Can I use it?** | Verdict, a scorecard of outputs × criteria, key findings each with its chart, and what to do next |
| 2 | **How accurate is each output?** | Per output: R², RMSE and MAE with their references, and four charts (predicted vs observed, residuals, residual distribution, ordered intervals). Multi-objective regret if it was computed |
| 3 | **Can I trust the error bars?** | Calibration curves for the Gaussian and conformal intervals on the same axes, coverage at 95% and the interval widths |
| 4 | **Did the low-fidelity data help?** | Multi-fidelity runs only: RMSE against an HF-only baseline, and how informative each lower level is |
| 5 | **What drives the outputs?** | A SHAP heatmap (share of mean \|SHAP\| per input and output) and partial dependence for the top three inputs |
| 6 | **Is the data adequate?** | Test-set size, the input design with the inferred sampling method, imputation and PCA |
| 7 | **How was the model chosen?** | The cross-validation score of every candidate with its fold spread, so ties are visible |
| A | Appendix (collapsed) | Verdict thresholds, the full metrics table, run settings and provenance |

Each chart heading says what it shows, and the line under it says what to look for or,
when the report found something, what it found ("9 of 50 points fall outside their 95%
interval, about 2 expected"). An optional step that did not run shows a grey
**Skipped** line with the reason, never a red error.

Each output keeps one color in every chart. In section 2 the buttons switch between
outputs; **Show all** stacks them, and printing always shows all of them.

## How the verdict is decided

Each output is judged on up to three criteria. Its overall status is the worst of them,
and the verdict sentence names the outputs in each group: **Ready**, **Use with care**
or **Not ready**.

| Criterion | Pass | Caution | Fail |
|---|---|---|---|
| Accuracy: test R² | ≥ 0.90 | 0.80 – 0.90 | < 0.80 |
| Error bars: coverage of the 95% interval | ≥ 95% − *b* | Gaussian fails but conformal passes, or ≥ 95% − 2*b* | below 95% − 2*b*, or no std |
| Low-fidelity data: RMSE vs HF-only | ≤ −5% (helps) | ≥ +5% (hurts) | — |

*b* is the coverage you can lose to test-set noise alone, 1.96 binomial standard
errors:

```{math}
b = 1.96 \sqrt{\frac{0.95 \times 0.05}{n_\text{test}}}
```

With 50 test points *b* ≈ 6 percentage points, so a coverage of 89% still passes; with
500 it is 2 points. Coverage above 95% passes (wide intervals are safe), and its cost is
shown separately as the interval width. A change within ±5% of the HF-only model is
**No gain**, which does not lower the overall status. A lower fidelity level is flagged
as weak when a model trained on it alone has |r| < 0.5 with the high-fidelity truth.

The thresholds are printed in every report's appendix. In Python they are the fields of
{class}`multioutreg.report.verdict.Thresholds`.

## Choosing the format

The redesigned report is the default. The previous layout ("classic") is still available
while the new one is reviewed:

::::{tab-set}

:::{tab-item} GUI
Pick **Report format: Redesigned / Classic / Both** next to **Run Grid Search**.
**Both** builds the two reports from the same run (no retraining) and offers a download
for each (`model_report_redesigned.html`, `model_report_classic.html`), so you can compare them.
:::

:::{tab-item} CLI
```bash
multioutreg grid_search data.csv x1,x2 y1,y2 --report-style classic
```
:::

:::{tab-item} Environment
```bash
export MOR_REPORT_STYLE=classic   # default for every caller
```
Setting `MOR_TEMPLATE_PATH` to a custom Jinja template always renders the classic
report with that template.
:::

:::{tab-item} Python
```python
from multioutreg.report.auto_detect_report import generate_html_report

html = generate_html_report(..., report_style="v2")
```
:::

::::

## The report data

The GUI also offers **Download report data (.json)**: every number behind the charts and
the verdict. Keep it with the report to archive a run, or rebuild the report later
without retraining:

```python
from multioutreg.report.data import ReportData
from multioutreg.report.render import render_report, report_json_from_html

html = open("mf_model_report.html").read()
data = ReportData.from_json(report_json_from_html(html))   # or open("..._data.json").read()
data.outputs[0]["r2"]                                      # the numbers, as plain Python
open("rebuilt.html", "w").write(render_report(data))
```

Scatter data is capped at 2,000 points per output (statistics use every point) and
rounded to 4 significant figures, so a typical report is 100–300 KB.

## Printing and sharing

Use **Print / Save as PDF** at the top of the report. Printing switches to the light
theme, shows every output and opens the appendix. The **Theme** button cycles between
following the system, light and dark.
