# Example Reports

Reports are the main output of the CLI and the GUI. The ones below come from real runs
(regenerate them with `python examples/make_example_reports.py`). Each opens in a new
tab as a standalone file; the redesigned version has a verdict and scorecard on top, and
{doc}`reading_the_report` explains every section.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Multi-fidelity surrogate
:img-top: _static/example_reports/multi_fidelity_v2.png

The Multi-Fidelity page's pipeline on two fidelity levels (Park and Currin benchmark pairs,
plus an output whose low level is pure noise). Shows the low-fidelity section: a gain for
some outputs and a level flagged as uninformative.
+++
<a href="example_reports/multi_fidelity_v2.html">Redesigned</a> ·
<a href="example_reports/multi_fidelity_classic.html">Classic</a>
:::

:::{grid-item-card} Auto-detected surrogate with conformal intervals
:img-top: _static/example_reports/auto_detect_v2.png

`multioutreg grid_search_auto_detect --conformal` on the bundled 3-output dataset: a model
chosen per output, with Gaussian and conformal error bars compared.
+++
<a href="example_reports/auto_detect_v2.html">Redesigned</a> ·
<a href="example_reports/auto_detect_classic.html">Classic</a>
:::

:::{grid-item-card} Per-output grid search
:img-top: _static/example_reports/grid_search_v2.png

`multioutreg grid_search` on the same data: the best configuration per output, with every
candidate's cross-validation score and fold spread.
+++
<a href="example_reports/grid_search_v2.html">Redesigned</a> ·
<a href="example_reports/grid_search_classic.html">Classic</a>
:::

:::{grid-item-card} Design mockups
:img-top: _static/example_reports/report_redesign.png
:link: report_redesign/report_redesign.html
:link-type: url

The proposal behind the redesign: the problems with the classic layout, the ideas, and
four interactive mockup pages on a synthetic wing-section run.
:::

::::

## Older examples

Outputs of the deprecated `generate_report*.py` scripts, kept for reference:
<a href="example_reports/surrogate_report_example.html">surrogate_report_example.html</a>,
<a href="example_reports/umap_2d_sampling_plot.html">umap_2d_sampling_plot.html</a>.
