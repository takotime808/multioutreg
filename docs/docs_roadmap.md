# Improving the Documentation

A checklist for making these Sphinx docs easier to land on, navigate and learn
from. Items are ordered by impact. See [Improving the GUI](gui_ux_roadmap.md)
for the Streamlit app.

*Last reviewed 2026-10-28.*

## Status at a Glance

| # | Item | Status |
|---|---|---|
| 1 | Landing page instead of the README | Done |
| 2 | A real Quickstart page | Done |
| 3 | Separate user docs from internal planning | Done |
| 4 | Clear toctree captions | Done |
| 5 | A real multi-objective metrics page | Open |
| 6 | Example reports gallery | Open |
| 7 | Trim the API reference | Open |
| 8 | Consistent tutorials | Open |
| 9 | Furo theme options | Open |
| 10 | Proper logo | Open |
| 11 | `sphinx_design` and `intersphinx` | Open |
| 12 | Repo-relative `autoapi_dirs` | Done |
| 13 | Clean up `conf.py` | Open |

## High Impact

- [x] **1. Landing page instead of the README.** The home page included the
  whole README: a placeholder line ("Notes about project."), GitHub workflow
  badges, and links to `docs/...` and `examples/...` that break once rendered
  (`_patch_readme_for_docs` only rewrote image paths). Replace it with a short
  pitch, the measured savings, and links to Install, Quickstart, Tutorials,
  API and the hosted app.
- [x] **2. A real Quickstart page.** `usage.md` was six lines saying "run
  `multioutreg`". The useful material lived in the README. Replace it with a
  Quickstart covering the Python API, the CLI (every command, with an example)
  and the GUI, each runnable against the bundled example datasets.
- [x] **3. Separate user docs from internal planning.** "Resources" mixed
  reference pages with roadmaps, plans and measurement reports that are
  dated, cite PRs and branches, and describe unbuilt work. Regroup the
  sidebar:

  | Section | Pages |
  |---|---|
  | Getting Started | Install, Quickstart |
  | Tutorials | The notebook tutorials |
  | Guides | Multi-fidelity GUI, Chronos forecasting, using the exported model |
  | Why multioutreg | Cost savings, active learning savings |
  | Reference | API, integrated surrogates, hyperparameters, multi-objective metrics, references |
  | Development | Roadmaps, plans, pair-plot study, audit checklist, this page |

- [x] **4. Clear toctree captions.** "Installation and Usage" held the cost
  savings and multi-fidelity GUI pages, and the API sat under a second
  caption also called "Usage". Rename by the grouping above.

## Medium Impact

- [ ] **5. A real multi-objective metrics page.** `multiobjective_metrics.md`
  only includes `multioutreg/metrics/README.md`. Give it a short example and
  the hypervolume, scalarized and ε-indicator definitions in math (MathJax is
  already enabled).
- [ ] **6. Example reports gallery.** `docs/example_reports/*.html` and
  `report_redesign/report_redesign.html` are not linked from any page. Add a
  page with a screenshot and link per report, since reports are the main
  output of the CLI and GUI.
- [ ] **7. Trim the API reference.** autoapi documents `private-members`,
  `special-members` and `undoc-members` across ~140 modules. Drop those
  options and add a hand-written "Core API" page above the generated tree
  listing the key classes (`AutoDetectMultiOutputRegressor`, `ActiveLearner`,
  `MultiFidelityActiveLearner`, the conformal predictors, the surrogates).
  Add a generated CLI reference (`sphinx-click` is installed in CI but not
  enabled; the commands use `typer`, so check it works or use
  `typer <module> utils docs`).
- [ ] **8. Consistent tutorials.** `forecasting_with_chronos.md` is commands
  with no output or explanation. Open every tutorial with what it covers, the
  extras it needs and roughly how long it takes to run (the notebooks already
  start with "What this tutorial covers"). Add a Binder or Colab badge via
  `nbsphinx_prolog` (a draft is commented out in `conf.py`).

## Polish

- [ ] **9. Furo theme options.** Replace the announcement banner ("Live and
  uncut...") with something useful, such as a link to the hosted app or the
  latest release. Set `html_title`, `source_repository`, `source_branch` and
  `source_directory` for "Edit this page" links, and add a GitHub footer icon.
- [ ] **10. Proper logo.** `html_logo` is a favicon (`favicon.ico.png`).
  Use a wordmark or SVG sized for the sidebar.
- [ ] **11. `sphinx_design` and `intersphinx`.** Cards and grids for the
  landing page, tabs for the Quickstart (Python / CLI / GUI), and
  `sphinx.ext.intersphinx` so `numpy`, `scikit-learn` and `pandas` types link
  out. Add the packages to the docs install step in `gh_pages.yml`.
- [x] **12. Repo-relative `autoapi_dirs`.** It was
  `../../multioutreg/multioutreg`, which only resolved when the checkout
  folder was named `multioutreg` (and failed in worktrees). Now `../multioutreg`.
- [ ] **13. Clean up `conf.py`.** About half of it is commented-out template
  code (LaTeX, EPUB, nbsphinx prolog, alternative themes). The CI step also
  installs packages the build does not use (`sphinx_rtd_theme`,
  `sphinx-gallery`, `sphinxcontrib-bibtex`, `jupyter-black`).
