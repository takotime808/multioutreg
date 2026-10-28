multioutreg
===========

**Multi-output, multi-fidelity surrogate models with uncertainty quantification.**

``multioutreg`` builds fast surrogate models of expensive simulations and
experiments that have several outputs. It picks the best model for each output
by cross-validation, predicts with calibrated uncertainty (including
distribution-free conformal intervals), and writes the results to an HTML
report. Active learning and multi-fidelity sampling choose which runs to do
next, so the surrogate reaches a target accuracy with fewer of them.

Use it from Python, from the ``multioutreg`` command line, or in the
`hosted Streamlit app <https://multioutreg.streamlit.app/>`_ without
installing anything.

Fewer expensive runs
--------------------

Measured on benchmark problems with a known answer, at the same accuracy as
the usual workflow:

- **31x fewer simulator runs** to propagate input uncertainty: 320 runs on a
  space-filling design instead of 10,000-sample Monte Carlo.
- **35x fewer runs** to find an optimal design: Bayesian optimization needs 24
  runs where random search needs 829.
- **3.6x cheaper** surrogate training when a 10x cheaper low-fidelity model is
  available, using multi-fidelity active learning.

The full table, priced in money and compute time, is on
:doc:`cost_savings`.

Where to go next
----------------

:doc:`install`
   Install from source, plus the optional extras for XGBoost, LightGBM,
   Chronos forecasting and more.

:doc:`how_it_works`
   End-to-end flow diagram: data checks, the sample-size-dependent
   evaluation plan, model selection, uncertainty and outputs.

:doc:`quickstart`
   Fit a multi-output surrogate with uncertainty in Python, from the CLI, or
   in the GUI, using the bundled example datasets.

:doc:`tutorials/multi_output_surrogates`
   Start here for the tutorials: model selection, uncertainty, SHAP and
   partial dependence, start to finish.

:doc:`api`
   Every module, class and function.

.. toctree::
   :hidden:
   :caption: Getting Started

   install
   how_it_works
   quickstart

.. toctree::
   :hidden:
   :caption: Tutorials

   tutorials/multi_output_surrogates
   tutorials/multi_fidelity_surrogates
   tutorials/active_learning_benchmarks
   tutorials/cost_savings
   tutorials/imputation_methods
   tutorials/active_learning_missing_inputs

.. toctree::
   :hidden:
   :caption: Guides

   multi_fidelity_gui
   tutorials/using_the_exported_model
   tutorials/forecasting_with_chronos

.. toctree::
   :hidden:
   :caption: Why multioutreg

   cost_savings
   active_learning_savings

.. toctree::
   :hidden:
   :caption: Reference

   api
   integrated_surrogates
   surrogate_hyperparameters
   multiobjective_metrics
   references

.. toctree::
   :hidden:
   :caption: Development

   surrogate_roadmap
   gui_ux_roadmap
   docs_roadmap
   active_learning
   parallel_computing
   pair_plot_report
   AUDIT_CHECKLIST
