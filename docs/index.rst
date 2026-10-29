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

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Install
      :link: install
      :link-type: doc

      From source, plus the optional extras for XGBoost, LightGBM, Chronos
      forecasting and more.

   .. grid-item-card:: Quickstart
      :link: quickstart
      :link-type: doc

      Fit a multi-output surrogate with uncertainty from Python, the CLI or the GUI,
      on the bundled example datasets.

   .. grid-item-card:: How it works
      :link: how_it_works
      :link-type: doc

      Data checks, the sample-size-dependent evaluation plan, model selection,
      uncertainty and outputs, end to end.

   .. grid-item-card:: Reading the report
      :link: reading_the_report
      :link-type: doc

      The verdict, the seven questions the report answers and the rules behind
      them, with :doc:`example reports <example_reports>` to open.

   .. grid-item-card:: Tutorials
      :link: tutorials/multi_output_surrogates
      :link-type: doc

      Model selection, uncertainty, SHAP and partial dependence, multi-fidelity and
      active learning, start to finish.

   .. grid-item-card:: API reference
      :link: core_api
      :link-type: doc

      The core classes first, then every module, and the CLI commands.

.. toctree::
   :hidden:
   :caption: Getting Started

   install
   how_it_works
   quickstart
   reading_the_report
   example_reports

.. toctree::
   :hidden:
   :caption: Tutorials

   tutorials/multi_output_surrogates
   tutorials/joint_scores
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

   core_api
   api
   cli
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
   report_redesign/implementation_plan
   active_learning
   parallel_computing
   pair_plot_report
   AUDIT_CHECKLIST
