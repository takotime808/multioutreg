.. Test nbsphinx documentation master file, created by
   sphinx-quickstart on Wed May 26 22:38:56 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. Rather than using: `.. include:: README.rst` and 
   inside that file having `.. include:: ../README.md` 
   just copy file and add to toctree, to preserve formatting.

Notes about project.

.. include:: _README_docs.md
   :parser: myst_parser.sphinx_


.. toctree::
   :caption: Installation and Usage

   install
   usage
   cost_savings
   active_learning_savings
   multi_fidelity_gui

.. toctree::
   :maxdepth: 3
   :caption: Tutorials

   tutorials/multi_output_surrogates
   tutorials/forecasting_with_chronos
   tutorials/multi_fidelity_surrogates
   tutorials/active_learning_benchmarks
   tutorials/cost_savings

.. toctree::
   :maxdepth: 2
   :caption: Resources

   integrated_surrogates
   surrogate_hyperparameters
   surrogate_roadmap
   gui_ux_roadmap
   active_learning
   multiobjective_metrics
   parallel_computing
   references

.. toctree::
   :maxdepth: 3
   :caption: Usage

   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`