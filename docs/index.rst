.. Test nbsphinx documentation master file, created by
   sphinx-quickstart on Wed May 26 22:38:56 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. Rather than using: `.. include:: README.rst` and 
   inside that file having `.. include:: ../README.md` 
   just copy file and add to toctree, to preserve formatting.

.. include:: ../README.md
    :parser: myst_parser.sphinx_

Notes about project.

.. toctree::
   :caption: Installation and Usage

   install
   usage

.. toctree::
   :maxdepth: 3
   :caption: Usage

   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`