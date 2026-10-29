# Copyright (c) 2025 takotime808
"""Sphinx configuration for the multioutreg docs.

See https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import sys

# Repo root, so `multioutreg` and `mor_cli` import regardless of the checkout folder name.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -------------------------------------------------------
project = "multioutreg"
copyright = "2025, takotime808"
author = "takotime808"
master_doc = "index"

# -- Extensions ----------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "autoapi.extension",
    "myst_parser",
    "nbsphinx",
    "sphinx_click",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns = [
    "build",
    "_build",
    "**.ipynb_checkpoints",
    ".DS_Store",
    "_templates",
]

# -- API reference (sphinx-autoapi) --------------------------------------------
# Public API only: private helpers and dunder methods made the ~140-module tree hard
# to browse. "undoc-members" stays: without it autoapi drops every module and package
# that has no docstring, including most __init__.py files. The hand-written core_api
# page sits above the tree.
autoapi_type = "python"
autoapi_dirs = ["../multioutreg"]
autoapi_add_toctree_entry = False
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_ignore = ["*/gui/pages/*", "*/gui/Grid_Search_Surrogate_Models.py"]  # Streamlit scripts

autodoc_default_options = {"show-inheritance": True}
autoclass_content = "both"  # class docstring + __init__ parameters
add_module_names = False
add_function_parentheses = False
toc_object_entries_show_parents = "hide"

napoleon_google_docstring = True
napoleon_use_param = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
intersphinx_timeout = 10

# -- Markdown and notebooks ----------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3
myst_fence_as_directive = ["mermaid"]  # ```mermaid fences render as diagrams

nbsphinx_execute = "never"  # notebooks are committed with their outputs

_REPO = "takotime808/multioutreg"
_BRANCH = "main"

# What each tutorial needs and roughly how long it runs (measured on a laptop).
TUTORIALS = {
    "tutorials/multi_output_surrogates": ("shap (core install)", "about 2 minutes"),
    "tutorials/joint_scores": ("core install (gpytorch for the multi-task GP)", "under a minute"),
    "tutorials/multi_fidelity_surrogates": ("core install", "under a minute"),
    "tutorials/active_learning_benchmarks": ("core install", "a few minutes"),
    "tutorials/cost_savings": ("core install", "about 10 minutes"),
    "tutorials/imputation_methods": ("the lightgbm extra", "under a minute"),
    "tutorials/active_learning_missing_inputs": ("the lightgbm extra", "up to 3 minutes"),
    "tutorials/using_the_exported_model": ("the lightgbm extra", "up to 3 minutes"),
}

# Rendered at the top of every notebook page: extras, run time, and open-in links.
nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) | string %}
{% set info = env.config.html_context.tutorials.get(env.docname) %}
.. raw:: html

    <div class="admonition note tutorial-meta">
      <p class="admonition-title">About this tutorial</p>
      <p>
      {% if info %}<b>Needs:</b> {{ info[0] }} &nbsp;·&nbsp; <b>Runs in:</b> {{ info[1] }} &nbsp;·&nbsp; {% endif %}
      <a href="https://colab.research.google.com/github/""" + _REPO + "/blob/" + _BRANCH + r"""/docs/{{ docname }}">Open in Colab</a>
      &nbsp;·&nbsp;
      <a href="https://github.com/""" + _REPO + "/blob/" + _BRANCH + r"""/docs/{{ docname }}">View on GitHub</a>
      </p>
    </div>
"""

# -- HTML output (Furo) --------------------------------------------------------
html_theme = "furo"
html_title = "multioutreg"
html_favicon = "_static/logos/favicon.png"
html_static_path = ["_static"]
html_extra_path = ["_extra"]  # example reports, served as-is at /example_reports/
html_css_files = ["css/active-learning-savings.css", "css/site.css"]
html_show_sphinx = False
html_show_sourcelink = True
html_context = {"tutorials": TUTORIALS}

html_theme_options = {
    "light_logo": "logos/wordmark-light.svg",
    "dark_logo": "logos/wordmark-dark.svg",
    "sidebar_hide_name": True,
    "announcement": (
        'Try it without installing: the <a href="https://multioutreg.streamlit.app/">'
        "hosted multioutreg app</a>."
    ),
    "source_repository": f"https://github.com/{_REPO}/",
    "source_branch": _BRANCH,
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": f"https://github.com/{_REPO}",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 '
                "3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53"
                "-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 "
                "1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 "
                "0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 "
                "2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 "
                "1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 "
                '2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}
