
"""Configuration file for the Repo Template project."""

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('../../repo-template/multioutreg'))
# sys.path.insert(0, os.path.abspath('../multioutreg'))


# -- Project information -----------------------------------------------------
master_doc = 'index' # the master toctree document

project = 'Repo Remplate -- nbphinx'
copyright = '2021, TT'
author = 'TT'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinxcontrib.mermaid",
    "autoapi.extension",
    "myst_parser",
    # 
    # 'sphinxcontrib.bibtex',        # for bibliographic references
    "sphinx_copybutton",           # for adding “copy to clipboard” buttons to all text/code boxes | commented due to multiple scrollbar issue https://github.com/cameronraysmith/nbsphinx-template/issues/1
    # 'sphinxcontrib.rsvgconverter', # for SVG->PDF conversion in LaTeX output
    # 'sphinx_last_updated_by_git',  # get "last updated" from Git
]


# -- Auto Stuff ---------------------------------------------------

autodoc_default_options = {
    "show-inheritance": True,
    "imported-members": True,
    "inherited-members": True,
    "no-special-members": True,
}

add_module_names = False # rm namespaces from class/method signatures
autosummary_generate = True # Needs sphinx.ext.autosummary
autosummary_imported_members = True
autoapi_type = "python"

# Uncomment this if debugging autoapi
autoapi_add_toctree_entry = False
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "imported-members",
    "show-inheritance",
    "special-members",
    "show-module-summary",
]

autoapi_dirs = ["../../repo-template/multioutreg"]
# autoapi_dirs = ["../multioutreg"]

# Used to avoid error for too many levels on relative imports
# NOTE: to accomplish this, its better to use autoapi_ignore than exclude_patterns
# autoapi_ignore = ["**/module_name/*.py"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "build",
    "_build",
    "*checkpoint*",
    ".DS_Store",
    "Thumbs.db",
    "*_templates*", # NOTE: DO I WANT THIS HERE???????????????????????????????????????
    ".ipynb_checkpoints",
    "**.ipynb_checkpoints",
]

# source_suffix = [".rst", ".md", ".ipynb"]
source_suffix = {'.rst': 'restructuredtext', '.md': 'restructuredtext', '.ipynb': 'restructuredtext'}

source_suffix = [".rst", ".md", ".ipynb"]
# source_suffix = {'.rst': 'restructuredtext', '.md': ['jupytext.reads', {'fmt': 'md'}], '.ipynb': 'restructuredtext'}
# Support for notebook formats other than .ipynb
# nbsphinx_custom_formats = {
#     '.pct.py': ['jupytext.reads', {'fmt': 'py:percent'}],
#     # '.md': ['jupytext.reads', {'fmt': 'md'}],
# }


# autosummary_generate = True # already defined above
# autosummary_imported_members = True # already defined above
# master_doc = "index" # already defined above
# add_module_names = False # already defined above

# NOTE: autodoc_default_options above has these and:
#     - "show-inheritance": True,
#     - "no-special-members"
autodoc_default_flags = ["members", "inherited-members", "imported-members"]

autoclass_content = "both" # adds __init__ doc (parms, etc) to class summaries

# TODO: test the differences between T/F
html_show_sourcelink = (
    # False # rm 'view source code' form top of page (for html, not python)
    True
)

show_inheritance_diagram = True
add_function_parentheses = False
toc_object_entries_show_parents = "hide"

# -- Napoleon Settings ---------------------------------------------------
napoleon_google_docstring = True
napoleon_use_param = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_example = True
napoleon_use_admonition_for_notes = True

# -- Markdown File Options ---------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

# -- HTML Output Options ---------------------------------------------------
# TODO: test options below
# html_theme = "sphinx_rtd_theme" # NOTE: use 'logo_only' option in html_theme_options, when using this theme
html_theme = "furo"
# html_theme = "alabaster"
html_logo = "_static/logos/favicon.ico.png"
# html_favicon = "_static/logos/favicon.ico" # TODO: test ico is working, if not then ue png
html_favicon = "_static/logos/favicon.png" # TODO: test ico is working, if not then ue png
html_theme_options = {
    "announcement": "Live and uncut: <em>TT's Repo Template</em>",
    # "logo_only": True, # only for sphinx_rtd_theme
    # "display_version": True, # this line makes favicon fail to load...why? idk
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static'] # already defined above
html_static_path = ["_static"] # already defined above
html_static_path = [] # TODO: test these options












# # This is processed by Jinja2 and inserted before each notebook
# nbsphinx_prolog = r"""
# {% set docname = 'source/' + env.doc2path(env.docname, base=None) %}

# .. raw:: html

#     <div class="admonition note">
#       This page was generated from
#       <a class="reference external" href="https://github.com/cameronraysmith/nbsphinx-template/blob/master/{{ docname|e }}">{{ docname|e }}</a>.
#       <span style="white-space: nowrap;"><a href="https://mybinder.org/v2/gh/cameronraysmith/nbsphinx-template/master?filepath={{ docname|e }}"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a> or </span>
#       <script>
#         if (document.location.host) {
#           $(document.currentScript).replaceWith(
#             '<a class="reference external" ' +
#             'href="https://nbviewer.jupyter.org/url' +
#             (window.location.protocol == 'https:' ? 's/' : '/') +
#             window.location.host +
#             window.location.pathname.slice(0, -4) +
#             'ipynb"><em>nbviewer</em></a>.'
#           );
#         }
#       </script>
#     </div>

# .. raw:: latex

#     \nbsphinxstartnotebook{\scriptsize\noindent\strut
#     \textcolor{gray}{The following section was generated from
#     \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
# """

# # This is processed by Jinja2 and inserted after each notebook
# nbsphinx_epilog = r"""
# {% set docname = 'source/' + env.doc2path(env.docname, base=None) %}
# .. raw:: latex

#     \nbsphinxstopnotebook{\scriptsize\noindent\strut
#     \textcolor{gray}{\dotfill\ \sphinxcode{\sphinxupquote{\strut
#     {{ docname | escape_latex }}}} ends here.}}
# """



# mathjax3_config = {
#     'tex': {'tags': 'ams', 'useLabelIds': True},
# }

# bibtex_bibfiles = ['references.bib']

# # Support for notebook formats other than .ipynb
# nbsphinx_custom_formats = {
#     '.pct.py': ['jupytext.reads', {'fmt': 'py:percent'}],
#     '.md': ['jupytext.reads', {'fmt': 'md'}],
# }

# # -- Options for HTML output -------------------------------------------------

# # The theme to use for HTML and HTML Help pages.  See the documentation for
# # a list of builtin themes.
# #
# # html_theme = 'alabaster'

# # Add any paths that contain custom static files (such as style sheets) here,
# # relative to this directory. They are copied after the builtin static files,
# # so a file named "default.css" will overwrite the builtin "default.css".
# # html_static_path = ['_static'] # already defined above

# # -- Get version information and date from Git ----------------------------

# try:
#     from subprocess import check_output
#     release = check_output(['git', 'describe', '--tags', '--always'])
#     release = release.decode().strip()
#     today = check_output(['git', 'show', '-s', '--format=%ad', '--date=short'])
#     today = today.decode().strip()
# except Exception:
#     release = ''
#     today = ''

# # -- Options for HTML output ----------------------------------------------

# # html_favicon = 'favicon.svg'
# # html_title = project + ' version ' + release
# html_title = project

# # -- Options for LaTeX output ---------------------------------------------

# # See https://www.sphinx-doc.org/en/master/latex.html
# latex_elements = {
#     'papersize': 'a4paper',
#     'printindex': '',
#     'sphinxsetup': r"""
#         %verbatimwithframe=false,
#         %verbatimwrapslines=false,
#         %verbatimhintsturnover=false,
#         VerbatimColor={HTML}{F5F5F5},
#         VerbatimBorderColor={HTML}{E0E0E0},
#         noteBorderColor={HTML}{E0E0E0},
#         noteborder=1.5pt,
#         warningBorderColor={HTML}{E0E0E0},
#         warningborder=1.5pt,
#         warningBgColor={HTML}{FBFBFB},
#     """,
#     'preamble': r"""
# \usepackage[sc,osf]{mathpazo}
# \linespread{1.05}  % see http://www.tug.dk/FontCatalogue/urwpalladio/
# \renewcommand{\sfdefault}{pplj}  % Palatino instead of sans serif
# \IfFileExists{zlmtt.sty}{
#     \usepackage[light,scaled=1.05]{zlmtt}  % light typewriter font from lmodern
# }{
#     \renewcommand{\ttdefault}{lmtt}  % typewriter font from lmodern
# }
# \usepackage{booktabs}  % for Pandas dataframes
# """,
# }

# latex_documents = [
#     (master_doc, 'nbsphinx.tex', project, author, 'howto'),
# ]

# latex_show_urls = 'footnote'
# latex_show_pagerefs = True

# # -- Options for EPUB output ----------------------------------------------

# # These are just defined to avoid Sphinx warnings related to EPUB:
# version = release
# suppress_warnings = ['epub.unknown_project_files']

# # -- Set default HTML theme (if none was given above) ---------------------
# # html_theme = 'sphinx_book_theme'

# if 'html_theme' not in globals():
#     try:
#         import insipid_sphinx_theme
#     except ImportError:
#         pass
#     else:
#         html_theme = 'insipid'
#         html_copy_source = False
#         html_permalinks_icon = '\N{SECTION SIGN}'