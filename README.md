# multioutreg

Multi-Output Multi-Fidelity Surrogate Modeling with Uncertainty Quantification.

This repository provides utilities for evaluating multi-output surrogate models
with uncertainty estimation. Example notebooks in `examples/` demonstrate the
plotting functions and performance metrics. A new script `examples/report.py`
shows how to create an HTML report using a Jinja2 template that collects all
metrics and figures in one document. Outputs reports are shown the docs
directory [example_reports](docs/example_reports/).

The deployed streamlit application can be found 
at [https://multioutreg-report.streamlit.app/](https://multioutreg-report.streamlit.app/).