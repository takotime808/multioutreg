# multioutreg #

Multi-Output Multi-Fidelity Surrogate Modeling with Uncertainty Quantification.

This repository provides utilities for evaluating multi-output surrogate models
with uncertainty estimation. Example notebooks in `examples/` demonstrate the
plotting functions and performance metrics. A new script `examples/report.py`
shows how to create an HTML report using a Jinja2 template that collects all
metrics and figures in one document. Outputs reports are shown the docs
directory [example_reports](docs/example_reports/).

----
### Deployments ###

The deployed streamlit application can be found 
at [https://multioutreg-report.streamlit.app/](https://multioutreg-report.streamlit.app/).

Example files that can be used in the deployed app can be 
found [here](./docs/_static/example_datasets/).

----

### Installation & Usage ###

Install with:
```sh
pip install .
```

[Notebooks and scripts](examples/) can be run once the tool is installed.

To launch the Streamlit App:
```sh
streamlit run multioutreg/gui/Grid_Search_Surrogate_Models.py
```

The `AutoDetectMultiOutputRegressor` can now automatically search across all
vendor-provided surrogates. See [`examples/AutoDetectMultiOutputRegressor.ipynb`](./examples/AutoDetectMultiOutputRegressor.ipynb) 
for a short demonstration.

**NOTE:** The CLI is not deployed yet.