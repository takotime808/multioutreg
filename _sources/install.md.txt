# Installation Instructions #

To install this python library:

```bash
git clone https://github.com/takotime808/multioutreg.git && cd multioutreg

pip install .
```

The core install includes Streamlit, Plotly, PyTorch and Chronos, because the
hosted Streamlit app installs core dependencies only. The `gui`, `torch` and
`ts` extras below list the same packages, ready for when they leave core.

## Optional extras ##

Some surrogates and features depend on packages that are not installed by
default. Install them by naming the extra in brackets (quote the argument in
`zsh`, e.g. `pip install ".[xgboost]"`):

| Extra | Installs | Enables |
|-------|----------|---------|
| `gui` | `streamlit`, `plotly`, `pyarrow<26` | The Streamlit app in `multioutreg/gui` |
| `torch` | `torch` | `BNNSurrogate`, `DeepEnsembleSurrogate` |
| `ts` | `chronos-forecasting`, `torch`, `boto3`, `python-dotenv` | Time-series commands: Chronos zero-shot forecasting (`ts-forecast`) and `ts-pipeline` |
| `ngboost` | `ngboost`, `scikit-learn>=1.6` | `NGBoostSurrogate` |
| `xgboost` | `xgboost` | `XGBoostSurrogate` |
| `lightgbm` | `lightgbm` | `LightGBMSurrogate` |
| `catboost` | `catboost` | `CatBoostSurrogate` |
| `gpytorch` | `gpytorch` | `SparseGPSurrogate` |
| `smt` | `smt[gpx]` | `KPLSSurrogate`, `GPXSurrogate` |
| `lstm` | `keras` | LSTM model in `ts-pipeline` |
| `test` | `pytest`, `pytest-cov`, `pytest-sugar`, `coverage-badge` | Running the test suite |
| `examples` | `ray`, `hyperopt`, `plotly`, `nbformat`, `ipykernel` | Running the example notebooks in `examples/` |
| `all` | every extra above **except** `examples` | Every library feature plus the test suite |

```bash
pip install ".[all]"            # every library feature + test tooling
pip install ".[all,examples]"   # ...plus the example-notebook dependencies
uv sync --extra all             # same as the first line, using uv
```

### Why `all` excludes `examples` ###

By convention, an `all` extra means "every optional feature of the package".
It is the union of the extras that unlock functionality in the library's own
code. `ray`, `hyperopt` and `plotly` are imported only by the example
notebooks, never by `multioutreg`. Including them would make `.[all]` larger
and more fragile (`ray` in particular is a heavy install and often lags new
Python releases and platforms) without enabling anything in the package.
Install `examples` explicitly when you want to run the notebooks.
