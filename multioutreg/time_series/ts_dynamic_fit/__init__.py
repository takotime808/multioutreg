# Copyright (c) 2025 takotime808
"""ARIMA / SARIMA / LSTM forecasting pipeline.

Exports are resolved lazily so importing the package does not pull in
statsmodels or torch until one of the names below is used.
"""

from importlib import import_module

_EXPORTS = {
    "ARIMA": "algs.arima",
    "SARIMA": "algs.sarima",
    "LSTM": "algs.lstm",
    "LSTMModel": "algs.lstm",
    "DataProcessor": "data_handling.DataProcessor",
    "Ranker": "src.ranker",
    "ForecastAutoIntervals": "src.ForecastAutoIntervals",
    "visualize_model": "src.visualize",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
