# Copyright (c) 2025 takotime808
"""Time-series forecasting: foundation-model adapters, surrogate forecasters and CV.

Exports are resolved lazily so importing the package does not pull in
torch, chronos or matplotlib until one of the names below is used.
"""

from importlib import import_module

# name -> submodule it lives in (None: the name is itself a submodule)
_EXPORTS = {
    "metrics": None,
    "ChronosForecaster": "chronos_adapter",
    "ForecastResult": "chronos_adapter",
    "load_financial_csv": "financial",
    "forecast_with_chronos": "financial",
    "check_ts_suitability": "ts_suitability",
    "make_lag_features": "lag_features",
    "rolling_window_features": "lag_features",
    "LagFeatureForecaster": "lag_forecaster",
    "AutoSurrogateForecaster": "surrogate_forecaster",
    "WalkForwardCV": "cv",
    "walk_forward_splits": "cv",
    "TimeSeriesSplitWrapper": "cv",
    "TSFoldResult": "cv",
    "gaussian_quantiles": "uncertainty",
    "conformal_interval_from_residuals": "uncertainty",
    "propagate_uncertainty_recursive": "uncertainty",
    "plot_forecast_result": "figures",
    "ProphetForecaster": "prophet_adapter",
    "NeuralForecaster": "neuralforecast_adapter",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    submodule = _EXPORTS[name]
    if submodule is None:
        value = import_module(f"{__name__}.{name}")
    else:
        value = getattr(import_module(f"{__name__}.{submodule}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
