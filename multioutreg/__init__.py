# Copyright (c) 2025 takotime808

from importlib import import_module

from multioutreg.model_selection import AutoDetectMultiOutputRegressor
from multioutreg.conformal import SplitConformalPredictor, CVPlusConformalPredictor

# Time-series exports are resolved lazily so `import multioutreg` does not
# pull in chronos or matplotlib.
_TS_EXPORTS = {
    "ChronosForecaster": "ChronosForecaster",
    "ForecastResult": "ForecastResult",
    "check_ts_suitability": "check_ts_suitability",
    "ts_metrics": "metrics",
}

__all__ = [
    "AutoDetectMultiOutputRegressor",
    "SplitConformalPredictor",
    "CVPlusConformalPredictor",
    *_TS_EXPORTS,
]


def __getattr__(name):
    if name not in _TS_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("multioutreg.time_series"), _TS_EXPORTS[name])
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
