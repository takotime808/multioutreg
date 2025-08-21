"""Time series utilities for `multioutreg`.

This submodule provides functionality for simple time-series forecasting
based on the approach outlined in "Programmable forecasting in the age of
large language models" (arXiv:2403.07815).
"""

from .forecasting import TimeSeriesForecaster

__all__ = ["TimeSeriesForecaster"]
