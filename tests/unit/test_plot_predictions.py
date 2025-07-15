import sys
from pathlib import Path
import importlib.util
import matplotlib
matplotlib.use("Agg")
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

from multioutreg.figures.prediction_plots import (
    plot_predictions,
    # plot_predictions_with_error_bars,
)

# ROOT = Path(__file__).resolve().parents[2]
# spec = importlib.util.spec_from_file_location(
#     "perf",
#     ROOT / "multioutreg" / "figures" / "performance_metric_figures.py",
# )
# perf = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(perf)
# plot_predictions = perf.plot_predictions


def test_plot_predictions_single_output():
    rng = np.random.RandomState(0)
    X = rng.rand(20, 3)
    y = X @ rng.rand(3) + rng.randn(20) * 0.1
    model = LinearRegression().fit(X, y)
    ax = plot_predictions(model, X, y)
    assert ax.get_title()


def test_plot_predictions_multi_output():
    rng = np.random.RandomState(1)
    X = rng.rand(20, 3)
    Y = X @ rng.rand(3, 2) + rng.randn(20, 2) * 0.1
    model = MultiOutputRegressor(LinearRegression()).fit(X, Y)
    ax = plot_predictions(model, X, Y)
    assert ax.get_title()

