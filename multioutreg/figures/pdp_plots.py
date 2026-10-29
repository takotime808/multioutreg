# Copyright (c) 2025 takotime808


import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import PartialDependenceDisplay

from multioutreg.gui.report_plotting_utils import (
    plot_to_b64,
)


class _OutputColumn(RegressorMixin, BaseEstimator):
    """Already-fitted single-output view of one column of ``model.predict``.

    Lets sklearn's PDP explain models that have no ``estimators_`` (e.g.
    MultiFidelitySurrogate, MixtureOfExpertsSurrogate, BNNSurrogate), or whose
    per-output estimators sklearn rejects. ``predict`` may return
    ``(mean, std)``; only the mean is used.
    """

    def __init__(self, model: Any = None, index: int = 0):
        self.model = model
        self.index = index

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        pred = self.model.predict(np.asarray(X, dtype=np.float64))
        if isinstance(pred, tuple):
            pred = pred[0]
        pred = np.asarray(pred)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred[:, self.index]

    def __sklearn_is_fitted__(self):
        return True


def _pdp_candidates(model: Any, i: int) -> List[Any]:
    """Estimators to try, in order, for PDP of output ``i``."""
    cands = []
    if hasattr(model, "estimators_") and i < len(model.estimators_):
        est = model.estimators_[i]
        cands.append(est)
        if hasattr(est, "predict"):
            cands.append(_OutputColumn(est, 0))
    elif hasattr(model, "predict"):
        cands.append(_OutputColumn(model, i))
    return cands


def generate_pdp_plot(
    model: Any,
    X: np.ndarray,
    output_names: List[str],
    feature_names: List[str]
) -> Dict[str, str]:
    """
    Generate partial dependence plots (PDPs) for each output dimension.

    Parameters
    ----------
    model : Any
        Fitted multi-output model. Uses ``model.estimators_[i]`` when present,
        and otherwise column ``i`` of ``model.predict``.
    X : np.ndarray
        Input features.
    output_names : List[str]
        Names of output dimensions.
    feature_names : List[str]
        Names of the input features.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping output names to base64-encoded PDP plots.
    """
    plots = {}
    for i, name in enumerate(output_names):
        # Resolve candidates outside the closure to avoid the closure-in-loop bug.
        cands = _pdp_candidates(model, i)

        def plot_fn(_name=name, _cands=cands):
            plt.figure()
            plt.title(f"{_name}")
            for est in _cands:
                try:
                    PartialDependenceDisplay.from_estimator(
                        est, X, range(X.shape[1]),
                        feature_names=feature_names, ax=plt.gca()
                    )
                    return
                except Exception:
                    plt.clf()
                    plt.title(f"{_name}")
            plt.text(0.5, 0.5,
                     f"PDP not supported for {type(model).__name__}",
                     ha='center')
            plt.axis('off')

        plots[name] = plot_to_b64(plot_fn)
    return plots


def _predict_matrix(model: Any, X: np.ndarray, n_outputs: int) -> np.ndarray:
    """``model.predict(X)`` as an ``(n, n_outputs)`` array (mean only)."""
    if hasattr(model, "estimators_") and len(model.estimators_) == n_outputs:
        try:
            cols = [_OutputColumn(est, 0).predict(X) for est in model.estimators_]
            return np.column_stack(cols)
        except Exception:
            pass
    pred = model.predict(np.asarray(X, dtype=np.float64))
    if isinstance(pred, tuple):
        pred = pred[0]
    pred = np.asarray(pred, dtype=np.float64).reshape(len(X), -1)
    if pred.shape[1] != n_outputs:
        raise ValueError(
            f"model.predict returned {pred.shape[1]} columns, expected {n_outputs}")
    return pred


def pdp_grid_data(
    model: Any,
    X: np.ndarray,
    output_names: List[str],
    feature_names: List[str],
    features: List[int] | None = None,
    grid_resolution: int = 20,
    max_rows: int = 150,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Partial dependence and ICE spread as numbers, for charts drawn elsewhere.

    For each feature in ``features`` (default: all), the feature is set to each
    of ``grid_resolution`` values between its 5th and 95th percentile on up to
    ``max_rows`` rows of ``X`` and the model is evaluated. The mean over rows is
    the partial dependence; the 10th and 90th percentiles of the individual
    (ICE) curves show how much the effect varies between rows.

    Returns ``{"features": [names], "grid": {feature: [x...]},
    "curves": {output: {feature: {"mean": [...], "p10": [...], "p90": [...]}}}}``.
    """
    X = np.asarray(X, dtype=np.float64)
    if features is None:
        features = list(range(X.shape[1]))
    rng = np.random.default_rng(random_state)
    rows = X if len(X) <= max_rows else X[rng.choice(len(X), max_rows, replace=False)]
    n_out = len(output_names)
    grid: Dict[str, list] = {}
    curves: Dict[str, Dict[str, dict]] = {name: {} for name in output_names}
    for f in features:
        fname = feature_names[f]
        lo, hi = np.percentile(X[:, f], [5, 95])
        values = np.unique(X[:, f])
        xs = values if len(values) <= grid_resolution else np.linspace(lo, hi, grid_resolution)
        stacked = np.repeat(rows[np.newaxis, :, :], len(xs), axis=0)
        stacked[:, :, f] = xs[:, np.newaxis]
        pred = _predict_matrix(model, stacked.reshape(-1, X.shape[1]), n_out)
        ice = pred.reshape(len(xs), len(rows), n_out)  # (grid, rows, outputs)
        grid[fname] = xs.tolist()
        for j, name in enumerate(output_names):
            curves[name][fname] = {
                "mean": ice[:, :, j].mean(axis=1).tolist(),
                "p10": np.percentile(ice[:, :, j], 10, axis=1).tolist(),
                "p90": np.percentile(ice[:, :, j], 90, axis=1).tolist(),
            }
    return {"features": [feature_names[f] for f in features], "grid": grid, "curves": curves}
