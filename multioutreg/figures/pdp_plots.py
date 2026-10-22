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
