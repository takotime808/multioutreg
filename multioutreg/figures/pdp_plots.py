# Copyright (c) 2025 takotime808


import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from sklearn.inspection import PartialDependenceDisplay

from multioutreg.gui.report_plotting_utils import (
    plot_to_b64,
)


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
        Multi-output model with `estimators_` attribute.
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
        def plot_fn():
            plt.figure()
            plt.title(f"{name}")
            try:
                PartialDependenceDisplay.from_estimator(
                    model.estimators_[i], X, range(X.shape[1]), feature_names=feature_names, ax=plt.gca()
                )
            except Exception as e:
                plt.text(0.5, 0.5, f"PDP not supported for {type(model.estimators_[i]).__name__}", ha='center')
                plt.axis('off')
        plots[name] = plot_to_b64(plot_fn)
    return plots

