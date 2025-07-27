# Copyright (c) 2025 takotime808

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from multioutreg.utils.figure_utils import plot_to_b64

def generate_error_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_names: List[str]
) -> List[Dict[str, str]]:
    """
    Generate error histograms for each output dimension.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    output_names : List[str]
        Names of output dimensions.

    Returns
    -------
    List[Dict[str, str]]
        List of dictionaries containing base64-encoded histograms and metadata.
    """
    plots = []
    for i, name in enumerate(output_names):
        def plot_fn():
            plt.figure()
            plt.hist(y_pred[:, i] - y_true[:, i], bins=20, alpha=0.8)
            plt.xlabel("Prediction Error")
            plt.ylabel("Frequency")
            plt.title(f"Error Histogram ({name})")
        plots.append({
            "img_b64": plot_to_b64(plot_fn),
            "title": f"Error Histogram", 
            "caption": f"Histogram of prediction errors for {name}."
        })
    return plots
