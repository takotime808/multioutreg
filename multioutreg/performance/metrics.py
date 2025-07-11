# Copyright (c) 2025 takotime808

import numpy as np
import pandas as pd
from typing import Any
import uncertainty_toolbox as uct

def get_uq_performance_metrics(
        model: Any, 
        X_test: np.ndarray, 
        y_test: np.ndarray, 
        std_method: str ='return_std',
):
    """
    Computes uncertainty-toolbox performance metrics for each output of a multioutput regressor.

    Args:
        model: MultiOutputRegressor or similar with predict(X, return_std=...) support
        X_test: Test features, shape (n_samples, n_features)
        y_test: Test targets, shape (n_samples, n_outputs)
        std_method: Which kwarg to use for uncertainty (default 'return_std')

    Returns:
        metrics_df: pd.DataFrame with metrics per output
        metrics_overall: dict of metrics computed on all outputs at once (flattened)
    """
    # Predict means and stds
    try:
        # Some regressors require return_std=True, others return_uncertainty=True
        y_pred, y_std = model.predict(X_test, **{std_method: True})
    except TypeError:
        # Try 'return_std' as fallback if user passed wrong std_method
        y_pred, y_std = model.predict(X_test, return_std=True)

    y_true = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    y_std  = np.asarray(y_std)
    n_outputs = y_true.shape[1]

    metrics_list = []
    for i in range(n_outputs):
        metrics = uct.get_all_metrics(
            y_true[:, i],
            y_pred[:, i],
            y_std[:, i]
        )
        metrics['output'] = i
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)

    # Overall metrics on flattened arrays
    overall_metrics = uct.get_all_metrics(
        y_true.flatten(),
        y_pred.flatten(),
        y_std.flatten()
    )

    print(metrics_df)
    print("Overall metrics:", overall_metrics)
    return metrics_df, overall_metrics

# Example usage:
# metrics_df, overall_metrics = get_uq_performance_metrics(my_multioutput_model, X_test, y_test)
