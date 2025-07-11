# Copyright (c) 2025 takotime808

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from uncertainty_toolbox.metrics import get_all_metrics


def get_uq_performance_metrics_flexible(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    uncertainty_method: Optional[str] = None,
    y_pred_std: Optional[np.ndarray] = None,
    std_is_var: bool = False
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Computes UQ metrics for any model: handles different APIs for uncertainty prediction.

    Args:
        model: Your trained regressor (must implement predict and, optionally, predict_std or predict_var).
        X_test: Test features, shape (n_samples, n_features).
        y_test: Test targets, shape (n_samples, n_outputs).
        uncertainty_method: Optional string, one of:
            ['return_std', 'return_cov', 'predict_std', 'predict_var']
        y_pred_std: Optional (n_samples, n_outputs) ndarray, directly supply stddevs or variances.
        std_is_var: If True, will sqrt the std array passed in (for variance-to-std conversion).

    Returns:
        metrics_df: pd.DataFrame with metrics per output.
        overall_metrics: dict of metrics computed on all outputs at once (flattened).
    """
    # For different python versions.
    try:
        y_pred: Optional[np.ndarray] = None
        y_std: Optional[np.ndarray] = None
    except:
        y_pred = None
        y_std = None

    if y_pred_std is not None:
        y_pred = model.predict(X_test)
        y_std = np.sqrt(y_pred_std) if std_is_var else y_pred_std
    else:
        tried: bool = False
        if uncertainty_method == 'return_std':
            try:
                y_pred, y_std = model.predict(X_test, return_std=True)
                tried = True
            except Exception:
                pass
        if not tried and uncertainty_method == 'return_cov':
            try:
                y_pred, y_cov = model.predict(X_test, return_cov=True)
                y_std = np.sqrt(np.diagonal(y_cov, axis1=1, axis2=2))
                tried = True
            except Exception:
                pass
        if not tried and uncertainty_method == 'predict_std':
            try:
                y_pred = model.predict(X_test)
                y_std = model.predict_std(X_test)
                tried = True
            except Exception:
                pass
        if not tried and uncertainty_method == 'predict_var':
            try:
                y_pred = model.predict(X_test)
                y_var = model.predict_var(X_test)
                y_std = np.sqrt(y_var)
                tried = True
            except Exception:
                pass
        if not tried:
            try:
                y_pred, y_std = model.predict(X_test, return_std=True)
                tried = True
            except Exception:
                pass
        if not tried:
            try:
                y_pred, y_cov = model.predict(X_test, return_cov=True)
                y_std = np.sqrt(np.diagonal(y_cov, axis1=1, axis2=2))
                tried = True
            except Exception:
                pass
        if not tried:
            try:
                y_pred = model.predict(X_test)
                y_std = model.predict_std(X_test)
                tried = True
            except Exception:
                pass
        if not tried:
            try:
                y_pred = model.predict(X_test)
                y_var = model.predict_var(X_test)
                y_std = np.sqrt(y_var)
                tried = True
            except Exception:
                pass
        if not tried:
            raise RuntimeError(
                "Could not extract uncertainty from model. Please supply y_pred_std manually or specify 'uncertainty_method'."
            )

    y_true: np.ndarray = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    y_std = np.abs(np.asarray(y_std))
    if y_pred.shape != y_true.shape or y_std.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}, y_std {y_std.shape}"
        )

    n_outputs: int = y_true.shape[1]
    metrics_list: list[Dict[str, float | int]] = []
    for i in range(n_outputs):
        metrics = get_all_metrics(
            y_true[:, i],
            y_pred[:, i],
            y_std[:, i]
        )
        metrics['output'] = i
        metrics_list.append(metrics)

    metrics_df: pd.DataFrame = pd.DataFrame(metrics_list)
    overall_metrics: Dict[str, float] = get_all_metrics(
        y_true.flatten(),
        y_pred.flatten(),
        y_std.flatten()
    )
    print(metrics_df)
    print("\nOverall metrics:", overall_metrics)
    return metrics_df, overall_metrics

