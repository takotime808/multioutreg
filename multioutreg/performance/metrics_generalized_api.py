# Copyright (c) 2025 takotime808

import numpy as np
import pandas as pd
import uncertainty_toolbox as uct
from typing import Optional, Tuple, Dict, Any
from sklearn.multioutput import MultiOutputRegressor


def _predict_with_uncertainty(
    estimator: Any,
    X: np.ndarray,
    method: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predicts mean and uncertainty (standard deviation) from a scikit-learn-style estimator
    using a flexible API.

    Attempts the provided method first, then tries several common uncertainty prediction APIs:
      - predict(X, return_std=True)
      - predict(X, return_cov=True)
      - predict(X) and predict_std(X)
      - predict(X) and predict_var(X)

    Args:
        estimator (Any): The trained regressor or estimator object.
        X (np.ndarray): Input feature matrix of shape (n_samples, n_features).
        method (Optional[str]): If specified, try this method first.
            One of: 'return_std', 'return_cov', 'predict_std', 'predict_var'.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - pred: Predicted means, shape (n_samples, [n_outputs])
            - std: Predicted standard deviations (uncertainties), shape matches pred.

    Raises:
        RuntimeError: If none of the prediction APIs are supported by the estimator.

    Example:
        pred, std = _predict_with_uncertainty(my_gp, X_test, method='return_std')
    """
    tried = False
    if method == "return_std":
        try:
            pred, std = estimator.predict(X, return_std=True)
            return pred, std
        except Exception:
            tried = False
        else:
            tried = True
    if not tried and method == "return_cov":
        try:
            pred, cov = estimator.predict(X, return_cov=True)
            std = np.sqrt(np.diagonal(cov, axis1=1, axis2=2)) if cov.ndim == 3 else np.sqrt(np.diag(cov))
            return pred, std
        except Exception:
            tried = False
        else:
            tried = True
    if not tried and method == "predict_std":
        try:
            pred = estimator.predict(X)
            std = estimator.predict_std(X)
            return pred, std
        except Exception:
            tried = False
        else:
            tried = True
    if not tried and method == "predict_var":
        try:
            pred = estimator.predict(X)
            var = estimator.predict_var(X)
            return pred, np.sqrt(var)
        except Exception:
            tried = False
        else:
            tried = True

    # Try a sequence of common methods as fallback
    try:
        pred, std = estimator.predict(X, return_std=True)
        return pred, std
    except Exception:
        pass

    try:
        pred, cov = estimator.predict(X, return_cov=True)
        std = np.sqrt(np.diagonal(cov, axis1=1, axis2=2)) if cov.ndim == 3 else np.sqrt(np.diag(cov))
        return pred, std
    except Exception:
        pass

    try:
        pred = estimator.predict(X)
        std = estimator.predict_std(X)
        return pred, std
    except Exception:
        pass

    try:
        pred = estimator.predict(X)
        var = estimator.predict_var(X)
        return pred, np.sqrt(var)
    except Exception:
        pass

    raise RuntimeError(
        "Could not extract uncertainty from the provided estimator."
    )

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
        if isinstance(model, MultiOutputRegressor):
            try:
                preds = []
                stds = []
                for est in model.estimators_:
                    p, s = _predict_with_uncertainty(est, X_test, uncertainty_method)
                    preds.append(p.reshape(-1, 1))
                    stds.append(s.reshape(-1, 1))
                y_pred = np.hstack(preds)
                y_std = np.hstack(stds)
                tried = True
            except Exception:
                tried = False
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
    y_std = np.clip(y_std, 1e-12, None)
    if y_pred.shape != y_true.shape or y_std.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}, y_std {y_std.shape}"
        )

    n_outputs: int = y_true.shape[1]
    metrics_list: list[Dict[str, float | int]] = []
    for i in range(n_outputs):
        try:
            metrics = uct.get_all_metrics(
                y_pred[:, i],
                y_std[:, i],
                y_true[:, i]
            )
        except AssertionError:
            metrics = uct.get_all_metrics(
                y_pred[:, i],
                np.clip(y_std[:, i], 1e-12, None),
                y_true[:, i]
            )

        flat_metrics = {
            'rmse': metrics['accuracy']['rmse'],
            'mae': metrics['accuracy']['mae'],
            'nll': metrics['scoring_rule']['nll'],
            'miscal_area': metrics['avg_calibration']['miscal_area'],
            'output': i,
        }
        metrics_list.append(flat_metrics)

    metrics_df: pd.DataFrame = pd.DataFrame(metrics_list)
    try:
        overall_metrics: Dict[str, float] = uct.get_all_metrics(
            y_pred.flatten(),
            y_std.flatten(),
            y_true.flatten()
        )
    except AssertionError:
        overall_metrics = uct.get_all_metrics(
            y_pred.flatten(),
            np.clip(y_std.flatten(), 1e-12, None),
            y_true.flatten()
        )
    overall_flat = {
        'rmse': overall_metrics['accuracy']['rmse'],
        'mae': overall_metrics['accuracy']['mae'],
        'nll': overall_metrics['scoring_rule']['nll'],
        'miscal_area': overall_metrics['avg_calibration']['miscal_area'],
    }
    print(metrics_df)
    print("\nOverall metrics:", overall_flat)
    return metrics_df, overall_flat