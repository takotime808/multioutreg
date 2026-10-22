# Copyright (c) 2025 takotime808

import numpy as np
import pandas as pd
import uncertainty_toolbox as uct
from typing import Optional, Tuple, Dict, Any
from sklearn.multioutput import MultiOutputRegressor


def _std_from_cov(cov: np.ndarray, n_samples: int) -> np.ndarray:
    """Per-sample std from a ``return_cov`` covariance.

    sklearn's GP returns ``(n, n)`` for one target and ``(n, n, n_targets)``
    for several; per-sample output covariances ``(n, k, k)`` are also accepted.
    """
    cov = np.asarray(cov)
    if cov.ndim == 2:
        return np.sqrt(np.diag(cov))
    if cov.shape[0] == cov.shape[1] == n_samples:
        return np.sqrt(np.diagonal(cov, axis1=0, axis2=1).T)
    return np.sqrt(np.diagonal(cov, axis1=1, axis2=2))


def _via_return_std(est, X, n_samples):
    pred, std = est.predict(X, return_std=True)
    return pred, std


def _via_return_cov(est, X, n_samples):
    pred, cov = est.predict(X, return_cov=True)
    return pred, _std_from_cov(cov, n_samples)


def _via_predict_std(est, X, n_samples):
    return est.predict(X), est.predict_std(X)


def _via_predict_var(est, X, n_samples):
    return est.predict(X), np.sqrt(est.predict_var(X))


_UNCERTAINTY_APIS = {
    "return_std": _via_return_std,
    "return_cov": _via_return_cov,
    "predict_std": _via_predict_std,
    "predict_var": _via_predict_var,
}


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
    methods = ["return_std", "return_cov", "predict_std", "predict_var"]
    if method in methods:
        methods.remove(method)
        methods.insert(0, method)
    n_samples = len(X)
    for name in methods:
        try:
            return _UNCERTAINTY_APIS[name](estimator, X, n_samples)
        except Exception:
            continue

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
        overall_metrics: dict of metrics computed on all outputs at once (flattened),
            after dividing each output by the std of its ``y_test`` column, so
            rmse/mae/nll are in standard-deviation units rather than mixed units.
    """
    y_pred: Optional[np.ndarray] = None
    y_std: Optional[np.ndarray] = None

    if y_pred_std is not None:
        y_pred = model.predict(X_test)
        y_std = np.sqrt(y_pred_std) if std_is_var else y_pred_std
    else:
        y_pred = None
        if isinstance(model, MultiOutputRegressor) and hasattr(model, "estimators_"):
            try:
                preds, stds = [], []
                for est in model.estimators_:
                    p, sd = _predict_with_uncertainty(est, X_test, uncertainty_method)
                    preds.append(np.asarray(p).reshape(-1, 1))
                    stds.append(np.asarray(sd).reshape(-1, 1))
                y_pred, y_std = np.hstack(preds), np.hstack(stds)
            except Exception:
                y_pred = None
        if y_pred is None:
            try:
                y_pred, y_std = _predict_with_uncertainty(model, X_test, uncertainty_method)
            except RuntimeError:
                raise RuntimeError(
                    "Could not extract uncertainty from model. Please supply y_pred_std manually or specify 'uncertainty_method'."
                ) from None

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
    # Outputs can have different units, so pool them on a common scale: each
    # output divided by the std of its y_true (1 when that is 0 or undefined).
    scale = np.std(y_true, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    z_true, z_pred, z_std = y_true / scale, y_pred / scale, y_std / scale
    try:
        overall_metrics: Dict[str, float] = uct.get_all_metrics(
            z_pred.flatten(),
            z_std.flatten(),
            z_true.flatten()
        )
    except AssertionError:
        overall_metrics = uct.get_all_metrics(
            z_pred.flatten(),
            np.clip(z_std.flatten(), 1e-12, None),
            z_true.flatten()
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