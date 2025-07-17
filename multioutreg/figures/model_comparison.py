# Copyright (c) 2025 takotime808

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin, BaseEstimator
from typing import Optional, Union

def plot_surrogate_model_summary(
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    model: Optional[RegressorMixin] = None,
    compare: bool = True,
    rmse_plot_index: int = 1,
    savefig: Optional[str] = "surrogate_model_summary.png",
    admin_mode: bool = True,
) -> Union[str, plt.Figure]:
    """
    Generate a 3-row summary visualization of a multi-output regression surrogate model.

    Rows:
        1. Z-score histograms (uncertainty calibration)
        2. Prediction vs. true values (accuracy)
        3. RMSE bar chart comparing the given model and a noisy baseline (optional)

    Parameters:
        X_train (np.ndarray): Training features of shape (n_samples, n_features)
        X_test (np.ndarray): Test features of shape (n_samples, n_features)
        Y_train (np.ndarray): Training targets of shape (n_samples, n_outputs)
        Y_test (np.ndarray): Test targets of shape (n_samples, n_outputs)
        model (RegressorMixin, optional): Fitted regression model; if None, a linear model is used.
        compare (bool): If True, compares to a noisy model as a baseline in RMSE plot.
        rmse_plot_index (int): Column index in grid to place the RMSE plot (default: 1)
        savefig (str or None): Path to save PNG output. If None, returns the matplotlib Figure.
        admin_mode (bool): If True, print status messages.

    Returns:
        Union[str, plt.Figure]: Path to saved file if `savefig` is given, else the matplotlib Figure.
    """
    n_targets = Y_train.shape[1]
    
    if model is None:
        model = MultiOutputRegressor(LinearRegression())
        model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    residuals = Y_test - Y_pred
    Y_std = np.std(residuals, axis=0, keepdims=True) * np.ones_like(Y_pred)

    # Optional comparison with noisy model
    if compare:
        noise_model = MultiOutputRegressor(LinearRegression())
        Y_train_noisy = Y_train + np.random.normal(0, 10, size=Y_train.shape)
        noise_model.fit(X_train, Y_train_noisy)
        Y_pred_noise = noise_model.predict(X_test)
        rmse_noise = np.sqrt(mean_squared_error(Y_test, Y_pred_noise, multioutput='raw_values'))
    else:
        rmse_noise = None

    rmse_clean = np.sqrt(mean_squared_error(Y_test, Y_pred, multioutput='raw_values'))

    # === Grid layout ===
    n_cols = min(n_targets, 3)
    n_rows = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 12))
    if n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])  # force 2D indexing

    fig.suptitle("Surrogate Model Summary Report", fontsize=20)

    # --- Row 1: Z-score Histograms
    for i in range(n_cols):
        if i >= n_targets:
            axes[0, i].axis('off')
            continue
        z_scores = (Y_test[:, i] - Y_pred[:, i]) / Y_std[:, i]
        sns.histplot(z_scores, bins=30, kde=True, ax=axes[0, i], color="skyblue")
        axes[0, i].set_title(f"Z-score Histogram (Target {i})")
        axes[0, i].set_xlabel("Z-score")
        axes[0, i].set_ylabel("Frequency")

    # --- Row 2: Prediction vs True
    for i in range(n_cols):
        if i >= n_targets:
            axes[1, i].axis('off')
            continue
        sns.scatterplot(x=Y_pred[:, i], y=Y_test[:, i], ax=axes[1, i])
        axes[1, i].plot([Y_test[:, i].min(), Y_test[:, i].max()],
                        [Y_test[:, i].min(), Y_test[:, i].max()], 'r--')
        axes[1, i].set_title(f"Prediction vs True (Target {i})")
        axes[1, i].set_xlabel("Predicted")
        axes[1, i].set_ylabel("True")

    # --- Row 3: RMSE Comparison Bar Chart
    ax_rmse = axes[2, rmse_plot_index]
    width = 0.35
    indices = np.arange(n_targets)
    ax_rmse.bar(indices - width/2, rmse_clean, width, label='Model A', color='green')
    if rmse_noise is not None:
        ax_rmse.bar(indices + width/2, rmse_noise, width, label='Model B', color='orange')

    ax_rmse.set_xticks(indices)
    ax_rmse.set_xticklabels([f"Target {i}" for i in range(n_targets)])
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_title("Model Comparison: RMSE per Output")
    ax_rmse.legend()

    # Turn off unused subplots
    for j in range(n_cols):
        if j != rmse_plot_index:
            axes[2, j].axis('off')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if savefig:
        fig.savefig(savefig, dpi=300)
        if admin_mode:
            print(f"Saved: {savefig}")
        return savefig
    else:
        return fig
