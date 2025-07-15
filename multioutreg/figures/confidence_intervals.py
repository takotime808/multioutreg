# Copyright (c) 2025 takotime808

import numpy as np
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def plot_multioutput_confidence_intervals(
    X, Y, z=1.96, base_estimator=None, test_size=0.2, random_state=42
):
    """
    Fit a MultiOutputRegressor and plot predictions with z-scaled confidence intervals.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    Y : np.ndarray
        Target matrix of shape (n_samples, n_targets).
    z : float
        Z-value for confidence interval (e.g., 1.96 for 95% CI).
    base_estimator : sklearn estimator, optional
        Regressor to use in MultiOutputRegressor (default: RandomForestRegressor).
    test_size : float
        Fraction of data to use for testing.
    random_state : int
        Random seed for reproducibility.
    """
    if base_estimator is None:
        base_estimator = RandomForestRegressor(n_estimators=100, random_state=random_state)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    model = MultiOutputRegressor(base_estimator)
    model.fit(X_train, Y_train)
    
    means = []
    stds = []
    for est in model.estimators_:
        preds = np.array([tree.predict(X_test) for tree in est.estimators_])
        means.append(preds.mean(axis=0))
        stds.append(preds.std(axis=0))
    means = np.stack(means, axis=1)
    stds = np.stack(stds, axis=1)

    n_targets = Y.shape[1]
    fig, axes = plt.subplots(n_targets, 1, figsize=(8, 3*n_targets), sharex=True)
    if n_targets == 1:
        axes = [axes]
    x_axis = np.arange(X_test.shape[0])

    for i in range(n_targets):
        ax = axes[i]
        ax.plot(x_axis, Y_test[:, i], 'o', label='True')
        ax.plot(x_axis, means[:, i], '-', label='Predicted')
        ax.fill_between(
            x_axis,
            means[:, i] - z * stds[:, i],
            means[:, i] + z * stds[:, i],
            color='orange',
            alpha=0.3,
            label=f'{z:.2f}σ band' if i == 0 else None,
        )
        ax.set_title(f'Target {i+1}')
        ax.legend()
        ax.set_ylabel('Value')
    axes[-1].set_xlabel('Test Sample Index')
    plt.tight_layout()
    plt.show()


def plot_confidence_interval(
    y_true, y_pred, y_std, output_names=None, n_cols=3, alpha=0.05
):
    """
    Plot predicted values with 95% confidence intervals (confidence interval plot).
    Handles single and multi-output regression.

    Parameters:
    - y_true: shape (n_samples, n_outputs) or (n_samples,)
    - y_pred: shape (n_samples, n_outputs) or (n_samples,)
    - y_std: shape (n_samples, n_outputs) or (n_samples,)
    - output_names: list of output variable names (optional)
    - n_cols: number of subplot columns for multi-output (default: 3)
    - alpha: significance level for confidence interval (default 0.05 for 95%)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_std  = np.asarray(y_std)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        y_std  = y_std.reshape(-1, 1)

    n_targets = y_true.shape[1]
    z = abs(np.percentile(np.random.normal(0, 1, 100000), 100 * (1 - alpha / 2)))  # ~1.96 for 95%

    # Use provided names or default
    if output_names is None:
        output_names = [f"Output {i}" for i in range(n_targets)]

    # Single output case
    if n_targets == 1:
        fig, ax = plt.subplots(figsize=(7, 5))
        lower = y_pred[:, 0] - z * y_std[:, 0]
        upper = y_pred[:, 0] + z * y_std[:, 0]
        ax.plot(y_true[:, 0], label='True', marker='o', linestyle='None', alpha=0.6)
        ax.plot(y_pred[:, 0], label='Predicted', marker='x', linestyle='None', alpha=0.8)
        ax.fill_between(
            np.arange(len(y_pred)),
            lower,
            upper,
            color="gray",
            alpha=0.3,
            label=f'{int((1-alpha)*100)}% CI'
        )
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Value")
        ax.set_title(output_names[0])
        ax.legend()
        plt.tight_layout()
        plt.show()
        return

    # Multi-output
    n_rows = int(np.ceil(n_targets / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs = np.array(axs).reshape(-1)

    for i in range(n_targets):
        lower = y_pred[:, i] - z * y_std[:, i]
        upper = y_pred[:, i] + z * y_std[:, i]
        axs[i].plot(y_true[:, i], label='True', marker='o', linestyle='None', alpha=0.6)
        axs[i].plot(y_pred[:, i], label='Predicted', marker='x', linestyle='None', alpha=0.8)
        axs[i].fill_between(
            np.arange(len(y_pred)),
            lower,
            upper,
            color="gray",
            alpha=0.3,
            label=f'{int((1-alpha)*100)}% CI'
        )
        axs[i].set_xlabel("Sample index")
        axs[i].set_ylabel("Value")
        axs[i].set_title(output_names[i])
        axs[i].legend()
    # Hide unused axes
    for j in range(n_targets, n_rows * n_cols):
        axs[j].set_visible(False)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
        plt.close()
    else:
        plt.show()
    return