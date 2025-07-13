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


if __name__ == "__main__":
    # Generate example data and run the function
    n_samples = 200
    n_features = 3
    n_targets = 3
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features)
    Y = X @ rng.randn(n_features, n_targets) + rng.randn(n_samples, n_targets) * 0.5

    plot_multioutput_confidence_intervals(X, Y, z=1.96)


