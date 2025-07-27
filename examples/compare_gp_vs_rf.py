import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.base import clone
import matplotlib.pyplot as plt

# ==== Ensemble wrapper for uncertainty estimation ====
class EnsembleRegressor:
    def __init__(self, base_estimator, n_estimators=10, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = []

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.models = []
        for _ in range(self.n_estimators):
            model = clone(self.base_estimator)
            indices = rng.choice(len(X), size=len(X), replace=True)
            model.fit(X[indices], y[indices])
            self.models.append(model)
        return self

    def predict(self, X, return_std=False):
        predictions = np.array([model.predict(X) for model in self.models])
        mean_prediction = np.mean(predictions, axis=0)
        if return_std:
            std_prediction = np.std(predictions, axis=0)
            return mean_prediction, std_prediction
        return mean_prediction

# ==== Generate synthetic regression data ====
X, y = make_regression(n_samples=300, n_features=4, n_targets=3, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# ==== Create GP and RF regressors per target ====
gp_ensemble = [
    EnsembleRegressor(GaussianProcessRegressor(kernel=RBF()), n_estimators=10, random_state=i)
    for i in range(y.shape[1])
]

rf_ensemble = [
    EnsembleRegressor(RandomForestRegressor(n_estimators=100, max_depth=5), n_estimators=10, random_state=i)
    for i in range(y.shape[1])
]

# ==== Train all models ====
for i in range(y.shape[1]):
    gp_ensemble[i].fit(X_train, y_train[:, i])
    rf_ensemble[i].fit(X_train, y_train[:, i])

# ==== Predict and collect outputs ====
y_gp_pred, y_gp_std = zip(*[gp_ensemble[i].predict(X_test, return_std=True) for i in range(y.shape[1])])
y_rf_pred, y_rf_std = zip(*[rf_ensemble[i].predict(X_test, return_std=True) for i in range(y.shape[1])])

y_gp_pred = np.stack(y_gp_pred, axis=1)
y_gp_std = np.stack(y_gp_std, axis=1)
y_rf_pred = np.stack(y_rf_pred, axis=1)
y_rf_std = np.stack(y_rf_std, axis=1)

# ==== Print comparison ====
df = pd.DataFrame({
    "Target 1 GP Pred": y_gp_pred[:, 0],
    "Target 1 GP Std": y_gp_std[:, 0],
    "Target 1 RF Pred": y_rf_pred[:, 0],
    "Target 1 RF Std": y_rf_std[:, 0],
    "Target 2 GP Pred": y_gp_pred[:, 1],
    "Target 2 GP Std": y_gp_std[:, 1],
    "Target 2 RF Pred": y_rf_pred[:, 1],
    "Target 2 RF Std": y_rf_std[:, 1],
    "Target 3 GP Pred": y_gp_pred[:, 2],
    "Target 3 GP Std": y_gp_std[:, 2],
    "Target 3 RF Pred": y_rf_pred[:, 2],
    "Target 3 RF Std": y_rf_std[:, 2],
})

print("Sample predictions and uncertainty (GP vs RF):")
print(df.head())

# ==== Optional: Plot comparison per target ====
for i in range(y.shape[1]):
    plt.figure(figsize=(10, 5))
    plt.errorbar(np.arange(len(y_gp_pred)), y_gp_pred[:, i], yerr=y_gp_std[:, i], fmt='o', alpha=0.5, label='GP')
    plt.errorbar(np.arange(len(y_rf_pred)), y_rf_pred[:, i], yerr=y_rf_std[:, i], fmt='x', alpha=0.5, label='RF')
    plt.title(f"Target {i+1} Prediction with Uncertainty (GP vs RF)")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Prediction")
    plt.legend()
    plt.tight_layout()
    plt.show()
