import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.base import clone
import lightgbm as lgb
import matplotlib.pyplot as plt

# ==== Gaussian Process Ensemble Wrapper ====
class EnsembleHeteroscedasticRegressor:
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

# ==== LightGBM Quantile Regressor ====
class LightGBMQuantileRegressor:
    def __init__(self, alpha=0.9, lower_alpha=0.1, **kwargs):
        self.alpha = alpha
        self.lower_alpha = lower_alpha
        self.kwargs = kwargs
        self.model_upper = None
        self.model_lower = None
        self.model_median = None

    def fit(self, X, y):
        self.model_lower = lgb.LGBMRegressor(objective='quantile', alpha=self.lower_alpha, **self.kwargs)
        self.model_upper = lgb.LGBMRegressor(objective='quantile', alpha=self.alpha, **self.kwargs)
        self.model_median = lgb.LGBMRegressor(objective='quantile', alpha=0.5, **self.kwargs)
        self.model_lower.fit(X, y)
        self.model_upper.fit(X, y)
        self.model_median.fit(X, y)
        return self

    def predict(self, X, return_std=False):
        pred = self.model_median.predict(X)
        if return_std:
            upper = self.model_upper.predict(X)
            lower = self.model_lower.predict(X)
            std = np.abs((upper - lower) / 2.0)  # Ensure non-negative std
            return pred, std
        return pred

# ==== Generate Synthetic Data ====
X, y = make_regression(n_samples=300, n_features=4, n_targets=3, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# ==== Create Estimators ====
gp_estimators = [
    EnsembleHeteroscedasticRegressor(
        GaussianProcessRegressor(kernel=RBF(length_scale=1.0)), n_estimators=10, random_state=i
    ) for i in range(y.shape[1])
]

lgb_estimators = [
    LightGBMQuantileRegressor(n_estimators=100, max_depth=5, random_state=i)
    for i in range(y.shape[1])
]

# ==== Fit Models ====
for i in range(y.shape[1]):
    gp_estimators[i].fit(X_train, y_train[:, i])
    lgb_estimators[i].fit(X_train, y_train[:, i])

# ==== Predict ====
y_gp_pred, y_gp_std = zip(*[gp_estimators[i].predict(X_test, return_std=True) for i in range(y.shape[1])])
y_lgb_pred, y_lgb_std = zip(*[lgb_estimators[i].predict(X_test, return_std=True) for i in range(y.shape[1])])

y_gp_pred = np.stack(y_gp_pred, axis=1)
y_gp_std = np.stack(y_gp_std, axis=1)
y_lgb_pred = np.stack(y_lgb_pred, axis=1)
y_lgb_std = np.stack(y_lgb_std, axis=1)

# ==== Show Summary ====
df = pd.DataFrame({
    "Target 1 GP Pred": y_gp_pred[:, 0],
    "Target 1 GP Std": y_gp_std[:, 0],
    "Target 1 LGB Pred": y_lgb_pred[:, 0],
    "Target 1 LGB Std": y_lgb_std[:, 0],
    "Target 2 GP Pred": y_gp_pred[:, 1],
    "Target 2 GP Std": y_gp_std[:, 1],
    "Target 2 LGB Pred": y_lgb_pred[:, 1],
    "Target 2 LGB Std": y_lgb_std[:, 1],
    "Target 3 GP Pred": y_gp_pred[:, 2],
    "Target 3 GP Std": y_gp_std[:, 2],
    "Target 3 LGB Pred": y_lgb_pred[:, 2],
    "Target 3 LGB Std": y_lgb_std[:, 2],
})

print("Sample predictions with uncertainties:")
print(df.head())

# ==== Optional: Plot uncertainty comparison for Target 1 ====
plt.figure(figsize=(10, 6))
plt.errorbar(range(len(y_gp_pred)), y_gp_pred[:, 0], yerr=y_gp_std[:, 0],
             label='GP', fmt='o', alpha=0.5)
plt.errorbar(range(len(y_lgb_pred)), y_lgb_pred[:, 0], yerr=y_lgb_std[:, 0],
             label='LightGBM', fmt='x', alpha=0.5)
plt.title("Target 1 Prediction with Uncertainty (Error Bars)")
plt.xlabel("Test Sample Index")
plt.ylabel("Prediction")
plt.legend()
plt.tight_layout()
plt.show()
