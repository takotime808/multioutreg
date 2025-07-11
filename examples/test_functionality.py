# Copyright (c) 2025 takotime808

import numpy as np
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split

from multioutreg.performance.metrics_generalized_api import get_uq_performance_metrics_flexible

rng = np.random.RandomState(42)
X = rng.rand(300, 5)
Y = np.dot(X, rng.rand(5, 3)) + rng.randn(300, 3) * 0.1

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

base_gp = GaussianProcessRegressor(random_state=0)
multi_gp = MultiOutputRegressor(base_gp)
multi_gp.fit(X_train, Y_train)

metrics_df, overall_metrics = get_uq_performance_metrics_flexible(multi_gp, X_test, Y_test)

print("Available columns:", metrics_df.columns)
# metrics_to_plot = [m for m in ['rmse', 'mae', 'nll', 'miscal_area'] if m in metrics_df.columns]
metrics_to_plot = 
if not metrics_to_plot:
    print("No matching metrics found in metrics_df. Available columns:", metrics_df.columns)
else:
    ax = metrics_df[metrics_to_plot].plot.bar(figsize=(10, 6))
    ax.set_xticklabels([f"Output {i}" for i in metrics_df['output']])
    plt.xlabel('Output')
    plt.title('Uncertainty Toolbox Metrics per Output')
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.show()
