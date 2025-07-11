# Copyright (c) 2025 takotime808

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
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
metrics_to_plot = [m for m in ["rmse", "mae", "miscal_area"] if m in metrics_df.columns]
has_nll = "nll" in metrics_df.columns

if not metrics_to_plot and not has_nll:
    print("No matching metrics found in metrics_df. Available columns:", metrics_df.columns)
else:
    fig, ax = plt.subplots(figsize=(10, 6))
    if metrics_to_plot:
        metrics_df.plot(x="output", y=metrics_to_plot, kind="bar", ax=ax)

    if has_nll:
        ax2 = ax.twinx()
        ax2.plot(
            ax.get_xticks(),
            metrics_df["nll"],
            color="C3",
            marker="o",
            linestyle="--",
            label="nll",
        )
        ax2.set_ylabel("NLL")

    ax.set_xticklabels([f"Output {i}" for i in metrics_df["output"]])
    plt.xlabel("Output")
    plt.title("Uncertainty Toolbox Metrics per Output")
    handles, labels = ax.get_legend_handles_labels()
    if has_nll:
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2
    ax.legend(handles, labels, title="Metric")
    plt.tight_layout()
    plt.savefig("metrics.png")
