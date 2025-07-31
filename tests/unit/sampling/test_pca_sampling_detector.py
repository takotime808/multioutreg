# Copyright (c) 2025 takotime808

import numpy as np
import pytest
from scipy.stats import qmc
from multioutreg.sampling.pca_sampling_detector import generate_sampling_plot_and_metrics


@pytest.mark.parametrize("label, sampler", [
    ("Grid", lambda: np.stack(np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20)), -1).reshape(-1, 2)),
    ("Random", lambda: np.random.rand(400, 2)),
    ("Sobol", lambda: qmc.Sobol(d=2, scramble=False).random(256)),
    ("LHS", lambda: qmc.LatinHypercube(d=2).random(400)),
    ("Uncertain", lambda: np.ones((50, 2)))
])
def test_sampling_method_detection(label, sampler):
    np.random.seed(42)
    X = sampler()
    _, metrics = generate_sampling_plot_and_metrics(X, random_state=42)

    print(f"{label} → Detected: {metrics['method']}, Explanation: {metrics['explanation']}")

    if label == "Uncertain":
        assert metrics["method"] == "Uncertain"
    elif label == "Random":
        assert metrics["method"] in ["Random", "LHS"], f"Expected {label}, got {metrics['method']} — {metrics}"
    elif label == "LHS":
        assert metrics["method"] in ["LHS", "Random"], f"Expected {label}, got {metrics['method']} — {metrics}"
    else:
        assert metrics["method"] == label, f"Expected {label}, got {metrics['method']} — {metrics}"


def test_output_structure():
    X = np.random.rand(100, 3)
    fig, metrics = generate_sampling_plot_and_metrics(X)
    assert hasattr(fig, "gca"), "Output is not a matplotlib figure"
    assert isinstance(metrics, dict)
    for key in ["method", "explanation", "projection", "std", "silhouette", "entropy"]:
        assert key in metrics
