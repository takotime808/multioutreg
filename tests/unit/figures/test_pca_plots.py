# Copyright (c) 2025 takotime808

import pytest
import numpy as np
from sklearn.decomposition import PCA
from multioutreg.figures.pca_plots import generate_pca_variance_plot

def test_generate_pca_variance_plot():
    X = np.random.rand(50, 4)
    pca = PCA().fit(X)
    b64 = generate_pca_variance_plot(pca, n_selected=2, threshold=0.8)
    assert isinstance(b64, str)
    assert len(b64) > 100

# # For bar plot.
# def test_generate_pca_variance_plot():
#     X = np.random.rand(50, 4)
#     pca = PCA(n_components=3).fit(X)
#     b64 = generate_pca_variance_plot(pca)
#     assert isinstance(b64, str)
#     assert len(b64) > 100