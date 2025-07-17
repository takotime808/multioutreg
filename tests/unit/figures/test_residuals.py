# Copyright (c) 2025 takotime808

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for tests
# import matplotlib.pyplot as plt
import pytest
# import os

from multioutreg.figures import residuals

@pytest.mark.parametrize("n_targets", [1, 2, 3, 4])
def test_plot_residuals_multioutput_axes_count(tmp_path, n_targets):
    n_samples = 12
    y_true = np.random.randn(n_samples, n_targets)
    y_pred = y_true + 0.1 * np.random.randn(n_samples, n_targets)
    axes = residuals.plot_residuals_multioutput(y_pred, y_true, savefig=tmp_path / "fig.png")
    assert len(axes) == n_targets
    # Check the file was created
    assert (tmp_path / "fig.png").exists()

@pytest.mark.parametrize("n_targets", [1, 2, 4])
def test_plot_residuals_multioutput_with_regplot_axes_count(tmp_path, n_targets):
    n_samples = 8
    y_true = np.random.randn(n_samples, n_targets)
    y_pred = y_true + np.random.randn(n_samples, n_targets) * 0.2
    axes = residuals.plot_residuals_multioutput_with_regplot(y_pred, y_true, savefig=tmp_path / "regfig.png")
    assert len(axes) == n_targets
    # File exists
    assert (tmp_path / "regfig.png").exists()

def test_custom_target_list(tmp_path):
    n_samples, n_targets = 10, 3
    y_true = np.random.randn(n_samples, n_targets)
    y_pred = y_true + 0.1 * np.random.randn(n_samples, n_targets)
    custom_titles = ["OutputA", "OutputB", "OutputC"]
    axes = residuals.plot_residuals_multioutput(y_pred, y_true, target_list=custom_titles, savefig=tmp_path / "ct.png")
    for ax, t in zip(axes, custom_titles):
        assert t in ax.get_title()

def test_shape_input_1d_and_transpose(tmp_path):
    n_samples = 9
    y_true = np.random.randn(n_samples)
    y_pred = y_true + 0.2 * np.random.randn(n_samples)
    # Should work for 1d input
    axes = residuals.plot_residuals_multioutput(y_pred, y_true, savefig=tmp_path / "s1.png")
    assert len(axes) == 1
    # Should work for shape (1, n_samples)
    axes2 = residuals.plot_residuals_multioutput(y_pred.reshape(1, -1).T, y_true.reshape(1, -1).T, savefig=tmp_path / "s2.png")
    assert len(axes2) == 1

# def test_shared_labels_and_suptitle(tmp_path):
#     n_samples, n_targets = 6, 2
#     y_true = np.random.randn(n_samples, n_targets)
#     y_pred = y_true + 0.1 * np.random.randn(n_samples, n_targets)
#     supxlabel = "Fake X"
#     supylabel = "Fake Y"
#     suptitle = "Main Title"
#     axes = residuals.plot_residuals_multioutput(y_pred, y_true, suptitle=suptitle, supxlabel=supxlabel, supylabel=supylabel, savefig=tmp_path / "lbls.png")
#     fig = axes[0].get_figure()
#     # Check the suptitle and supxlabel/supylabel
#     assert suptitle in fig._suptitle.get_text()
#     assert fig.get_supxlabel() == supxlabel
#     assert fig.get_supylabel() == supylabel

def test_plot_residuals_multioutput_with_regplot_runs(tmp_path):
    n_samples, n_targets = 7, 3
    y_true = np.random.randn(n_samples, n_targets)
    y_pred = y_true + 0.2 * np.random.randn(n_samples, n_targets)
    axes = residuals.plot_residuals_multioutput_with_regplot(y_pred, y_true, savefig=tmp_path / "regplot.png")
    assert len(axes) == n_targets
    assert (tmp_path / "regplot.png").exists()
