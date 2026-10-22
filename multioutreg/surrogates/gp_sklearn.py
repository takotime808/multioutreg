# Copyright (c) 2025 takotime808

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from multioutreg.surrogates.base_sklearn import BaseSurrogate


def default_gp_kernel():
    """Amplitude * RBF + learned noise.

    sklearn's default (fixed amplitude 1, no noise term, alpha=1e-10) has to
    interpolate noisy data and collapses to a tiny length scale.
    """
    return ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-2)


def matern_gp_kernel():
    """Amplitude * Matern(nu=2.5) + learned noise."""
    return ConstantKernel(1.0) * Matern(1.0, nu=2.5) + WhiteKernel(1e-2)


class GaussianProcessSurrogate(BaseSurrogate):
    """Gaussian Process surrogate with uncertainty.

    Defaults to :func:`default_gp_kernel`, ``normalize_y=True`` and standardised
    inputs; any ``GaussianProcessRegressor`` keyword overrides them.
    """
    def __init__(self, scale_inputs=True, **kwargs):
        kwargs.setdefault("kernel", default_gp_kernel())
        kwargs.setdefault("normalize_y", True)
        super().__init__(GaussianProcessRegressor(**kwargs), scale_inputs=scale_inputs)