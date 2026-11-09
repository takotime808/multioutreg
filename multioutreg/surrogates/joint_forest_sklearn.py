# Copyright (c) 2026 takotime808

"""Joint-tree random forest: one forest whose splits serve every output.

``RandomForestSurrogate`` fits one forest per output column. scikit-learn's
``RandomForestRegressor`` also accepts a 2-D target directly, choosing each
split to reduce the summed squared error across all outputs. The trees then
share their partition of the input space, which helps when the outputs respond
to the same features and thresholds, and costs one forest instead of ``m``.

The std is the same as ``RandomForestSurrogate``: per-output spread of the
tree predictions plus a held-out noise variance (see
:mod:`multioutreg.surrogates.forest_uncertainty`).
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from multioutreg.surrogates.conformal_mixin import ConformalMixin
from multioutreg.surrogates.forest_uncertainty import (
    _tree_predictions_3d,
    estimate_noise_variance,
)


class JointRandomForestSurrogate(ConformalMixin):
    """Random forest fitted on all outputs at once.

    Parameters
    ----------
    include_noise : bool, default True
        Add the held-out noise variance to the tree variance. ``False`` gives
        the tree spread alone (epistemic only; under-covers on noisy data).
    noise_cv : int, default 5
        Folds used to estimate the noise when the forest does not bootstrap.
    **kwargs
        Passed to :class:`sklearn.ensemble.RandomForestRegressor`.

    Attributes
    ----------
    forest_ : RandomForestRegressor
    noise_var_ : np.ndarray
        Per-output noise variance added to the tree variance.
    """

    _multi_output = True  # signals AutoDetect to evaluate on full Y

    def __init__(self, include_noise=True, noise_cv=5, **kwargs):
        self.include_noise = include_noise
        self.noise_cv = noise_cv
        self.forest_kwargs = kwargs

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.n_outputs_ = Y.shape[1]
        target = Y if self.n_outputs_ > 1 else Y[:, 0]
        self.forest_ = RandomForestRegressor(**self.forest_kwargs).fit(X, target)
        if self.include_noise:
            self.noise_var_ = estimate_noise_variance(self.forest_, X, Y, self.noise_cv)
        else:
            self.noise_var_ = np.zeros(self.n_outputs_)
        return self

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=float)
        mean = np.asarray(self.forest_.predict(X)).reshape(len(X), -1)
        if not return_std:
            return mean
        tree_var = _tree_predictions_3d(self.forest_, X).var(axis=0)
        return mean, np.sqrt(tree_var + self.noise_var_)
