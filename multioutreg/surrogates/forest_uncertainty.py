# Copyright (c) 2026 takotime808

"""Predictive std for tree-ensemble surrogates: tree spread + noise.

The spread of the trees' predictions measures how unsure the forest is about
the mean (epistemic). It does not include the noise in y (aleatoric), so on
noisy data intervals built from it alone are too narrow. ``ForestSurrogate``
adds a noise variance per output, estimated on held-out points:

    noise_var = max(mean((y_i - m_i)^2) - mean(v_i), 0)

where ``m_i`` and ``v_i`` are the mean and variance of the trees that did not
see point i (out-of-bag trees when the forest bootstraps, otherwise K-fold
refits). This matches the average predicted variance to the held-out mean
squared error. The predictive std is ``sqrt(tree_var(x) + noise_var)``.

``noise_var`` is the part of the error the tree spread misses, not an estimate
of the noise variance itself: fully grown trees fit the noise, so their spread
already carries some of it, and ``noise_var`` is usually below the true noise.
It is constant in x; for input-dependent noise wrap the surrogate with
conformal prediction or use a model with a variance head.
"""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold

from multioutreg.surrogates.base_sklearn import BaseSurrogate


def tree_predictions(forest, X):
    """Per-tree predictions, shape ``(n_trees, n_samples)``."""
    return np.array([tree.predict(X) for tree in forest.estimators_])


def _tree_predictions_3d(forest, X):
    """Per-tree predictions, shape ``(n_trees, n_samples, n_outputs)``."""
    preds = tree_predictions(forest, X)
    return preds.reshape(preds.shape[0], preds.shape[1], -1)


def _oob_mean_var(forest, X, min_trees=2):
    """Out-of-bag mean and variance of the tree predictions at each training point."""
    preds = _tree_predictions_3d(forest, X)
    oob = np.ones(preds.shape[:2], dtype=bool)
    for t, in_bag in enumerate(forest.estimators_samples_):
        oob[t, in_bag] = False
    count = oob.sum(axis=0)
    keep = count >= min_trees
    if keep.sum() < max(2, X.shape[0] // 2):
        return None
    preds, w, count = preds[:, keep], oob[:, keep, None], count[keep, None]
    mean = (preds * w).sum(axis=0) / count
    var = (((preds - mean) ** 2) * w).sum(axis=0) / count
    return keep, mean, var


def _plain_forest(forest):
    """Unfitted copy of the nearest scikit-learn class of *forest*.

    Subclasses whose ``fit`` estimates the noise (e.g. the grid-search wrappers)
    would recurse if refitted here, so refit the plain scikit-learn forest.
    """
    base = next(c for c in type(forest).__mro__ if c.__module__.startswith("sklearn."))
    return base(**clone(forest).get_params())


def _kfold_mean_var(forest, X, y, cv):
    """K-fold mean and variance of the tree predictions at each training point."""
    n = X.shape[0]
    mean, var = np.empty((n, y.shape[1])), np.empty((n, y.shape[1]))
    folds = KFold(n_splits=min(cv, n), shuffle=True, random_state=0)
    for train, test in folds.split(X):
        y_train = y[train] if y.shape[1] > 1 else y[train, 0]
        fitted = _plain_forest(forest).fit(X[train], y_train)
        preds = _tree_predictions_3d(fitted, X[test])
        mean[test], var[test] = preds.mean(axis=0), preds.var(axis=0)
    return np.ones(n, dtype=bool), mean, var


def estimate_noise_variance(forest, X, y, cv=5):
    """Noise variance for a fitted forest (see module docstring).

    Returns a float for 1-D ``y`` and one value per column for 2-D ``y``.
    """
    X, y = np.asarray(X), np.asarray(y)
    y2 = y.reshape(y.shape[0], -1)
    if X.shape[0] < 4:
        noise = np.zeros(y2.shape[1])
    else:
        held_out = _oob_mean_var(forest, X) if getattr(forest, "bootstrap", False) else None
        if held_out is None:
            held_out = _kfold_mean_var(forest, X, y2, cv)
        keep, mean, var = held_out
        noise = np.maximum(np.mean((y2[keep] - mean) ** 2, axis=0) - np.mean(var, axis=0), 0.0)
    return float(noise[0]) if y.ndim == 1 else noise


class ForestSurrogate(BaseSurrogate):
    """Multi-output tree-ensemble surrogate with tree spread + noise std.

    Parameters
    ----------
    forest : RandomForestRegressor | ExtraTreesRegressor
        Unfitted single-output forest, wrapped per output.
    include_noise : bool, default True
        Add the held-out noise variance to the tree variance. ``False`` gives
        the tree spread alone (epistemic only; under-covers on noisy data).
    noise_cv : int, default 5
        Folds used to estimate the noise when the forest does not bootstrap.
    """

    def __init__(self, forest, include_noise=True, noise_cv=5):
        self.include_noise = include_noise
        self.noise_cv = noise_cv
        super().__init__(forest)

    def fit(self, X, Y):
        super().fit(X, Y)
        Y = np.asarray(Y)
        Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
        if self.include_noise:
            self.noise_var_ = np.array([
                estimate_noise_variance(est, X, Y[:, j], self.noise_cv)
                for j, est in enumerate(self.model.estimators_)
            ])
        else:
            self.noise_var_ = np.zeros(Y.shape[1])
        return self

    def predict(self, X, return_std=False):
        preds = self.model.predict(X)
        if not return_std:
            return preds
        tree_var = np.column_stack([
            tree_predictions(est, X).var(axis=0) for est in self.model.estimators_
        ])
        return preds, np.sqrt(tree_var + self.noise_var_)
