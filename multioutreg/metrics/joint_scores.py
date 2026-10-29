# Copyright (c) 2026 takotime808

"""Proper scores for the joint predictive distribution of several outputs.

Per-output metrics (RMSE, NLL, coverage, per-output CRPS) cannot tell whether
a model captures how outputs vary *together*: a model that predicts every
output's marginal perfectly but treats them as independent scores the same as
one that gets their correlation right. The scores here are computed on the
whole output vector, so they can.

Both take samples from the predictive distribution, shape
``(n_rows, n_samples, n_outputs)``; :func:`gaussian_samples` draws them from a
mean and a per-row covariance (or std) as returned by
``predict(X, return_cov=True)`` / ``predict(X, return_std=True)``. Lower is
better for both. Outputs on very different scales dominate the Euclidean
distances, so standardize them (e.g. by the training std) before scoring when
comparing across outputs matters.
"""

import numpy as np


def _check(Y_true, samples):
    Y_true = np.asarray(Y_true, dtype=float)
    samples = np.asarray(samples, dtype=float)
    if Y_true.ndim == 1:
        Y_true = Y_true.reshape(-1, 1)
    if samples.ndim == 2:
        samples = samples[:, :, None]
    if samples.ndim != 3 or samples.shape[0] != Y_true.shape[0] or samples.shape[2] != Y_true.shape[1]:
        raise ValueError(
            "samples must have shape (n_rows, n_samples, n_outputs) matching Y_true "
            f"{Y_true.shape}; got {samples.shape}"
        )
    return Y_true, samples


def gaussian_samples(mean, cov, n_samples=500, seed=None):
    """Draw samples from per-row Gaussian predictive distributions.

    Parameters
    ----------
    mean : array-like of shape (n_rows, n_outputs)
    cov : array-like of shape (n_rows, n_outputs, n_outputs) or (n_rows, n_outputs)
        Per-row covariance, or per-output std (independent outputs).
    n_samples : int
    seed : int | np.random.Generator | None

    Returns
    -------
    np.ndarray of shape (n_rows, n_samples, n_outputs)
    """
    rng = np.random.default_rng(seed)
    mean = np.asarray(mean, dtype=float)
    if mean.ndim == 1:
        mean = mean.reshape(-1, 1)
    cov = np.asarray(cov, dtype=float)
    n, m = mean.shape
    eps = rng.standard_normal((n, n_samples, m))
    if cov.shape == (n, m):
        return mean[:, None, :] + eps * cov[:, None, :]
    if cov.shape != (n, m, m):
        raise ValueError(f"cov must have shape {(n, m)} or {(n, m, m)}; got {cov.shape}")
    # Eigen-decomposition rather than Cholesky: tolerates singular (rank-deficient)
    # covariances such as those from PCAOutputSurrogate with few components.
    w, V = np.linalg.eigh(cov)
    root = V * np.sqrt(np.clip(w, 0.0, None))[:, None, :]  # (n, m, m), root @ root.T = cov
    return mean[:, None, :] + np.einsum("nij,nsj->nsi", root, eps)


def energy_score(Y_true, samples, average=True):
    """Energy score (Gneiting & Raftery 2007), the multivariate CRPS.

    ``ES = E||X - y|| - 0.5 E||X - X'||`` with ``X, X'`` independent draws from
    the predictive distribution, estimated from the samples. It is proper, and
    reduces to the CRPS for a single output.

    Parameters
    ----------
    Y_true : array-like of shape (n_rows, n_outputs)
    samples : array-like of shape (n_rows, n_samples, n_outputs)
    average : bool
        Return the mean over rows (default) or the per-row scores.
    """
    Y_true, samples = _check(Y_true, samples)
    term1 = np.linalg.norm(samples - Y_true[:, None, :], axis=2).mean(axis=1)
    diffs = samples[:, :, None, :] - samples[:, None, :, :]  # (n, S, S, m)
    term2 = np.linalg.norm(diffs, axis=3).mean(axis=(1, 2))
    scores = term1 - 0.5 * term2
    return float(scores.mean()) if average else scores


def variogram_score(Y_true, samples, p=0.5, weights=None, average=True):
    """Variogram score of order ``p`` (Scheuerer & Hamill 2015).

    ``VS_p = sum_{i,j} w_ij (|y_i - y_j|^p - E|X_i - X_j|^p)^2`` over all output
    pairs. It compares the observed differences between outputs with those the
    predictive distribution expects, so unlike the energy score it is
    sensitive to a wrong correlation structure even when the marginals are
    right. It ignores errors that shift every output by the same amount, so
    report it alongside a marginal score.

    Parameters
    ----------
    Y_true : array-like of shape (n_rows, n_outputs)
    samples : array-like of shape (n_rows, n_samples, n_outputs)
    p : float
        Order; 0.5 is the usual choice and is the most robust to outliers.
    weights : array-like of shape (n_outputs, n_outputs) | None
        Pair weights ``w_ij``; all ones by default.
    average : bool
        Return the mean over rows (default) or the per-row scores.
    """
    Y_true, samples = _check(Y_true, samples)
    m = Y_true.shape[1]
    w = np.ones((m, m)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (m, m):
        raise ValueError(f"weights must have shape {(m, m)}; got {w.shape}")
    obs = np.abs(Y_true[:, :, None] - Y_true[:, None, :]) ** p  # (n, m, m)
    pred = (np.abs(samples[:, :, :, None] - samples[:, :, None, :]) ** p).mean(axis=1)
    scores = (w * (obs - pred) ** 2).sum(axis=(1, 2))
    return float(scores.mean()) if average else scores
