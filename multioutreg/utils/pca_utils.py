# Copyright (c) 2026 takotime808

"""Input dimensionality reduction shared by the grid-search CLI and GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class InputPCAResult:
    """Output of :func:`fit_input_pca`."""

    X_train: np.ndarray
    X_test: np.ndarray
    transformer: Pipeline
    preview_pca: PCA
    n_components: int
    kaiser_k: int
    feature_names: List[str]


def fit_input_pca(
    X_train: np.ndarray,
    X_test: np.ndarray,
    method: Optional[str] = None,
    n_components: Optional[int] = None,
    threshold: Optional[float] = None,
) -> InputPCAResult:
    """Standardize inputs and project them onto principal components.

    The scaler and PCA are fitted on ``X_train`` only. Standardizing first means
    each input contributes on the same footing regardless of its units, and makes
    the Kaiser rule (keep eigenvalues > 1) meaningful: on standardized data the
    eigenvalues are those of the correlation matrix, so 1 is the variance of a
    single original input.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Training and test inputs.
    method : str, optional
        ``"Manual"`` (use ``n_components``), ``"Explained variance threshold"``
        (smallest k whose cumulative ratio reaches ``threshold``), or anything
        else for the Kaiser rule. ``"Manual"`` without ``n_components`` and the
        threshold method without ``threshold`` also fall back to Kaiser.
    n_components : int, optional
        Number of components for the manual method.
    threshold : float, optional
        Cumulative explained-variance target in (0, 1].

    Returns
    -------
    InputPCAResult
        Transformed arrays, the fitted scaler+PCA pipeline, the full preview PCA
        (for scree plots), the chosen k and the Kaiser suggestion.
    """
    scaler = StandardScaler().fit(X_train)
    preview_pca = PCA().fit(scaler.transform(X_train))
    kaiser_k = int(np.sum(preview_pca.explained_variance_ > 1))
    n_max = len(preview_pca.explained_variance_ratio_)

    if method == "Manual" and n_components is not None:
        k = int(n_components)
    elif method == "Explained variance threshold" and threshold is not None:
        cum = np.cumsum(preview_pca.explained_variance_ratio_)
        k = int(np.searchsorted(cum, threshold) + 1)
    else:
        k = kaiser_k
    k = min(max(1, k), n_max)

    transformer = make_pipeline(StandardScaler(), PCA(n_components=k))
    X_train_t = transformer.fit_transform(X_train)
    X_test_t = transformer.transform(X_test)
    return InputPCAResult(
        X_train=X_train_t,
        X_test=X_test_t,
        transformer=transformer,
        preview_pca=preview_pca,
        n_components=k,
        kaiser_k=kaiser_k,
        feature_names=[f"PC{i + 1}" for i in range(k)],
    )
