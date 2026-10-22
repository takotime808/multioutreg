# Copyright (c) 2026 takotime808
"""Shared pytest configuration."""

import importlib
import os
import sys
import types

import matplotlib

# Non-interactive backend: no plot windows, and plt.show() never blocks the run.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every matplotlib figure after each test."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Fast UMAP stand-in
# ---------------------------------------------------------------------------
# Importing umap takes over a minute per process (pynndescent compiles its
# numba distance functions at import), and UMAP is only ever used to draw a
# picture. So every test gets a fake ``umap`` module whose UMAP returns the
# first ``n_components`` input columns. Tests that need the real package use
# the ``real_umap`` fixture; set MOR_REAL_UMAP=1 to use it everywhere.


class _FastUMAP:
    """Drop-in for ``umap.UMAP`` that projects onto the first input columns."""

    def __init__(self, n_components=2, **kwargs):
        self.n_components = n_components

    def _project(self, X):
        X = np.asarray(X, dtype=float)
        out = np.zeros((X.shape[0], self.n_components))
        k = min(self.n_components, X.shape[1])
        out[:, :k] = X[:, :k]
        return out

    def fit(self, X, y=None):
        self.embedding_ = self._project(X)
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X).embedding_

    def transform(self, X):
        return self._project(X)


_FAKE_UMAP = types.ModuleType("umap")
_FAKE_UMAP.__path__ = []  # a package, so ``import umap.umap_`` resolves
_FAKE_UMAP.UMAP = _FastUMAP
_FAKE_UMAP.umap_ = types.ModuleType("umap.umap_")
_FAKE_UMAP.umap_.UMAP = _FastUMAP

if not os.environ.get("MOR_REAL_UMAP") and "umap" not in sys.modules:
    sys.modules["umap"] = _FAKE_UMAP
    sys.modules["umap.umap_"] = _FAKE_UMAP.umap_


@pytest.fixture
def real_umap(monkeypatch):
    """Use the real umap package for one test (slow: about a minute to import)."""
    if sys.modules.get("umap") is _FAKE_UMAP:
        monkeypatch.delitem(sys.modules, "umap")
        monkeypatch.delitem(sys.modules, "umap.umap_")
    real = importlib.import_module("umap")
    # Modules that ran ``import umap`` at top level hold the fake; repoint them.
    for mod in list(sys.modules.values()):
        if getattr(mod, "umap", None) is _FAKE_UMAP:
            monkeypatch.setattr(mod, "umap", real)
    return real
