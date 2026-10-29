# Copyright (c) 2026 takotime808

"""Save and load fitted surrogates together with what is needed to reuse them.

A fitted model on its own is not enough to reuse it later: you also need the
input columns it expects (in order), the names of its outputs, the fidelity
levels of a multi-fidelity model, and the package versions it was fitted
with (a pickle from one scikit-learn version may not load, or may predict
differently, under another). :func:`save_model` stores all of these in one
``joblib`` file and :func:`load_model` returns them as a :class:`ModelBundle`.

The file layout uses the same keys as the Grid Search app's
"Download Trained Model" bundle (``input_columns``, ``output_columns``,
``versions``), and :func:`load_model` also opens those files and plain
``joblib.dump(model)`` files.

.. warning::
   ``joblib`` files are pickles: loading one runs code from it. Only load
   files you created or trust.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np

FORMAT = "multioutreg.model_bundle"
FORMAT_VERSION = 1

# Versions recorded at save time and compared at load time.
_CHECKED_PACKAGES = ("multioutreg", "scikit-learn", "numpy")
_RECORDED_PACKAGES = _CHECKED_PACKAGES + ("joblib", "torch", "gpytorch")


def package_versions() -> Dict[str, str]:
    """Installed versions of Python and the packages a saved model depends on."""
    versions = {"python": sys.version.split()[0]}
    for pkg in _RECORDED_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


@dataclass
class ModelBundle:
    """A fitted model plus the information needed to use it.

    ``predict`` accepts a DataFrame (columns are selected and ordered by
    ``input_columns``) or an array (used as is), and passes keyword arguments
    such as ``return_std=True`` through to the model.
    """

    model: Any
    input_columns: Optional[List[str]] = None
    output_columns: Optional[List[str]] = None
    fidelity_levels: Optional[List[str]] = None
    versions: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    conformal: Optional[Dict[str, Any]] = None

    def _prepare(self, X):
        if hasattr(X, "columns") and self.input_columns is not None:
            missing = [c for c in self.input_columns if c not in X.columns]
            if missing:
                raise KeyError(f"input columns missing from X: {missing}")
            return X[self.input_columns].to_numpy()
        return X

    def predict(self, X, as_frame: bool = False, **kwargs):
        """Predict with the stored model.

        With ``as_frame=True`` every returned array is a DataFrame with
        ``output_columns`` as its columns (requires pandas and output names).
        """
        out = self.model.predict(self._prepare(X), **kwargs)
        if not as_frame:
            return out
        import pandas as pd

        index = X.index if hasattr(X, "index") else None
        cols = self.output_columns

        def frame(a):
            a = np.asarray(a)
            return pd.DataFrame(a.reshape(len(a), -1), columns=cols, index=index)

        return tuple(frame(a) for a in out) if isinstance(out, tuple) else frame(out)


def _infer_input_columns(model):
    names = getattr(model, "feature_names_in_", None)
    return None if names is None else [str(n) for n in names]


def save_model(
    model: Any,
    path,
    input_columns: Optional[Sequence[str]] = None,
    output_columns: Optional[Sequence[str]] = None,
    fidelity_levels: Optional[Sequence[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    compress: int = 3,
):
    """Save a fitted model with its column names, fidelity levels and versions.

    Parameters
    ----------
    model : object
        Any fitted surrogate, pipeline or estimator.
    path : str | os.PathLike | file-like
        Destination (``.joblib`` by convention).
    input_columns, output_columns : sequence of str, optional
        Feature and output names. ``input_columns`` defaults to the model's
        ``feature_names_in_`` when it has one.
    fidelity_levels : sequence of str, optional
        Defaults to the model's ``fidelity_levels`` attribute when it has one
        (``MultiFidelitySurrogate``, ``StackedVFMSurrogate``).
    metadata : dict, optional
        Anything else to keep with the model (training data source, metrics,
        the configuration that produced it).
    compress : int
        ``joblib`` compression level, 0-9.

    Returns
    -------
    The list of file names written, as returned by ``joblib.dump``.
    """
    if input_columns is None:
        input_columns = _infer_input_columns(model)
    if fidelity_levels is None:
        fidelity_levels = getattr(model, "fidelity_levels", None)
    bundle = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "model": model,
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "input_columns": None if input_columns is None else [str(c) for c in input_columns],
        "output_columns": None if output_columns is None else [str(c) for c in output_columns],
        "fidelity_levels": None if fidelity_levels is None else [str(f) for f in fidelity_levels],
        "versions": package_versions(),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metadata": dict(metadata or {}),
    }
    return joblib.dump(bundle, path, compress=compress)


def _warn_on_version_mismatch(saved: Dict[str, str]):
    current = package_versions()
    diffs = [
        f"{pkg} {saved[pkg]} (now {current[pkg]})"
        for pkg in _CHECKED_PACKAGES
        if pkg in saved and saved[pkg] != current.get(pkg)
    ]
    if diffs:
        warnings.warn(
            "Model was saved with different package versions: " + ", ".join(diffs)
            + ". Predictions may differ; refit if possible.",
            UserWarning,
            stacklevel=3,
        )


def load_model(path, check_versions: bool = True) -> ModelBundle:
    """Load a model saved by :func:`save_model`, the Grid Search app, or ``joblib.dump``.

    Only load files you trust: ``joblib`` files are pickles.

    Parameters
    ----------
    path : str | os.PathLike | file-like
    check_versions : bool
        Warn if ``multioutreg``, scikit-learn or numpy differ from the
        versions the model was saved with.
    """
    obj = joblib.load(path)
    if isinstance(obj, dict) and obj.get("format") == FORMAT:
        if obj.get("format_version", 0) > FORMAT_VERSION:
            raise ValueError(
                f"{path!r} uses bundle format {obj['format_version']}, newer than this "
                f"multioutreg supports ({FORMAT_VERSION}); upgrade multioutreg."
            )
        bundle = ModelBundle(
            model=obj["model"],
            input_columns=obj.get("input_columns"),
            output_columns=obj.get("output_columns"),
            fidelity_levels=obj.get("fidelity_levels"),
            versions=obj.get("versions", {}),
            metadata={**obj.get("metadata", {}), "saved_at": obj.get("saved_at")},
            conformal=obj.get("metadata", {}).get("conformal"),
        )
    elif isinstance(obj, dict) and "pipeline" in obj:
        # Grid Search app download (multioutreg.gui.grid_search_core.build_model_bundle).
        bundle = ModelBundle(
            model=obj["pipeline"],
            input_columns=obj.get("input_columns"),
            output_columns=obj.get("output_columns"),
            versions=obj.get("versions", {}),
            metadata={"config": obj.get("config")},
            conformal=obj.get("conformal"),
        )
    else:
        bundle = ModelBundle(model=obj)
    if check_versions and bundle.versions:
        _warn_on_version_mismatch(bundle.versions)
    return bundle
