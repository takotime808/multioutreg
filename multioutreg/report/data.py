# Copyright (c) 2026 takotime808

"""The data behind a redesigned (v2) report.

:class:`ReportData` holds everything the v2 template and its charts need, as
plain dicts and lists, so it can be written into the report as JSON and read
back (``ReportData.from_json``) to rebuild the report without retraining.
:func:`multioutreg.report.build.build_report_data` fills it.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

import numpy as np

#: Scatter data above this many points per output is downsampled in the JSON.
MAX_POINTS = 2000
SCHEMA_VERSION = 1


@dataclass
class ReportData:
    """Everything shown in a v2 report.

    ``run`` holds metadata (title, model, data hash, seed, version, date, CV
    description, sizes). ``outputs`` has one dict per output with its metrics,
    status, derived series (coverage curves, interval widths, residual trend)
    and the (possibly downsampled) test arrays. The other fields are ``None`` or
    empty when the caller had nothing to put there, and the matching section is
    hidden or marked as skipped.
    """

    run: Dict[str, Any]
    outputs: List[Dict[str, Any]]
    thresholds: Dict[str, float]
    verdict: Dict[str, Any] = field(default_factory=dict)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    next_steps: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty_metrics: Dict[str, float] = field(default_factory=dict)
    conformal_alpha: Optional[float] = None
    levels: List[Dict[str, Any]] = field(default_factory=list)
    baseline_label: Optional[str] = None
    cv: Optional[Dict[str, Any]] = None
    shap: Optional[Dict[str, Any]] = None
    pdp: Optional[Dict[str, Any]] = None
    imputation: Optional[Dict[str, Any]] = None
    regret: Optional[Dict[str, Any]] = None
    pca: Optional[Dict[str, Any]] = None
    sampling: Optional[Dict[str, Any]] = None
    figures: List[Dict[str, Any]] = field(default_factory=list)
    skipped: List[Dict[str, str]] = field(default_factory=list)
    summary_points: List[str] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self, include_images: bool = True) -> Dict[str, Any]:
        """Plain-Python copy: numpy converted, NaN/inf as ``None``, floats in
        arrays rounded to 4 significant figures."""
        d = _clean(asdict(self))
        if not include_images:
            d["figures"] = [{k: v for k, v in f.items() if k != "img_b64"} for f in d["figures"]]
            if d.get("pca"):
                d["pca"] = {k: v for k, v in d["pca"].items() if k != "plot_b64"}
            if d.get("sampling"):
                d["sampling"] = {k: v for k, v in d["sampling"].items() if k != "plots"}
        return d

    def to_json(self, include_images: bool = True, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(include_images), allow_nan=False, indent=indent,
                          ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReportData":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @classmethod
    def from_json(cls, text: str) -> "ReportData":
        return cls.from_dict(json.loads(text))


def _round(x: float) -> Optional[float]:
    if not math.isfinite(x):
        return None
    return float(f"{x:.4g}")


def _clean(obj: Any, in_array: bool = False) -> Any:
    """Make ``obj`` JSON-safe. Floats inside lists are rounded (they are chart
    data); scalar floats keep full precision."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v, in_array=True) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v, in_array=True) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if not math.isfinite(x):
            return None
        return _round(x) if in_array else x
    return obj


def downsample_index(n: int, cap: int = MAX_POINTS, seed: int = 0) -> np.ndarray:
    """Sorted indices of at most ``cap`` of ``n`` points (all of them if n <= cap)."""
    if n <= cap:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, cap, replace=False))
