# Copyright (c) 2026 takotime808

"""Conformal prediction for distribution-free prediction intervals."""

from multioutreg.conformal.split_conformal import SplitConformalPredictor
from multioutreg.conformal.cv_plus import CVPlusConformalPredictor
from multioutreg.conformal.metrics import (
    conformal_coverage,
    conformal_interval_width,
    conformal_summary,
    conditional_coverage,
)

__all__ = [
    "SplitConformalPredictor",
    "CVPlusConformalPredictor",
    "conformal_coverage",
    "conformal_interval_width",
    "conformal_summary",
    "conditional_coverage",
]
