# Copyright (c) 2026 takotime808
"""Active learning: acquisition functions, batch selection and sampling loops."""

from multioutreg.active.acquisition import (
    ACQUISITIONS,
    CommitteeSurrogate,
    aggregate,
    expected_improvement,
    max_variance,
    output_scale,
    predict_mean_std,
    probability_of_improvement,
    upper_confidence_bound,
)
from multioutreg.active.batch import select_batch
from multioutreg.active.benchmarks import BENCHMARKS, Benchmark, get_benchmark
from multioutreg.active.loop import ActiveLearner, heldout_metrics, space_filling
from multioutreg.active.multi_fidelity import (
    FidelityInfo,
    MultiFidelityActiveLearner,
    effective_costs,
    fidelity_info,
    level_trust,
    mf_expected_improvement,
    variance_reduction_per_cost,
)

__all__ = [
    "ACQUISITIONS",
    "CommitteeSurrogate",
    "aggregate",
    "expected_improvement",
    "max_variance",
    "output_scale",
    "predict_mean_std",
    "probability_of_improvement",
    "upper_confidence_bound",
    "select_batch",
    "BENCHMARKS",
    "Benchmark",
    "get_benchmark",
    "ActiveLearner",
    "heldout_metrics",
    "space_filling",
    "FidelityInfo",
    "MultiFidelityActiveLearner",
    "effective_costs",
    "fidelity_info",
    "level_trust",
    "mf_expected_improvement",
    "variance_reduction_per_cost",
]
