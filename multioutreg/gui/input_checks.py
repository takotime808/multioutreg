# Copyright (c) 2026 takotime808

"""Checks on the chosen columns, run before any model is fitted.

:func:`check_run_inputs` returns the problems that would make a grid-search run
fail or give meaningless results, so the page can show them in the form and
stop instead of fitting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import pandas as pd

from multioutreg.model_selection.evaluation import (
    N_OUTER_FOLDS,
    min_calibration_points,
    plan_evaluation,
)
from multioutreg.utils.imputation import drop_unusable_rows

#: Fewest usable rows for a run: two per outer fold of the nested CV that
#: small datasets get, so every outer training fold can still be cross-validated
#: for model selection. Below this the pipeline may run, but its scores are noise.
MIN_ROWS = 2 * N_OUTER_FOLDS


@dataclass(frozen=True)
class InputProblem:
    """One reason not to start the run. ``kind`` names the check."""

    kind: str      # "overlap", "non_numeric", "too_few_rows" or "conformal_rows"
    message: str


def _names(cols: Sequence[str]) -> str:
    return ", ".join(f"'{c}'" for c in cols)


def check_run_inputs(
    df: pd.DataFrame,
    input_cols: Sequence[str],
    output_cols: Sequence[str],
    *,
    conformal_alpha: Optional[float] = None,
    drop_row_cols: Sequence[str] = (),
) -> List[InputProblem]:
    """Problems with the chosen columns, in the order the page shows them.

    ``conformal_alpha`` is the miscoverage level when conformal intervals are
    requested (``None`` otherwise). ``drop_row_cols`` are the columns whose
    missing values the user chose to handle by dropping rows; rows missing an
    output are always dropped. An empty list means the run can start.
    """
    problems: List[InputProblem] = []

    both = [c for c in input_cols if c in set(output_cols)]
    if both:
        problems.append(InputProblem(
            "overlap",
            f"{_names(both)} {'is' if len(both) == 1 else 'are'} chosen as both an input "
            "and an output. A column can't predict itself; remove it from one of the lists."))

    chosen = list(dict.fromkeys(list(input_cols) + list(output_cols)))
    non_numeric = [c for c in chosen if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        examples = []
        for c in non_numeric[:3]:
            bad = df[c][pd.to_numeric(df[c], errors="coerce").isna() & df[c].notna()]
            if len(bad):
                examples.append(f"'{c}' has values like {bad.iloc[0]!r}")
        problems.append(InputProblem(
            "non_numeric",
            f"Non-numeric column(s): {_names(non_numeric)}. Every input and output must be "
            "numeric; encode categories as numbers or leave these columns out."
            + (f" ({'; '.join(examples)}.)" if examples else "")))

    # Row counts only mean something once the columns themselves are usable.
    if problems:
        return problems

    n = len(drop_unusable_rows(df, list(output_cols), list(drop_row_cols)))
    dropped = len(df) - n
    lost = f" ({dropped} of {len(df)} rows are dropped for missing values)" if dropped else ""
    if n < MIN_ROWS:
        problems.append(InputProblem(
            "too_few_rows",
            f"Only {n} usable row(s){lost}; at least {MIN_ROWS} are needed. With fewer, "
            "there is not enough data to hold out test rows and still cross-validate "
            "the model selection, so the scores would be noise."))
        return problems

    if conformal_alpha is not None:
        plan = plan_evaluation(n, alpha=conformal_alpha)
        need = min_calibration_points(conformal_alpha)
        if plan.scheme == "nested_cv":
            k = plan.outer_cv.n_splits
            n_fit = n - math.ceil(n / k)
        else:
            n_fit = n - math.ceil(plan.test_size * n)
        if n_fit < need:
            larger = next((a for a in (0.2, 0.25, 0.3, 0.4, 0.5)
                           if a > conformal_alpha and min_calibration_points(a) <= n_fit), None)
            fix = (f"raise the conformal alpha to {larger} or more" if larger
                   else "turn conformal intervals off")
            problems.append(InputProblem(
                "conformal_rows",
                f"Conformal intervals at alpha={conformal_alpha} need at least {need} "
                f"training rows to be finite, but only about {n_fit} of the {n} usable "
                f"rows{lost} are left for training. Add data, or {fix} under Advanced "
                "Settings."))
    return problems
