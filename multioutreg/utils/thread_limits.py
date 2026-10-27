# Copyright (c) 2026 takotime808

"""Cap native thread pools to the CPU a container is actually allowed to use.

OpenMP (LightGBM, XGBoost, HistGradientBoosting), BLAS and CatBoost size their
thread pools from the host's core count. In a CPU-limited container, such as
Streamlit Community Cloud, that means dozens of threads competing for one or
two cores: the grid search runs slower and the app gets throttled.
"""

from __future__ import annotations

import math
import os
from typing import Optional

ENV_VAR = "MULTIOUTREG_MAX_THREADS"


def _cgroup_cpu_limit() -> Optional[float]:
    """CPUs allowed by the cgroup (v2, then v1), or None when unlimited."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()[:2]
        if quota != "max":
            return int(quota) / int(period)
        return None
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        if quota > 0 and period > 0:
            return quota / period
    except (OSError, ValueError):
        pass
    return None


def max_threads() -> Optional[int]:
    """Thread cap from ``$MULTIOUTREG_MAX_THREADS`` or the cgroup CPU quota.

    None means no cap applies (e.g. a laptop or an unrestricted CI runner), so
    libraries keep their defaults.
    """
    env = os.environ.get(ENV_VAR, "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    cpus = _cgroup_cpu_limit()
    if cpus is None:
        return None
    n = max(1, math.ceil(cpus))
    return n if n < (os.cpu_count() or n + 1) else None


def catboost_thread_kwargs() -> dict:
    """``{"thread_count": n}`` under a cap, else ``{}``.

    CatBoost runs its own thread pool, which OpenMP/threadpoolctl limits don't
    reach, so the count has to be passed to the estimator.
    """
    n = max_threads()
    return {"thread_count": n} if n else {}


def limit_native_threads() -> Optional[int]:
    """Apply :func:`max_threads` to BLAS and OpenMP; return the cap (or None).

    OpenMP limits are per calling thread and Streamlit runs each script rerun
    in a fresh thread, so GUI scripts call this at the top of every run.
    """
    n = max_threads()
    if n is None:
        return None
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(n))  # for libraries not loaded yet
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=n)
    except Exception:
        pass
    return n
