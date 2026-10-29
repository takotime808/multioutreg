# Copyright (c) 2026 takotime808

"""Run slow plotting work in worker processes so it doesn't hold up the app.

Streamlit runs a page top to bottom in one thread, so a 10 s pair plot or a
report full of SHAP / PDP / UMAP figures would stall model fitting and every
result after it. :func:`submit` hands such work to a process pool and returns a
:class:`BackgroundJob` the page keeps in ``st.session_state`` and checks later.

Processes, not threads: pyplot keeps global "current figure" state (the report
helpers save whatever figure is current), and plotting is pure Python, so a
thread would neither be safe nor run in parallel under the GIL.

Each pool has a single worker: every worker re-imports numpy, sklearn and
matplotlib (the report worker also SHAP and UMAP), and Streamlit Community
Cloud allows as little as 690 MB (see ``docs/parallel_computing.md``).

There are two pools. ``"plots"`` starts light, so a pair plot begins drawing
within a second or two. ``"report"`` imports the report modules and warms up UMAP
(about 30 s to import and compile) when its workers start; call
:func:`warm_up` early so that cost is paid while the user is still choosing
columns.

Set ``MOR_BACKGROUND_JOBS=0`` to run every job inline instead (the test suite
does this; see ``tests/conftest.py``).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import threading
import types
from contextlib import contextmanager
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, Dict, Iterator

_LOCK = threading.Lock()
_EXECUTORS: Dict[str, ProcessPoolExecutor] = {}
_NOT_RUN = object()
_WORKERS = {"plots": 1, "report": 1}


def background_enabled() -> bool:
    """False when ``MOR_BACKGROUND_JOBS`` is set to 0/false/no/off."""
    return os.environ.get("MOR_BACKGROUND_JOBS", "1").strip().lower() not in {
        "0", "false", "no", "off"}


def _init_worker(pool: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    try:  # load lightgbm's OpenMP runtime before torch's (segfaults otherwise on some Macs)
        import lightgbm  # noqa: F401
    except ImportError:
        pass
    if pool == "report":
        import multioutreg.gui.grid_search_core  # noqa: F401  (pulls in SHAP, ...)
        _warm_umap()


def _warm_umap() -> None:
    """Import UMAP and run a tiny fit, so numba compiles it before the first report.

    ``generate_umap_plot`` imports UMAP lazily; left until the report, the
    import (~20 s) and first-fit compile (~10 s) land after "Run Grid Search".
    """
    try:
        import numpy as np
        import umap

        X = np.random.default_rng(0).uniform(size=(30, 3))
        umap.UMAP(n_components=2, n_neighbors=5, random_state=0).fit_transform(X)
    except Exception:
        pass


@contextmanager
def _without_main_module() -> Iterator[None]:
    """Stop new spawn workers from re-running the ``__main__`` module.

    A spawned worker imports the parent's ``__main__`` before it runs anything.
    Streamlit registers the page script as ``__main__``, so every worker would
    run the whole page (in bare mode). Jobs are functions from real modules and
    never need it, so workers started in this block get an empty stand-in.
    Workers start inside ``submit`` (spawn pools start them on demand), so
    callers hold ``_LOCK`` and wrap the submit.
    """
    real = sys.modules.get("__main__")
    stub = types.ModuleType("__main__")
    sys.modules["__main__"] = stub
    try:
        yield
    finally:
        if sys.modules.get("__main__") is stub:
            sys.modules["__main__"] = real


def _executor(pool: str) -> ProcessPoolExecutor:
    ex = _EXECUTORS.get(pool)
    if ex is None:
        # spawn, not fork: forking a process that has loaded OpenMP / torch can deadlock.
        ex = ProcessPoolExecutor(
            max_workers=_WORKERS.get(pool, 1),
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(pool,),
        )
        _EXECUTORS[pool] = ex
    return ex


def _submit(pool: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
    with _LOCK, _without_main_module():
        return _executor(pool).submit(fn, *args, **kwargs)


def _discard(pool: str) -> None:
    with _LOCK:
        ex = _EXECUTORS.pop(pool, None)
    if ex is not None:
        ex.shutdown(wait=False, cancel_futures=True)


def _noop() -> None:
    return None


def warm_up(pool: str = "report") -> None:
    """Start *pool*'s worker processes now so their imports overlap with other work."""
    if background_enabled():
        try:
            _submit(pool, _noop)
        except Exception:
            _discard(pool)


class BackgroundJob:
    """Result of ``fn(*args, **kwargs)``, computed in a worker process when possible.

    If the worker can't run it (the arguments don't pickle, the pool died),
    :meth:`result` runs ``fn`` inline instead, so a job always gives the same
    answer as a direct call, including raising the same exception.
    """

    def __init__(self, pool: str, fn: Callable[..., Any], *args: Any, **kwargs: Any):
        self._pool, self._fn, self._args, self._kwargs = pool, fn, args, kwargs
        self._result: Any = _NOT_RUN
        self._future: Future | None = None
        if background_enabled():
            try:
                self._future = _submit(pool, fn, *args, **kwargs)
            except Exception:
                _discard(pool)

    def done(self) -> bool:
        """True once :meth:`result` would return without waiting on a worker."""
        return self._result is not _NOT_RUN or self._future is None or self._future.done()

    def result(self) -> Any:
        """Return the job's value, waiting for the worker if it is still running."""
        if self._result is _NOT_RUN:
            value = _NOT_RUN
            if self._future is not None:
                try:
                    value = self._future.result()
                except BrokenProcessPool:
                    _discard(self._pool)
                except Exception:
                    # Run inline below: either the worker couldn't run it at all
                    # (e.g. a model that doesn't pickle), or it fails here too
                    # with the real error.
                    pass
            if value is _NOT_RUN:
                value = self._fn(*self._args, **self._kwargs)
            self._result = value
            # Drop the inputs (training data, fitted model) once they're used.
            self._fn = self._args = self._kwargs = self._future = None
        return self._result


def submit(pool: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> BackgroundJob:
    """Start ``fn(*args, **kwargs)`` in *pool* ("plots" or "report") and return the job."""
    return BackgroundJob(pool, fn, *args, **kwargs)
