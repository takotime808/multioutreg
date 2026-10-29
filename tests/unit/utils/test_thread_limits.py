# Copyright (c) 2026 takotime808

import pytest

from multioutreg.utils import thread_limits as tl


@pytest.fixture
def no_cgroup(monkeypatch):
    monkeypatch.delenv(tl.ENV_VAR, raising=False)
    monkeypatch.setattr(tl, "_cgroup_cpu_limit", lambda: None)


def test_no_cap_without_quota_or_env(no_cgroup):
    assert tl.max_threads() is None
    assert tl.catboost_thread_kwargs() == {}
    assert tl.limit_native_threads() is None


def test_env_var_sets_cap(no_cgroup, monkeypatch):
    monkeypatch.setenv(tl.ENV_VAR, "3")
    assert tl.max_threads() == 3
    assert tl.catboost_thread_kwargs() == {"thread_count": 3}


def test_bad_env_var_is_ignored(no_cgroup, monkeypatch):
    monkeypatch.setenv(tl.ENV_VAR, "lots")
    assert tl.max_threads() is None


def test_fractional_quota_rounds_up(no_cgroup, monkeypatch):
    monkeypatch.setattr(tl, "_cgroup_cpu_limit", lambda: 1.5)
    monkeypatch.setattr(tl.os, "cpu_count", lambda: 64)
    assert tl.max_threads() == 2


def test_quota_at_or_above_core_count_is_no_cap(no_cgroup, monkeypatch):
    monkeypatch.setattr(tl, "_cgroup_cpu_limit", lambda: 8.0)
    monkeypatch.setattr(tl.os, "cpu_count", lambda: 8)
    assert tl.max_threads() is None


def test_catboost_surrogate_gets_thread_count(no_cgroup, monkeypatch):
    pytest.importorskip("catboost")
    from multioutreg.surrogates.catboost_sklearn import CatBoostSurrogate

    monkeypatch.setenv(tl.ENV_VAR, "2")
    est = CatBoostSurrogate(n_estimators=5).model.estimator
    assert est.get_params()["thread_count"] == 2
    # An explicit thread_count wins.
    est = CatBoostSurrogate(n_estimators=5, thread_count=1).model.estimator
    assert est.get_params()["thread_count"] == 1


def _fake_open(files):
    """An ``open`` that serves ``files`` ({path: text}) and raises OSError otherwise."""
    import io

    def fake(path, *args, **kwargs):
        if path not in files:
            raise FileNotFoundError(path)
        return io.StringIO(files[path])

    return fake


@pytest.mark.parametrize("files, expected", [
    ({"/sys/fs/cgroup/cpu.max": "150000 100000\n"}, 1.5),           # cgroup v2 quota
    ({"/sys/fs/cgroup/cpu.max": "max 100000\n"}, None),             # cgroup v2 unlimited
    ({"/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "200000\n",
      "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000\n"}, 2.0),    # cgroup v1 quota
    ({"/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "-1\n",
      "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000\n"}, None),   # cgroup v1 unlimited
    ({"/sys/fs/cgroup/cpu.max": "garbage\n"}, None),                # unparsable v2, no v1
    ({}, None),                                                     # no cgroup at all
])
def test_cgroup_cpu_limit_parses_v2_then_v1(monkeypatch, files, expected):
    monkeypatch.setattr(tl, "open", _fake_open(files), raising=False)
    assert tl._cgroup_cpu_limit() == expected


def test_limit_native_threads_applies_the_cap(no_cgroup, monkeypatch):
    import threadpoolctl

    calls = []
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", lambda limits: calls.append(limits))
    monkeypatch.setenv(tl.ENV_VAR, "2")
    monkeypatch.setenv("MKL_NUM_THREADS", "7")  # already set: left alone
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.delenv(var, raising=False)

    assert tl.limit_native_threads() == 2
    assert calls == [2]
    assert tl.os.environ["OMP_NUM_THREADS"] == "2"
    assert tl.os.environ["OPENBLAS_NUM_THREADS"] == "2"
    assert tl.os.environ["MKL_NUM_THREADS"] == "7"


def test_limit_native_threads_survives_threadpoolctl_failure(no_cgroup, monkeypatch):
    import threadpoolctl

    def boom(limits):
        raise RuntimeError("no native libs")

    monkeypatch.setattr(threadpoolctl, "threadpool_limits", boom)
    monkeypatch.setenv(tl.ENV_VAR, "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert tl.limit_native_threads() == 1
