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
