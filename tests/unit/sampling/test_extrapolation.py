# Copyright (c) 2026 takotime808

import numpy as np
import pytest

from multioutreg.sampling.extrapolation import ExtrapolationDetector


@pytest.fixture
def design():
    return np.random.default_rng(0).uniform(0, 1, size=(200, 2))


@pytest.mark.parametrize("method", ["knn", "mahalanobis", "hull", "box"])
def test_far_points_flagged_and_central_points_not(method, design):
    det = ExtrapolationDetector(method=method).fit(design)
    inside = np.array([[0.5, 0.5], [0.4, 0.6]])
    outside = np.array([[3.0, 3.0], [-2.0, 0.5]])
    assert not det.is_extrapolation(inside).any()
    assert det.is_extrapolation(outside).all()


def test_training_points_mostly_unflagged_at_the_quantile(design):
    det = ExtrapolationDetector(method="knn", quantile=0.95).fit(design)
    assert np.mean(det.train_scores_ > det.threshold_) <= 0.05 + 1e-9


def test_knn_catches_a_hole_inside_the_bounding_box():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, size=(400, 2))
    X = X[np.linalg.norm(X - 0.5, axis=1) > 0.3]  # empty disc in the middle
    centre = np.array([[0.5, 0.5]])
    assert ExtrapolationDetector(method="knn").fit(X).is_extrapolation(centre).all()
    # The box and hull checks cannot see a hole.
    assert not ExtrapolationDetector(method="box").fit(X).is_extrapolation(centre).any()
    assert not ExtrapolationDetector(method="hull").fit(X).is_extrapolation(centre).any()


def test_scores_grow_with_distance(design):
    det = ExtrapolationDetector().fit(design)
    pts = np.array([[1.2, 0.5], [2.0, 0.5], [4.0, 0.5]])
    assert np.all(np.diff(det.score(pts)) > 0)


def test_one_dimensional_and_constant_inputs():
    X = np.linspace(0, 1, 30)
    det = ExtrapolationDetector(method="box").fit(X)
    assert det.is_extrapolation(np.array([1.5])).all()
    Xc = np.column_stack([np.linspace(0, 1, 20), np.ones(20)])
    det = ExtrapolationDetector(method="knn").fit(Xc)
    assert np.all(np.isfinite(det.score(Xc)))


def test_fit_flag_and_bad_arguments(design):
    flags = ExtrapolationDetector().fit_flag(design, np.array([[0.5, 0.5], [9.0, 9.0]]))
    assert flags.tolist() == [False, True]
    with pytest.raises(ValueError):
        ExtrapolationDetector(method="lof")
    with pytest.raises(ValueError):
        ExtrapolationDetector(quantile=0.0)
