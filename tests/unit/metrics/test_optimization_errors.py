# Copyright (c) 2026 takotime808
import numpy as np
import pytest

from multioutreg.metrics import design_error, goal_error, normalized_rmse, optimum_errors, total_error


def test_design_error_scales_to_unit_cube():
    bounds = [[-2, 2], [0, 10]]
    # (1, 5) -> (0.75, 0.5); (2, 10) -> (1, 1): distance sqrt(0.25^2 + 0.5^2) over sqrt(2)
    assert design_error([[2, 10]], [[1, 5]], bounds=bounds) == pytest.approx(np.hypot(0.25, 0.5) / np.sqrt(2))
    assert design_error([[0.2, 0.2]], [[0.2, 0.2]]) == 0.0


def test_design_error_uses_nearest_of_several_minimizers_or_a_distance():
    stars = [[0.0, 0.0], [1.0, 1.0]]
    assert design_error([[0.9, 1.0]], stars) == pytest.approx(0.1 / np.sqrt(2))
    face = lambda U: np.abs(U[:, 0])
    assert design_error([[0.4, 0.9]], distance=face) == pytest.approx(0.4 / np.sqrt(2))
    with pytest.raises(ValueError):
        design_error([[0.1, 0.1]])
    with pytest.raises(ValueError):
        design_error([[0.1, 0.1], [0.2, 0.2]], stars)
    with pytest.raises(ValueError):
        design_error([[0.1, 0.1]], [[0.1]])


def test_goal_total_and_rmse():
    assert goal_error(-6.0, -6.0, 14.0) == 0.0
    assert goal_error(4.0, -6.0, 14.0) == pytest.approx(0.5)
    assert total_error(0.3, 0.4) == pytest.approx(np.sqrt(0.125))
    y = np.array([0.0, 1.0, 2.0, 3.0])
    assert normalized_rmse(y, y + 0.2, 0.0, 4.0) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        goal_error(0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        normalized_rmse(y, y[:3], 0.0, 4.0)


def test_optimum_errors_bundle():
    e = optimum_errors([[0.5]], 0.0, -1.0, 1.0, x_star=[[0.5]])
    assert e == {"Ex": 0.0, "Ef": 0.5, "Et": pytest.approx(np.sqrt(0.125))}
