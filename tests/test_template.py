"""Tests for the pitch template."""

from __future__ import annotations

import numpy as np
import pytest

from football_vision.pitch.template import PitchTemplate


def test_landmark_count_and_known_values() -> None:
    """Checks the number of landmarks and the penalty-area coordinates."""
    landmarks = PitchTemplate().landmarks()
    assert len(landmarks) == 25
    assert landmarks["left_pa_outer_top"] == pytest.approx((16.5, 13.84))
    assert landmarks["left_ga_goalline_bottom"] == pytest.approx((0.0, 43.16))
    assert landmarks["center"] == pytest.approx((52.5, 34.0))


def test_right_half_mirrors_left_half() -> None:
    """Checks that right-side landmarks are mirrored copies of left-side ones."""
    template = PitchTemplate()
    landmarks = template.landmarks()
    for name, (x, y) in landmarks.items():
        if name.startswith("left_"):
            assert landmarks["right_" + name[5:]] == pytest.approx(
                (template.length - x, y)
            )


def test_landmark_array_order_and_unknown_name() -> None:
    """Checks the order of landmark_array and the error for unknown names."""
    template = PitchTemplate()
    np.testing.assert_allclose(
        template.landmark_array(["corner_br", "corner_tl"]), [[105.0, 68.0], [0.0, 0.0]]
    )
    with pytest.raises(KeyError):
        template.landmark_array(["not_a_landmark"])


def test_all_drawable_points_lie_inside_the_pitch() -> None:
    """Checks that sampled lines and arcs stay inside the pitch rectangle."""
    template = PitchTemplate()
    points = np.vstack(template.polylines())
    assert points.min() >= -1e-9
    assert points[:, 0].max() <= template.length + 1e-9
    assert points[:, 1].max() <= template.width + 1e-9
