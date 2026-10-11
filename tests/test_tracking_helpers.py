"""Tests for the implemented tracking helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_vision.config import TrackingConfig
from football_vision.tracking.association import build_iou_cost
from football_vision.tracking.base import make_tracker
from football_vision.tracking.bytetrack import ByteTrackTracker
from football_vision.tracking.kalman import constant_velocity_matrices
from football_vision.tracking.sort import SortTracker
from football_vision.tracking.stitching import drop_short_tracks


def test_constant_velocity_matrices() -> None:
    """Checks shapes, the motion model, symmetry and positive semi-definiteness of Q."""
    f, m, q, r = constant_velocity_matrices(
        dt=0.1, process_std=0.5, measurement_std=0.3
    )
    assert (
        f.shape == (4, 4)
        and m.shape == (2, 4)
        and q.shape == (4, 4)
        and r.shape == (2, 2)
    )
    np.testing.assert_allclose(f @ [1.0, 2.0, 3.0, 4.0], [1.3, 2.4, 3.0, 4.0])
    np.testing.assert_allclose(m @ [1.0, 2.0, 3.0, 4.0], [1.0, 2.0])
    np.testing.assert_allclose(q, q.T)
    assert np.linalg.eigvalsh(q).min() >= -1e-12
    np.testing.assert_allclose(r, 0.09 * np.eye(2))


def test_iou_cost_is_one_minus_iou() -> None:
    """Checks the association cost between tracks and detections."""
    cost = build_iou_cost([[0, 0, 2, 2]], [[0, 0, 2, 2], [10, 10, 12, 12]])
    np.testing.assert_allclose(cost, [[0.0, 1.0]])


def test_drop_short_tracks() -> None:
    """Checks that tracks below the minimum length are removed."""
    tracks = pd.DataFrame({"track_id": [1, 1, 1, 2], "frame_id": [0, 1, 2, 0]})
    assert drop_short_tracks(tracks, 2)["track_id"].tolist() == [1, 1, 1]
    assert len(drop_short_tracks(tracks.iloc[0:0], 2)) == 0


def test_tracker_factory() -> None:
    """Checks the tracker factory and its error for unknown trackers."""
    assert isinstance(make_tracker(TrackingConfig(tracker="sort"), 25.0), SortTracker)
    assert isinstance(
        make_tracker(TrackingConfig(tracker="bytetrack"), 25.0), ByteTrackTracker
    )
    with pytest.raises(ValueError):
        make_tracker(TrackingConfig(tracker="nope"), 25.0)  # type: ignore[arg-type]


def test_sort_reset_clears_state() -> None:
    """Checks that reset restarts id numbering."""
    tracker = SortTracker(TrackingConfig(tracker="sort"), 25.0)
    tracker._next_id = 9
    tracker.reset()
    assert tracker._next_id == 1
