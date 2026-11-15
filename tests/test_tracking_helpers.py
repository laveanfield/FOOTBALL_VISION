"""Tests for the implemented tracking helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_vision.artifacts import TRACK_COLUMNS
from football_vision.config import TrackingConfig
from football_vision.datatypes import Detections
from football_vision.tracking.association import (
    build_iou_cost,
    build_pitch_distance_cost,
    solve_assignment,
)
from football_vision.tracking.base import make_tracker
from football_vision.tracking.bytetrack import ByteTrackTracker
from football_vision.tracking.kalman import KalmanFilter, constant_velocity_matrices
from football_vision.tracking.sort import SortTracker
from football_vision.tracking.stitching import drop_short_tracks, stitch_tracklets


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


def test_kalman_edge_cases_and_mahalanobis_scale() -> None:
    """Checks empty gating inputs, exact Mahalanobis scaling, and dimension errors."""
    f, m, q, r = constant_velocity_matrices(
        dt=0.0, process_std=0.0, measurement_std=2.0
    )
    kalman = KalmanFilter(f, m, q, r, x0=[0.0, 0.0, 0.0, 0.0], p0=np.zeros((4, 4)))
    assert kalman.gating_distance(np.empty((0, 2))).shape == (0,)
    # With P=0 and R=4*I, distance^2 to [2.0, 0.0] equals 2^2 / 4 = 1.0.
    np.testing.assert_allclose(kalman.gating_distance([[2.0, 0.0]]), [1.0])
    with pytest.raises(ValueError):
        kalman.update([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        kalman.gating_distance([[1.0, 2.0, 3.0]])


def test_solve_assignment_edge_cases() -> None:
    """Checks empty columns, inclusive boundary gate, and +inf blocked costs."""
    assert solve_assignment(np.zeros((2, 0)), 0.5) == ([], [0, 1], [])
    assert solve_assignment(np.zeros((0, 0)), 0.5) == ([], [], [])
    cost = np.array([[0.5, np.inf], [np.inf, 0.2]])
    matches, unmatched_rows, unmatched_cols = solve_assignment(cost, max_cost=0.5)
    assert matches == [(0, 0), (1, 1)]
    assert unmatched_rows == [] and unmatched_cols == []
    # Infeasible row forced to match +inf gets rejected cleanly.
    cost_infeasible = np.array([[0.1, np.inf], [0.2, np.inf]])
    matches, unmatched_rows, unmatched_cols = solve_assignment(
        cost_infeasible, max_cost=0.5
    )
    assert matches == [(0, 0)]
    assert unmatched_rows == [1]
    assert unmatched_cols == [1]


def test_build_pitch_distance_cost() -> None:
    """Checks Euclidean distance in meters and inf masking for non-finite inputs."""
    cost = build_pitch_distance_cost([[0.0, 0.0]], [[3.0, 4.0], [np.nan, 0.0]])
    assert cost[0, 0] == pytest.approx(5.0)
    assert np.isinf(cost[0, 1])


def test_sort_tracker_keeps_ids_and_expires_after_buffer() -> None:
    """Checks image-space SORT tracking continuity and buffer expiration."""
    tracker = SortTracker(
        TrackingConfig(tracker="sort", space="image", track_buffer=2, match_thresh=0.8),
        fps=25.0,
    )
    for step in range(4):
        dx = step * 8.0
        dets = Detections(
            boxes_xyxy=np.array(
                [[100 + dx, 100, 140 + dx, 200], [300 + dx, 120, 340 + dx, 220]],
                dtype=np.float32,
            ),
            scores=np.array([0.9, 0.85], dtype=np.float32),
            class_ids=np.array([0, 0], dtype=np.int64),
        )
        out = tracker.update(dets, frame_id=step)
        assert out.track_ids is not None
        assert out.track_ids.tolist() == [1, 2]

    # Empty frames past track_buffer=2 expire existing tracks.
    for gap in range(3):
        empty_out = tracker.update(Detections.empty(), frame_id=4 + gap)
        assert len(empty_out) == 0

    reappear = Detections(
        boxes_xyxy=np.array([[132, 100, 172, 200]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        class_ids=np.array([0], dtype=np.int64),
    )
    out_new = tracker.update(reappear, frame_id=7)
    assert out_new.track_ids is not None
    assert out_new.track_ids.tolist() == [3]


def test_sort_tracker_pitch_space_handles_missing_homography() -> None:
    """Checks pitch-space SORT tracking and fail-soft behavior when H is None."""
    tracker = SortTracker(
        TrackingConfig(tracker="sort", space="pitch", track_buffer=3, match_thresh=1.5),
        fps=25.0,
    )
    homography = np.diag([0.1, 0.1, 1.0])
    det0 = Detections(
        boxes_xyxy=np.array([[100, 100, 140, 200]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        class_ids=np.array([0], dtype=np.int64),
    )
    out0 = tracker.update(det0, frame_id=0, homography=homography)
    assert out0.track_ids is not None and out0.track_ids.tolist() == [1]

    # Invalid homography frame returns empty detections without losing track state.
    det1 = Detections(
        boxes_xyxy=np.array([[104, 100, 144, 200]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        class_ids=np.array([0], dtype=np.int64),
    )
    out1 = tracker.update(det1, frame_id=1, homography=None)
    assert len(out1) == 0

    det2 = Detections(
        boxes_xyxy=np.array([[108, 100, 148, 200]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        class_ids=np.array([0], dtype=np.int64),
    )
    out2 = tracker.update(det2, frame_id=2, homography=homography)
    assert out2.track_ids is not None and out2.track_ids.tolist() == [1]


def test_bytetrack_recovers_low_score_and_ignores_low_noise() -> None:
    """Checks ByteTrack round-2 recovery and suppression of low-score spawns."""
    tracker = ByteTrackTracker(
        TrackingConfig(
            tracker="bytetrack",
            high_thresh=0.5,
            low_thresh=0.1,
            match_thresh=0.8,
            track_buffer=5,
        ),
        fps=25.0,
    )
    det0 = Detections(
        boxes_xyxy=np.array([[100, 100, 140, 200]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        class_ids=np.array([0], dtype=np.int64),
    )
    out0 = tracker.update(det0, frame_id=0)
    assert out0.track_ids is not None and out0.track_ids.tolist() == [1]

    # Tracked player drops to 0.25 (recovered); distant 0.20 noise must not spawn.
    det1 = Detections(
        boxes_xyxy=np.array(
            [[104, 100, 144, 200], [500, 300, 540, 400]], dtype=np.float32
        ),
        scores=np.array([0.25, 0.20], dtype=np.float32),
        class_ids=np.array([0, 0], dtype=np.int64),
    )
    out1 = tracker.update(det1, frame_id=1)
    assert out1.track_ids is not None and out1.track_ids.tolist() == [1]
    assert len(out1) == 1

    tracker.reset()
    out_reset = tracker.update(det0, frame_id=2)
    assert out_reset.track_ids is not None and out_reset.track_ids.tolist() == [1]


def test_stitch_tracklets_bridges_chains_and_respects_gates() -> None:
    """Checks temporal gap, spatial distance, overlap, and team gating in stitching."""
    rows = [
        # Chain 1 -> 2 -> 3 across small gaps & short spatial jumps (team=0).
        (0, 1, 90.0, 100.0, 110.0, 200.0, 0.9, 0, "player"),
        (2, 1, 95.0, 100.0, 115.0, 200.0, 0.9, 0, "player"),
        (5, 2, 105.0, 100.0, 125.0, 200.0, 0.8, 0, "player"),
        (7, 2, 112.0, 100.0, 132.0, 200.0, 0.8, 0, "player"),
        (10, 3, 120.0, 100.0, 140.0, 200.0, 0.85, 0, "player"),
        # Overlapping track 4 must never merge with track 1.
        (1, 4, 92.0, 100.0, 112.0, 200.0, 0.9, 0, "player"),
        (6, 4, 100.0, 100.0, 120.0, 200.0, 0.9, 0, "player"),
        # Conflicting team track 5 close in space/time must not merge with track 1/2/3.
        (13, 5, 125.0, 100.0, 145.0, 200.0, 0.9, 1, "player"),
        # Track 6 beyond max_gap_frames must stay separate.
        (45, 6, 125.0, 100.0, 145.0, 200.0, 0.9, 0, "player"),
    ]
    df = pd.DataFrame(rows, columns=list(TRACK_COLUMNS))
    stitched = stitch_tracklets(df, max_gap_frames=10, max_distance_px=50.0)
    assert len(stitched) == len(df)
    chain_mask = stitched["frame_id"].isin([0, 2, 5, 7, 10])
    assert stitched.loc[chain_mask, "track_id"].nunique() == 1
    assert stitched.loc[stitched["frame_id"] == 1, "track_id"].iloc[0] == 4
    assert stitched.loc[stitched["frame_id"] == 13, "track_id"].iloc[0] == 5
    assert stitched.loc[stitched["frame_id"] == 45, "track_id"].iloc[0] == 6

    empty = df.iloc[0:0]
    assert len(stitch_tracklets(empty)) == 0
