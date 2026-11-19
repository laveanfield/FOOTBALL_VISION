"""Own minimal SORT tracker: Kalman filter + Hungarian assignment (stage P5).

Writing this tracker from scratch (about 100 lines) is a deliberate course goal: it
makes the tracking logic explainable and gives a baseline to compare against
ByteTrack (ablation A5) and to run in pitch space (ablation A6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from football_vision.boxes import foot_points
from football_vision.config import TrackingConfig
from football_vision.datatypes import Detections, Homography
from football_vision.homography.geometry import apply_homography
from football_vision.tracking.association import (
    build_iou_cost,
    build_pitch_distance_cost,
    solve_assignment,
)
from football_vision.tracking.base import Tracker
from football_vision.tracking.kalman import KalmanFilter, constant_velocity_matrices

_IMAGE_PROCESS_STD = 60.0
_IMAGE_MEAS_STD = 4.0
_IMAGE_P0_DIAG = (25.0, 25.0, 900.0, 900.0)

_PITCH_PROCESS_STD = 1.5
_PITCH_MEAS_STD = 0.3
_PITCH_P0_DIAG = (1.0, 1.0, 25.0, 25.0)


@dataclass
class SortTrack:
    """State of one SORT track.

    Attributes:
        track_id: Unique positive id of the track.
        kalman: Kalman filter holding the motion state.
        hits: Number of frames in which the track was matched.
        age: Number of frames since the track was created.
        time_since_update: Frames since the last matched detection.
        box_size: Smoothed ``(width, height)`` in pixels for box prediction.
    """

    track_id: int
    kalman: KalmanFilter
    hits: int = 0
    age: int = 0
    time_since_update: int = 0
    box_size: tuple[float, float] = (0.0, 0.0)

    def predicted_box(self) -> npt.NDArray[np.float64]:
        """Reconstructs an ``(x1, y1, x2, y2)`` box from the predicted foot point.

        Returns:
            Array of shape ``(4,)`` centered horizontally on ``kalman.x[0]`` with
            bottom edge at ``kalman.x[1]``.
        """
        cx, y2 = float(self.kalman.x[0]), float(self.kalman.x[1])
        w, h = self.box_size
        return np.array([cx - 0.5 * w, y2 - h, cx + 0.5 * w, y2], dtype=np.float64)


class KalmanTrackPool:
    """Shared Kalman track lifecycle and cost helpers for SORT and ByteTrack."""

    def __init__(self, cfg: TrackingConfig, fps: float) -> None:
        """Initializes motion-model constants from tracking config and frame rate.

        Args:
            cfg: Tracking settings (space, thresholds, buffer).
            fps: Frame rate of the processed sequence.
        """
        self.cfg = cfg
        self.fps = fps
        dt = 1.0 / max(float(fps), 1e-6)
        if cfg.space == "pitch":
            self._model = constant_velocity_matrices(
                dt, _PITCH_PROCESS_STD, _PITCH_MEAS_STD
            )
            self._p0 = np.diag(_PITCH_P0_DIAG)
        else:
            self._model = constant_velocity_matrices(
                dt, _IMAGE_PROCESS_STD, _IMAGE_MEAS_STD
            )
            self._p0 = np.diag(_IMAGE_P0_DIAG)
        self.tracks: list[SortTrack] = []
        self.next_id = 1

    def reset(self) -> None:
        """Clears all active tracks and resets id numbering."""
        self.tracks = []
        self.next_id = 1

    def predict_all(self) -> None:
        """Advances every track by one motion-model prediction step."""
        for track in self.tracks:
            track.kalman.predict()
            track.age += 1
            track.time_since_update += 1

    def prune_expired(self) -> None:
        """Drops tracks not matched within ``cfg.track_buffer`` frames."""
        self.tracks = [
            t for t in self.tracks if t.time_since_update <= self.cfg.track_buffer
        ]

    def extract_measurements(
        self, detections: Detections, homography: Homography | None
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]] | None:
        """Computes per-detection state measurements in the configured space.

        Args:
            detections: Detections for the current frame.
            homography: Image-to-pitch homography when ``cfg.space == "pitch"``.

        Returns:
            Tuple ``(measurements, valid_mask)`` of shapes ``(N, 2)`` and ``(N,)``,
            or ``None`` if pitch-space tracking cannot run without a homography.
        """
        if len(detections) == 0:
            return np.empty((0, 2), dtype=np.float64), np.zeros(0, dtype=bool)
        feet_px = foot_points(detections.boxes_xyxy)
        if self.cfg.space == "pitch":
            if homography is None:
                return None
            pts = apply_homography(homography, feet_px)
            valid = np.all(np.isfinite(pts), axis=1)
            return pts, valid
        valid = np.all(np.isfinite(feet_px), axis=1)
        return feet_px, valid

    def build_cost(
        self,
        tracks: list[SortTrack],
        detections: Detections,
        measurements: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Builds the track-to-detection cost matrix in image or pitch space.

        Args:
            tracks: Tracks forming the rows of the cost matrix.
            detections: Detections forming the columns.
            measurements: Precomputed points of shape ``(len(detections), 2)``.

        Returns:
            Cost matrix of shape ``(len(tracks), len(detections))``.
        """
        if self.cfg.space == "pitch":
            track_pts = (
                np.stack([t.kalman.x[:2] for t in tracks])
                if tracks
                else np.empty((0, 2), dtype=np.float64)
            )
            return build_pitch_distance_cost(track_pts, measurements)
        track_boxes = (
            np.stack([t.predicted_box() for t in tracks])
            if tracks
            else np.empty((0, 4), dtype=np.float64)
        )
        return build_iou_cost(track_boxes, detections.boxes_xyxy)

    def update_track(
        self,
        track: SortTrack,
        box: npt.NDArray[np.floating],
        measurement: npt.NDArray[np.float64],
    ) -> None:
        """Updates a matched track's Kalman filter, smoothed box size, and counters.

        Args:
            track: Matched track instance.
            box: Matched bounding box ``(x1, y1, x2, y2)``.
            measurement: Measurement vector ``(2,)`` in the configured tracking space.
        """
        track.kalman.update(measurement)
        w = float(max(box[2] - box[0], 1.0))
        h = float(max(box[3] - box[1], 1.0))
        prev_w, prev_h = track.box_size
        track.box_size = (0.7 * prev_w + 0.3 * w, 0.7 * prev_h + 0.3 * h)
        track.hits += 1
        track.time_since_update = 0

    def spawn_track(
        self,
        box: npt.NDArray[np.floating],
        measurement: npt.NDArray[np.float64],
    ) -> SortTrack:
        """Creates and registers a new track from an unmatched detection.

        Args:
            box: Bounding box ``(x1, y1, x2, y2)`` of the detection.
            measurement: Initial position ``(2,)`` in the configured tracking space.

        Returns:
            The newly created :class:`SortTrack`.
        """
        f, m, q, r = self._model
        x0 = np.array(
            [float(measurement[0]), float(measurement[1]), 0.0, 0.0],
            dtype=np.float64,
        )
        kf = KalmanFilter(f, m, q, r, x0=x0, p0=self._p0.copy())
        w = float(max(box[2] - box[0], 1.0))
        h = float(max(box[3] - box[1], 1.0))
        track = SortTrack(
            track_id=self.next_id,
            kalman=kf,
            hits=1,
            age=1,
            time_since_update=0,
            box_size=(w, h),
        )
        self.next_id += 1
        self.tracks.append(track)
        return track


def finalize_tracked_detections(
    detections: Detections,
    det_to_track_id: dict[int, int],
) -> Detections:
    """Selects matched detections in index order and attaches their track ids.

    Args:
        detections: Input detections of the frame.
        det_to_track_id: Mapping from detection index to assigned ``track_id``.

    Returns:
        Filtered :class:`Detections` carrying ``track_ids``.
    """
    if not det_to_track_id:
        return Detections.empty().with_track_ids(np.empty(0, dtype=np.int64))
    indices = sorted(det_to_track_id)
    track_ids = [det_to_track_id[idx] for idx in indices]
    return detections.select(indices).with_track_ids(track_ids)


class SortTracker(Tracker):
    """Simple Online and Realtime Tracker.

    Per frame: predict all tracks, associate predictions with detections (cost
    ``1 - IoU`` in image space, or distance in meters in pitch space), update matched
    tracks, create tracks from unmatched detections, and delete tracks that were not
    matched for ``track_buffer`` frames.

    Attributes:
        cfg: Tracking settings.
        fps: Frame rate used to derive the motion-model time step.
    """

    def __init__(self, cfg: TrackingConfig, fps: float) -> None:
        """Creates an empty tracker.

        Args:
            cfg: Tracking settings (thresholds, buffer length, tracking space).
            fps: Frame rate of the processed sequence.
        """
        self.cfg = cfg
        self.fps = fps
        self._pool = KalmanTrackPool(cfg, fps)
        self._tracks = self._pool.tracks
        self._next_id = self._pool.next_id

    def update(
        self,
        detections: Detections,
        frame_id: int,
        homography: Homography | None = None,
    ) -> Detections:
        """Processes one frame.

        Args:
            detections: Person detections of the current frame.
            frame_id: 0-based id of the current frame.
            homography: Image-to-pitch homography; required for pitch-space tracking.

        Returns:
            Detections of confirmed, matched tracks with ``track_ids`` set.
        """
        del frame_id
        self._pool.predict_all()
        extracted = self._pool.extract_measurements(detections, homography)
        if extracted is None or len(detections) == 0:
            self._pool.prune_expired()
            self._tracks = self._pool.tracks
            return finalize_tracked_detections(detections, {})

        measurements, valid_mask = extracted
        valid_indices = np.flatnonzero(valid_mask).tolist()
        valid_dets = detections.select(valid_indices)
        valid_meas = measurements[valid_indices]

        cost = self._pool.build_cost(self._pool.tracks, valid_dets, valid_meas)
        matches, _, unmatched_cols = solve_assignment(
            cost, max_cost=self.cfg.match_thresh
        )

        det_to_track_id: dict[int, int] = {}
        for row, col in matches:
            track = self._pool.tracks[row]
            det_idx = valid_indices[col]
            self._pool.update_track(
                track, detections.boxes_xyxy[det_idx], valid_meas[col]
            )
            det_to_track_id[det_idx] = track.track_id

        for col in unmatched_cols:
            det_idx = valid_indices[col]
            new_track = self._pool.spawn_track(
                detections.boxes_xyxy[det_idx], valid_meas[col]
            )
            det_to_track_id[det_idx] = new_track.track_id

        self._pool.prune_expired()
        self._tracks = self._pool.tracks
        self._next_id = self._pool.next_id
        return finalize_tracked_detections(detections, det_to_track_id)

    def reset(self) -> None:
        """Deletes all tracks and restarts id numbering."""
        self._pool.reset()
        self._tracks = self._pool.tracks
        self._next_id = self._pool.next_id
