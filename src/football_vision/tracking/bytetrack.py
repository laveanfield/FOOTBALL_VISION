"""ByteTrack two-stage association tracker (stage P5)."""

from __future__ import annotations

import numpy as np

from football_vision.config import TrackingConfig
from football_vision.datatypes import Detections, Homography
from football_vision.tracking.association import solve_assignment
from football_vision.tracking.base import Tracker
from football_vision.tracking.sort import (
    KalmanTrackPool,
    finalize_tracked_detections,
)


class ByteTrackTracker(Tracker):
    """ByteTrack: associates every detection box, including low-score ones.

    Round 1 matches tracks with high-score detections (``score >= high_thresh``).
    Round 2 matches the still-unmatched tracks with low-score detections
    (``low_thresh <= score < high_thresh``), which are often occluded players. Only
    unmatched high-score detections start new tracks.

    Implemented cleanly on top of the project's own :class:`KalmanFilter` and
    :func:`solve_assignment` to avoid external tracking API breakage across
    dependency releases.

    Attributes:
        cfg: Tracking settings.
        fps: Frame rate used by the motion model.
    """

    def __init__(self, cfg: TrackingConfig, fps: float) -> None:
        """Creates the tracker.

        Args:
            cfg: Tracking settings (high/low thresholds, buffer, match threshold).
            fps: Frame rate of the processed sequence.
        """
        self.cfg = cfg
        self.fps = fps
        self._pool = KalmanTrackPool(cfg, fps)

    def update(
        self,
        detections: Detections,
        frame_id: int,
        homography: Homography | None = None,
    ) -> Detections:
        """Processes one frame with ByteTrack two-round association.

        Args:
            detections: Person detections of the current frame.
            frame_id: 0-based id of the current frame.
            homography: Image-to-pitch homography when ``cfg.space == "pitch"``.

        Returns:
            Detections of active tracks with ``track_ids`` set.
        """
        del frame_id
        self._pool.predict_all()
        extracted = self._pool.extract_measurements(detections, homography)
        if extracted is None or len(detections) == 0:
            self._pool.prune_expired()
            return finalize_tracked_detections(detections, {})

        measurements, valid_mask = extracted
        scores = detections.scores
        high_indices = np.flatnonzero(valid_mask & (scores >= self.cfg.high_thresh))
        low_indices = np.flatnonzero(
            valid_mask
            & (scores >= self.cfg.low_thresh)
            & (scores < self.cfg.high_thresh)
        )

        # Round 1: associate all existing tracks against high-score detections.
        high_dets = detections.select(high_indices)
        high_meas = measurements[high_indices]
        cost_high = self._pool.build_cost(self._pool.tracks, high_dets, high_meas)
        matches_high, unmatched_tracks, unmatched_high = solve_assignment(
            cost_high, max_cost=self.cfg.match_thresh
        )

        det_to_track_id: dict[int, int] = {}
        for row, col in matches_high:
            track = self._pool.tracks[row]
            det_idx = int(high_indices[col])
            self._pool.update_track(
                track, detections.boxes_xyxy[det_idx], high_meas[col]
            )
            det_to_track_id[det_idx] = track.track_id

        # Round 2: recover still-unmatched tracks using low-score detections.
        if unmatched_tracks and len(low_indices) > 0:
            rem_tracks = [self._pool.tracks[r] for r in unmatched_tracks]
            low_dets = detections.select(low_indices)
            low_meas = measurements[low_indices]
            cost_low = self._pool.build_cost(rem_tracks, low_dets, low_meas)
            matches_low, _, _ = solve_assignment(
                cost_low, max_cost=self.cfg.match_thresh
            )
            for row, col in matches_low:
                track = rem_tracks[row]
                det_idx = int(low_indices[col])
                self._pool.update_track(
                    track, detections.boxes_xyxy[det_idx], low_meas[col]
                )
                det_to_track_id[det_idx] = track.track_id

        # Only unmatched high-score detections may initialize new tracks.
        for col in unmatched_high:
            det_idx = int(high_indices[col])
            new_track = self._pool.spawn_track(
                detections.boxes_xyxy[det_idx], high_meas[col]
            )
            det_to_track_id[det_idx] = new_track.track_id

        self._pool.prune_expired()
        return finalize_tracked_detections(detections, det_to_track_id)

    def reset(self) -> None:
        """Forgets all tracks and restarts id numbering."""
        self._pool.reset()
