"""Own minimal SORT tracker: Kalman filter + Hungarian assignment (stage P5).

Writing this tracker from scratch (about 100 lines) is a deliberate course goal: it
makes the tracking logic explainable and gives a baseline to compare against
ByteTrack (ablation A5) and to run in pitch space (ablation A6).
"""

from __future__ import annotations

from dataclasses import dataclass

from football_vision.config import TrackingConfig
from football_vision.datatypes import Detections, Homography
from football_vision.tracking.base import Tracker
from football_vision.tracking.kalman import KalmanFilter


@dataclass
class SortTrack:
    """State of one SORT track.

    Attributes:
        track_id: Unique positive id of the track.
        kalman: Kalman filter holding the motion state.
        hits: Number of frames in which the track was matched.
        age: Number of frames since the track was created.
        time_since_update: Frames since the last matched detection.
    """

    track_id: int
    kalman: KalmanFilter
    hits: int = 0
    age: int = 0
    time_since_update: int = 0


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
        self._tracks: list[SortTrack] = []
        self._next_id = 1

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

        Raises:
            NotImplementedError: Stub; to be implemented by member C.
        """
        # TODO(C): 1) predict every track; 2) build the cost with
        #   association.build_iou_cost (or pitch distances after projecting foot
        #   points with `homography`); 3) association.solve_assignment with
        #   gate = cfg.match_thresh; 4) update/create/delete tracks.
        raise NotImplementedError("TODO(C): implement SortTracker.update")

    def reset(self) -> None:
        """Deletes all tracks and restarts id numbering."""
        self._tracks = []
        self._next_id = 1
