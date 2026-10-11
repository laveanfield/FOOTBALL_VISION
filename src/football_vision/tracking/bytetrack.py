"""ByteTrack baseline tracker (stage P5)."""

from __future__ import annotations

from football_vision.config import TrackingConfig
from football_vision.datatypes import Detections, Homography
from football_vision.tracking.base import Tracker


class ByteTrackTracker(Tracker):
    """ByteTrack: associates every detection box, including low-score ones.

    Round 1 matches tracks with high-score detections (``score >= high_thresh``).
    Round 2 matches the still-unmatched tracks with low-score detections
    (``low_thresh <= score < high_thresh``), which are often occluded players. Only
    unmatched high-score detections start new tracks.

    The implementation should wrap a maintained library (for example
    ``supervision.ByteTrack`` or the official code) rather than re-implement it.

    Attributes:
        cfg: Tracking settings.
        fps: Frame rate used by the backend's motion model.
    """

    def __init__(self, cfg: TrackingConfig, fps: float) -> None:
        """Creates the tracker.

        Args:
            cfg: Tracking settings (high/low thresholds, buffer, match threshold).
            fps: Frame rate of the processed sequence.
        """
        self.cfg = cfg
        self.fps = fps

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
            homography: Unused by the image-space ByteTrack backend.

        Returns:
            Detections of active tracks with ``track_ids`` set.

        Raises:
            NotImplementedError: Stub; to be implemented by member C.
        """
        # TODO(C): convert to the backend's detection type, call its update method,
        #   and map the returned track ids back onto `detections`. Pin the library
        #   version in requirements.txt, since the API changes between releases.
        raise NotImplementedError("TODO(C): implement ByteTrackTracker.update")

    def reset(self) -> None:
        """Forgets all tracks.

        Raises:
            NotImplementedError: Stub; to be implemented by member C.
        """
        # TODO(C): re-create the backend tracker.
        raise NotImplementedError("TODO(C): implement ByteTrackTracker.reset")
