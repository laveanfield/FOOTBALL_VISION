"""Tracker interface and factory."""

from __future__ import annotations

from abc import ABC, abstractmethod

from football_vision.config import TrackingConfig
from football_vision.datatypes import Detections, Homography


class Tracker(ABC):
    """Interface of an online multi-object tracker."""

    @abstractmethod
    def update(
        self,
        detections: Detections,
        frame_id: int,
        homography: Homography | None = None,
    ) -> Detections:
        """Associates the detections of one frame with existing tracks.

        It must be called once for every processed frame, also when there are no
        detections, so that the motion model keeps advancing.

        Args:
            detections: Person detections of the current frame.
            frame_id: 0-based id of the current frame.
            homography: Image-to-pitch homography of the frame. It is only needed
                when tracking in pitch space (``TrackingConfig.space == "pitch"``).

        Returns:
            The detections that belong to a confirmed track, with ``track_ids`` set.
        """

    @abstractmethod
    def reset(self) -> None:
        """Forgets all tracks so that the tracker can process a new sequence."""


def make_tracker(cfg: TrackingConfig, fps: float) -> Tracker:
    """Creates the tracker selected in the configuration.

    Args:
        cfg: Tracking settings; ``cfg.tracker`` selects the implementation.
        fps: Frame rate of the processed sequence (used for the time step).

    Returns:
        A ready-to-use tracker.

    Raises:
        ValueError: If ``cfg.tracker`` is not supported.
    """
    if cfg.tracker == "sort":
        from football_vision.tracking.sort import SortTracker

        return SortTracker(cfg, fps)
    if cfg.tracker == "bytetrack":
        from football_vision.tracking.bytetrack import ByteTrackTracker

        return ByteTrackTracker(cfg, fps)
    raise ValueError(f"Unsupported tracker '{cfg.tracker}'")
