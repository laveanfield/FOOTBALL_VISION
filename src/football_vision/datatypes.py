"""Shared type aliases and light-weight data containers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Frame = npt.NDArray[np.uint8]
"""BGR image of shape ``(height, width, 3)``."""

Mask = npt.NDArray[np.uint8]
"""Binary mask of shape ``(height, width)`` with values 0 or 255."""

Homography = npt.NDArray[np.float64]
"""Homography matrix of shape ``(3, 3)`` mapping image to pitch coordinates."""

Points2D = npt.NDArray[np.float64]
"""Array of 2D points with shape ``(N, 2)``."""


@dataclass(frozen=True)
class Detections:
    """Detections of a single frame in image coordinates.

    Attributes:
        boxes_xyxy: Float array of shape ``(N, 4)`` holding ``(x1, y1, x2, y2)``.
        scores: Float array of shape ``(N,)`` with confidence scores.
        class_ids: Integer array of shape ``(N,)`` with class ids.
        track_ids: Optional integer array of shape ``(N,)`` with track ids. It is
            ``None`` until a tracker has processed the detections.
    """

    boxes_xyxy: npt.NDArray[np.float32]
    scores: npt.NDArray[np.float32]
    class_ids: npt.NDArray[np.int64]
    track_ids: npt.NDArray[np.int64] | None = None

    def __post_init__(self) -> None:
        """Validates that all arrays agree on the number of detections.

        Raises:
            ValueError: If the boxes do not have shape ``(N, 4)`` or any other
                array does not have shape ``(N,)``.
        """
        n = len(self.scores)
        if self.boxes_xyxy.shape != (n, 4):
            raise ValueError(
                f"boxes_xyxy must have shape ({n}, 4), got {self.boxes_xyxy.shape}"
            )
        if self.class_ids.shape != (n,):
            raise ValueError(
                f"class_ids must have shape ({n},), got {self.class_ids.shape}"
            )
        if self.track_ids is not None and self.track_ids.shape != (n,):
            raise ValueError(
                f"track_ids must have shape ({n},), got {self.track_ids.shape}"
            )

    def __len__(self) -> int:
        """Returns the number of detections."""
        return len(self.scores)

    @classmethod
    def empty(cls) -> Detections:
        """Creates a container with zero detections.

        Returns:
            An empty :class:`Detections` instance without track ids.
        """
        return cls(
            boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            class_ids=np.zeros((0,), dtype=np.int64),
        )

    def select(self, selector: npt.ArrayLike) -> Detections:
        """Selects a subset of the detections.

        Args:
            selector: Boolean mask of shape ``(N,)`` or integer indices.

        Returns:
            A new :class:`Detections` instance holding the selected entries.
        """
        sel = np.asarray(selector)
        track_ids = None if self.track_ids is None else self.track_ids[sel]
        return Detections(
            boxes_xyxy=self.boxes_xyxy[sel],
            scores=self.scores[sel],
            class_ids=self.class_ids[sel],
            track_ids=track_ids,
        )

    def with_track_ids(self, track_ids: npt.ArrayLike) -> Detections:
        """Returns a copy of the detections with track ids attached.

        Args:
            track_ids: Integer track ids of shape ``(N,)``.

        Returns:
            A new :class:`Detections` instance carrying ``track_ids``.
        """
        return Detections(
            boxes_xyxy=self.boxes_xyxy,
            scores=self.scores,
            class_ids=self.class_ids,
            track_ids=np.asarray(track_ids, dtype=np.int64),
        )
