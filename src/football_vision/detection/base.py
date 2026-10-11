"""Detector interface shared by all detection back-ends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from football_vision.datatypes import Detections, Frame


class Detector(ABC):
    """Interface of an object detector that works on single frames."""

    @abstractmethod
    def detect(self, frame: Frame) -> Detections:
        """Detects objects in one frame.

        Args:
            frame: BGR image of shape ``(H, W, 3)``.

        Returns:
            The detections in image coordinates.
        """

    def detect_batch(self, frames: Sequence[Frame]) -> list[Detections]:
        """Detects objects in several frames.

        The default implementation loops over :meth:`detect`; back-ends that can
        batch inference should override it.

        Args:
            frames: BGR images.

        Returns:
            One :class:`~football_vision.datatypes.Detections` per input frame.
        """
        return [self.detect(frame) for frame in frames]
