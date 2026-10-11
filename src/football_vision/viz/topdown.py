"""Top-down pitch map: the "tactical view" of the players (stage P8)."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd

from football_vision.datatypes import Frame
from football_vision.pitch.template import PitchTemplate
from football_vision.viz.overlay import color_for

GRASS_BGR = (70, 140, 70)
LINE_BGR = (240, 240, 240)


class TopDownRenderer:
    """Draws the pitch and the players as a top-down map.

    Attributes:
        template: Pitch model that defines the lines and the extent.
        scale: Resolution of the map in pixels per meter.
        margin_px: Border around the pitch in pixels.
    """

    def __init__(
        self,
        template: PitchTemplate | None = None,
        scale_px_per_m: float = 8.0,
        margin_px: int = 20,
    ) -> None:
        """Creates a renderer and pre-draws the empty pitch.

        Args:
            template: Pitch model; defaults to the 105 x 68 m template.
            scale_px_per_m: Resolution of the map in pixels per meter.
            margin_px: Border around the pitch in pixels.
        """
        self.template = template or PitchTemplate()
        self.scale = scale_px_per_m
        self.margin_px = margin_px
        width, height = self.size
        self._blank = np.full((height, width, 3), GRASS_BGR, dtype=np.uint8)
        for line in self.template.polylines():
            pts = self.to_pixels(line).reshape(-1, 1, 2)
            cv2.polylines(self._blank, [pts], False, LINE_BGR, 2, cv2.LINE_AA)

    @property
    def size(self) -> tuple[int, int]:
        """Size ``(width, height)`` of the rendered map in pixels."""
        width = int(round(self.template.length * self.scale)) + 2 * self.margin_px
        height = int(round(self.template.width * self.scale)) + 2 * self.margin_px
        return width, height

    def to_pixels(self, xy: npt.ArrayLike) -> npt.NDArray[np.int32]:
        """Converts pitch coordinates in meters to map pixels.

        Args:
            xy: Array of shape ``(N, 2)`` with ``(X, Y)`` in meters.

        Returns:
            Integer array of shape ``(N, 2)`` with ``(px, py)``.
        """
        pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        return np.round(pts * self.scale + self.margin_px).astype(np.int32)

    def render(
        self,
        rows: pd.DataFrame,
        trails: Mapping[int, npt.NDArray[np.float64]] | None = None,
    ) -> Frame:
        """Draws one frame of the map.

        Args:
            rows: Rows of ``pitch_xy.csv`` of the current frame; invalid rows are
                skipped.
            trails: Optional mapping from track id to past positions ``(n, 2)``
                in meters.

        Returns:
            A BGR image of shape ``(height, width, 3)``.
        """
        canvas = self._blank.copy()
        valid = rows[rows["valid"] == 1] if len(rows) else rows
        colors = {
            int(r.track_id): color_for(int(r.team), str(r.role))
            for r in valid.itertuples(index=False)
        }
        for track_id, history in (trails or {}).items():
            if len(history) >= 2 and track_id in colors:
                pts = self.to_pixels(history).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts], False, colors[track_id], 1, cv2.LINE_AA)
        radius = max(int(round(0.8 * self.scale)), 3)
        for r in valid.itertuples(index=False):
            center = tuple(int(v) for v in self.to_pixels([[r.X, r.Y]])[0])
            cv2.circle(canvas, center, radius, colors[int(r.track_id)], -1, cv2.LINE_AA)
            cv2.circle(canvas, center, radius, (0, 0, 0), 1, cv2.LINE_AA)
        return canvas


class TrailBuffer:
    """Keeps the last positions of every track for drawing trails.

    Attributes:
        length: Maximum number of positions kept per track.
    """

    def __init__(self, length: int) -> None:
        """Creates an empty buffer.

        Args:
            length: Maximum number of positions kept per track.
        """
        self.length = length
        self._history: dict[int, deque[tuple[float, float]]] = {}
        self._last_seen: dict[int, int] = {}

    def update(self, frame_id: int, rows: pd.DataFrame | None) -> None:
        """Adds the positions of the current frame and forgets stale tracks.

        Args:
            frame_id: Id of the current frame.
            rows: Rows of ``pitch_xy.csv`` of the frame, or ``None`` if there are none.
        """
        if rows is not None:
            for r in rows[rows["valid"] == 1].itertuples(index=False):
                track_id = int(r.track_id)
                self._history.setdefault(track_id, deque(maxlen=self.length)).append(
                    (float(r.X), float(r.Y))
                )
                self._last_seen[track_id] = frame_id
        stale = [
            t for t, seen in self._last_seen.items() if frame_id - seen > self.length
        ]
        for track_id in stale:
            self._history.pop(track_id, None)
            self._last_seen.pop(track_id, None)

    def trails(self) -> dict[int, npt.NDArray[np.float64]]:
        """Returns the buffered positions.

        Returns:
            Mapping from track id to an array of shape ``(n, 2)`` in meters.
        """
        return {t: np.asarray(h, dtype=np.float64) for t, h in self._history.items()}
