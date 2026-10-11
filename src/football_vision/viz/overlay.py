"""Drawing helpers that overlay detections, tracks and the pitch on video frames.

All functions return a modified copy and never change the input frame.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from football_vision.constants import (
    ROLE_GOALKEEPER,
    ROLE_PLAYER,
    ROLE_REFEREE,
    TEAM_UNKNOWN,
)
from football_vision.datatypes import Detections, Frame, Homography
from football_vision.homography.geometry import apply_homography, invert_homography
from football_vision.pitch.template import PitchTemplate

Color = tuple[int, int, int]

TEAM_COLORS_BGR: dict[int, Color] = {
    0: (60, 60, 255),
    1: (255, 120, 40),
    TEAM_UNKNOWN: (170, 170, 170),
}
REFEREE_COLOR_BGR: Color = (0, 220, 255)
GOALKEEPER_COLOR_BGR: Color = (60, 220, 60)


def color_for(team: int, role: str = ROLE_PLAYER) -> Color:
    """Chooses the drawing color of a player.

    Args:
        team: Team id (``-1`` for unknown).
        role: Role label such as ``"player"``, ``"goalkeeper"`` or ``"referee"``.

    Returns:
        A BGR color. Referees and goalkeepers have their own colors.
    """
    if role == ROLE_REFEREE:
        return REFEREE_COLOR_BGR
    if role == ROLE_GOALKEEPER:
        return GOALKEEPER_COLOR_BGR
    return TEAM_COLORS_BGR.get(int(team), TEAM_COLORS_BGR[TEAM_UNKNOWN])


def _label(canvas: Frame, text: str, origin: tuple[int, int], color: Color) -> None:
    """Draws a small label with a dark outline for readability."""
    x, y = origin
    cv2.putText(
        canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
    )


def draw_detections(
    frame: Frame, dets: Detections, color: Color = (0, 255, 0), thickness: int = 2
) -> Frame:
    """Draws detection boxes with their confidence scores.

    Args:
        frame: BGR frame.
        dets: Detections of the frame.
        color: BGR color of the boxes.
        thickness: Line thickness in pixels.

    Returns:
        A copy of the frame with the boxes drawn.
    """
    out = frame.copy()
    for box, score in zip(dets.boxes_xyxy, dets.scores, strict=True):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        _label(out, f"{score:.2f}", (x1, max(y1 - 4, 12)), color)
    return out


def draw_tracks(frame: Frame, tracks: pd.DataFrame, thickness: int = 2) -> Frame:
    """Draws tracked players with their id, colored by team and role.

    Args:
        frame: BGR frame.
        tracks: Rows of ``tracks.csv`` that belong to this frame.
        thickness: Line thickness in pixels.

    Returns:
        A copy of the frame with boxes and ``#id`` labels drawn.
    """
    out = frame.copy()
    for row in tracks.itertuples(index=False):
        color = color_for(int(row.team), str(row.role))
        x1, y1, x2, y2 = (int(round(v)) for v in (row.x1, row.y1, row.x2, row.y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        _label(out, f"#{int(row.track_id)}", (x1, max(y1 - 4, 12)), color)
    return out


def draw_pitch_template(
    frame: Frame,
    homography: Homography,
    template: PitchTemplate | None = None,
    color: Color = (0, 255, 255),
    thickness: int = 2,
) -> Frame:
    """Projects the pitch markings into the image to check the homography visually.

    If the homography is right, the drawn lines sit exactly on the real markings.

    Args:
        frame: BGR frame.
        homography: Matrix mapping image to pitch; it is inverted for drawing.
        template: Pitch model; defaults to the 105 x 68 m template.
        color: BGR color of the lines.
        thickness: Line thickness in pixels.

    Returns:
        A copy of the frame with the projected pitch lines drawn.
    """
    model = template or PitchTemplate()
    inverse = invert_homography(homography)
    out = frame.copy()
    for line in model.polylines():
        pts = apply_homography(inverse, line)
        if not np.isfinite(pts).all():
            continue
        pts = np.clip(pts, -1e5, 1e5).round().astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, color, thickness, cv2.LINE_AA)
    return out
