"""Metric model of a football pitch, the target space of the homography.

The template provides named landmarks (for clicking and matching), straight line
segments and sampled curves (for drawing the pitch over a video or on the
top-down map). Dimensions follow the IFAB Laws of the Game.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from football_vision.constants import (
    CENTER_CIRCLE_RADIUS_M,
    GOAL_AREA_DEPTH_M,
    GOAL_AREA_WIDTH_M,
    PENALTY_AREA_DEPTH_M,
    PENALTY_AREA_WIDTH_M,
    PENALTY_SPOT_DISTANCE_M,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
)
from football_vision.datatypes import Points2D

Segment = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class PitchTemplate:
    """Metric pitch with named landmarks and drawable lines.

    Coordinates are in meters. The origin is the top-left corner, ``X`` runs along
    the length and ``Y`` along the width. Landmarks of the two halves are prefixed
    with ``left_`` and ``right_``.

    Attributes:
        length: Pitch length along ``X`` in meters.
        width: Pitch width along ``Y`` in meters.
    """

    length: float = PITCH_LENGTH_M
    width: float = PITCH_WIDTH_M

    @property
    def penalty_area_y(self) -> tuple[float, float]:
        """``Y`` range of the penalty area, centered on the goal."""
        half = PENALTY_AREA_WIDTH_M / 2
        return (self.width / 2 - half, self.width / 2 + half)

    @property
    def goal_area_y(self) -> tuple[float, float]:
        """``Y`` range of the 5.5 m goal area, centered on the goal."""
        half = GOAL_AREA_WIDTH_M / 2
        return (self.width / 2 - half, self.width / 2 + half)

    def landmarks(self) -> dict[str, tuple[float, float]]:
        """Lists all named landmarks of the pitch.

        Returns:
            Mapping from landmark name to ``(X, Y)`` in meters (25 entries):
            four corners, the two halfway-line ends, the center, and nine
            landmarks per half (penalty area, goal area and penalty spot).
        """
        length, width = self.length, self.width
        pa_y0, pa_y1 = self.penalty_area_y
        ga_y0, ga_y1 = self.goal_area_y
        points: dict[str, tuple[float, float]] = {
            "corner_tl": (0.0, 0.0),
            "corner_tr": (length, 0.0),
            "corner_br": (length, width),
            "corner_bl": (0.0, width),
            "halfway_top": (length / 2, 0.0),
            "halfway_bottom": (length / 2, width),
            "center": (length / 2, width / 2),
        }
        left = {
            "pa_goalline_top": (0.0, pa_y0),
            "pa_goalline_bottom": (0.0, pa_y1),
            "pa_outer_top": (PENALTY_AREA_DEPTH_M, pa_y0),
            "pa_outer_bottom": (PENALTY_AREA_DEPTH_M, pa_y1),
            "ga_goalline_top": (0.0, ga_y0),
            "ga_goalline_bottom": (0.0, ga_y1),
            "ga_outer_top": (GOAL_AREA_DEPTH_M, ga_y0),
            "ga_outer_bottom": (GOAL_AREA_DEPTH_M, ga_y1),
            "penalty_spot": (PENALTY_SPOT_DISTANCE_M, width / 2),
        }
        for name, (x, y) in left.items():
            points[f"left_{name}"] = (x, y)
            points[f"right_{name}"] = (length - x, y)
        return points

    def landmark_array(self, names: Sequence[str]) -> Points2D:  # noqa: DOC502
        """Looks up several landmarks at once.

        Args:
            names: Landmark names, as returned by :meth:`landmarks`.

        Returns:
            Array of shape ``(len(names), 2)`` in the order of ``names``.

        Raises:
            KeyError: If a name is not a known landmark.
        """
        table = self.landmarks()
        return np.array([table[name] for name in names], dtype=np.float64).reshape(
            -1, 2
        )

    def segments(self) -> list[Segment]:
        """Lists the straight line segments of the pitch markings.

        Returns:
            Segments as ``((x0, y0), (x1, y1))`` in meters: touchlines, goal
            lines, halfway line, and three sides of each penalty and goal area.
        """
        length, width = self.length, self.width
        segs: list[Segment] = [
            ((0.0, 0.0), (length, 0.0)),
            ((0.0, width), (length, width)),
            ((0.0, 0.0), (0.0, width)),
            ((length, 0.0), (length, width)),
            ((length / 2, 0.0), (length / 2, width)),
        ]
        areas = (
            (PENALTY_AREA_DEPTH_M, self.penalty_area_y),
            (GOAL_AREA_DEPTH_M, self.goal_area_y),
        )
        for depth, (y0, y1) in areas:
            segs += [
                ((0.0, y0), (depth, y0)),
                ((depth, y0), (depth, y1)),
                ((depth, y1), (0.0, y1)),
                ((length, y0), (length - depth, y0)),
                ((length - depth, y0), (length - depth, y1)),
                ((length - depth, y1), (length, y1)),
            ]
        return segs

    def polylines(self, arc_points: int = 48) -> list[Points2D]:
        """Samples every marking as a polyline, ready for drawing.

        Args:
            arc_points: Number of samples per half circle or arc.

        Returns:
            One ``(N, 2)`` array per marking: straight segments have two points,
            the center circle and the two penalty arcs are sampled curves.
        """
        lines = [np.array(seg, dtype=np.float64) for seg in self.segments()]
        cx, cy, radius = self.length / 2, self.width / 2, CENTER_CIRCLE_RADIUS_M
        angles = np.linspace(0.0, 2 * math.pi, 2 * arc_points + 1)
        lines.append(
            np.stack(
                [cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1
            )
        )
        # Penalty arcs: the part of the circle around the penalty spot outside the box.
        cos_limit = (PENALTY_AREA_DEPTH_M - PENALTY_SPOT_DISTANCE_M) / radius
        theta0 = math.acos(cos_limit)
        theta = np.linspace(-theta0, theta0, arc_points)
        left_x = PENALTY_SPOT_DISTANCE_M + radius * np.cos(theta)
        arc_y = cy + radius * np.sin(theta)
        lines.append(np.stack([left_x, arc_y], axis=1))
        lines.append(np.stack([self.length - left_x, arc_y], axis=1))
        return lines
