"""Team shape metrics: centroid, extent, hull area and stretch (stage P7)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

from football_vision.constants import ROLE_GOALKEEPER


@dataclass(frozen=True)
class TeamShape:
    """Shape descriptors of one team at one instant.

    Attributes:
        centroid_x: Mean ``X`` of the players in meters.
        centroid_y: Mean ``Y`` of the players in meters.
        length_m: Extent along the pitch length, ``max(X) - min(X)``.
        width_m: Extent across the pitch, ``max(Y) - min(Y)``.
        hull_area_m2: Area of the convex hull of the players in square meters.
        stretch_index_m: Mean distance of the players from the centroid.
        n_players: Number of players used.
    """

    centroid_x: float
    centroid_y: float
    length_m: float
    width_m: float
    hull_area_m2: float
    stretch_index_m: float
    n_players: int


def compute_team_shape(points: npt.ArrayLike) -> TeamShape:
    """Computes the shape descriptors of a set of player positions.

    Length and width are sensitive to a single distant player, so the stretch index
    and the hull area are reported as well.

    Args:
        points: Positions of shape ``(N, 2)`` as ``(X, Y)`` in meters, ``N >= 1``.

    Returns:
        The :class:`TeamShape`; the hull area is zero for fewer than three
        points or collinear points.

    Raises:
        ValueError: If no point is given.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        raise ValueError("compute_team_shape needs at least one point")
    centroid = pts.mean(axis=0)
    try:
        area = (
            float(ConvexHull(pts).volume) if len(pts) >= 3 else 0.0
        )  # 2D: volume is area
    except QhullError:
        area = 0.0
    return TeamShape(
        centroid_x=float(centroid[0]),
        centroid_y=float(centroid[1]),
        length_m=float(np.ptp(pts[:, 0])),
        width_m=float(np.ptp(pts[:, 1])),
        hull_area_m2=area,
        stretch_index_m=float(np.linalg.norm(pts - centroid, axis=1).mean()),
        n_players=len(pts),
    )


def team_shape_timeseries(
    pitch_xy: pd.DataFrame,
    team: int,
    exclude_goalkeeper: bool = True,
    min_players: int = 3,
) -> pd.DataFrame:
    """Computes the team shape for every frame.

    Args:
        pitch_xy: ``pitch_xy.csv`` table.
        team: Team id to analyse.
        exclude_goalkeeper: Whether to ignore goalkeepers, who distort length and hull.
        min_players: Frames with fewer valid players are skipped.

    Returns:
        Table with ``frame_id`` and the fields of :class:`TeamShape`, one row per
        frame with enough players.
    """
    rows = pitch_xy[(pitch_xy["valid"] == 1) & (pitch_xy["team"] == team)]
    if exclude_goalkeeper:
        rows = rows[rows["role"] != ROLE_GOALKEEPER]
    records = []
    for frame_id, group in rows.groupby("frame_id"):
        if len(group) < min_players:
            continue
        shape = compute_team_shape(group[["X", "Y"]].to_numpy())
        records.append({"frame_id": int(frame_id), **asdict(shape)})
    columns = ["frame_id", *TeamShape.__dataclass_fields__.keys()]
    return pd.DataFrame(records, columns=columns)
