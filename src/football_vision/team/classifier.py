"""Team and role classification from jersey colors (stage P6)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.cluster import KMeans

from football_vision.artifacts import TEAM_LABEL_COLUMNS
from football_vision.config import TeamConfig
from football_vision.constants import (
    ROLE_GOALKEEPER,
    ROLE_PLAYER,
    ROLE_REFEREE,
    ROLE_UNKNOWN,
    TEAM_UNKNOWN,
)

_GK_EXTREME_X_RATIO = 0.08
_MIN_TRACKS_FOR_GK_HEURISTIC = 4


class TeamClassifier:
    """Clusters jersey-color features with KMeans.

    Fit on features gathered from many frames, then label every sample with a
    cluster id. Cluster ids are arbitrary; :func:`assign_teams_and_roles` maps them
    to teams and roles.

    Attributes:
        cfg: Team settings (number of clusters, color space).
    """

    def __init__(self, cfg: TeamConfig) -> None:
        """Creates an unfitted classifier.

        Args:
            cfg: Team settings; ``cfg.n_clusters`` sets the number of KMeans clusters.
        """
        self.cfg = cfg
        self._model: KMeans | None = None

    def fit(self, features: npt.NDArray[np.float64]) -> None:
        """Learns the color clusters.

        Args:
            features: Array of shape ``(N, D)`` with one feature per player sample.

        Raises:
            ValueError: If ``features`` is empty or not two-dimensional.
        """
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or len(x) == 0:
            raise ValueError(
                f"Expected non-empty 2D feature array, got shape {x.shape}"
            )
        unique_rows = len(np.unique(np.round(x, decimals=6), axis=0))
        n_clusters = max(1, min(int(self.cfg.n_clusters), len(x), unique_rows))
        self._model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        self._model.fit(x)

    def predict(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """Assigns each feature to its nearest color cluster.

        Args:
            features: Array of shape ``(N, D)``.

        Returns:
            Integer cluster ids of shape ``(N,)``.

        Raises:
            RuntimeError: If :meth:`predict` is called before :meth:`fit`.
            ValueError: If ``features`` is not two-dimensional.
        """
        if self._model is None:
            raise RuntimeError("TeamClassifier must be fitted before calling predict()")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D feature array, got shape {x.shape}")
        if len(x) == 0:
            return np.empty(0, dtype=np.int64)
        return self._model.predict(x).astype(np.int64)


def vote_per_track(samples: pd.DataFrame) -> pd.DataFrame:
    """Stabilizes cluster labels with a per-track majority vote.

    A single frame can be mislabeled by occlusion or lighting; a track keeps the
    cluster that most of its samples received.

    Args:
        samples: Table with the columns ``track_id`` and ``cluster``, one row per
            player sample.

    Returns:
        Table with one row per track and the columns ``track_id`` and ``cluster``.
    """
    if len(samples) == 0:
        return pd.DataFrame(columns=["track_id", "cluster"])
    voted = (
        samples.groupby("track_id", sort=True)["cluster"]
        .agg(lambda s: int(s.mode().iloc[0]))
        .reset_index()
    )
    voted["track_id"] = voted["track_id"].astype(np.int64)
    voted["cluster"] = voted["cluster"].astype(np.int64)
    return voted


def _classify_outlier_role(
    track_id: int,
    tracks: pd.DataFrame,
    x_min: float,
    x_span: float,
    n_total_tracks: int,
) -> str:
    """Distinguishes edge-bound goalkeepers from a central-play referee."""
    if n_total_tracks < _MIN_TRACKS_FOR_GK_HEURISTIC or x_span <= 0.0:
        return ROLE_REFEREE
    rows = tracks[tracks["track_id"] == track_id]
    if len(rows) == 0:
        return ROLE_REFEREE
    median_x = float(((rows["x1"] + rows["x2"]) * 0.5).median())
    rel_x = (median_x - x_min) / x_span
    if rel_x <= _GK_EXTREME_X_RATIO or rel_x >= (1.0 - _GK_EXTREME_X_RATIO):
        return ROLE_GOALKEEPER
    return ROLE_REFEREE


def assign_teams_and_roles(
    track_clusters: pd.DataFrame,
    tracks: pd.DataFrame,
    cfg: TeamConfig,
) -> pd.DataFrame:
    """Maps color clusters to teams and roles.

    The two largest clusters are the teams (team ids 0 and 1). Smaller clusters are
    goalkeepers and the referee: a goalkeeper stays near one goal for a long time,
    whereas the referee is the lone third color and stays near the play.

    Args:
        track_clusters: Output of :func:`vote_per_track`.
        tracks: Full ``tracks.csv`` table, used for position-based rules.
        cfg: Team settings.

    Returns:
        Table with the columns of ``TEAM_LABEL_COLUMNS``: ``track_id``, ``team``
        and ``role``.
    """
    del cfg
    all_track_ids = (
        sorted(int(tid) for tid in tracks["track_id"].unique())
        if len(tracks) > 0
        else []
    )
    if not all_track_ids:
        return pd.DataFrame(columns=list(TEAM_LABEL_COLUMNS))

    cluster_by_track: dict[int, int] = (
        {
            int(row.track_id): int(row.cluster)
            for row in track_clusters.itertuples(index=False)
        }
        if len(track_clusters) > 0
        else {}
    )
    if not cluster_by_track:
        return pd.DataFrame(
            {
                "track_id": all_track_ids,
                "team": TEAM_UNKNOWN,
                "role": ROLE_UNKNOWN,
            },
            columns=list(TEAM_LABEL_COLUMNS),
        )

    counts = (
        track_clusters.groupby("cluster")["track_id"]
        .count()
        .sort_values(ascending=False, kind="mergesort")
    )
    ranked_clusters = [int(c) for c in counts.index.tolist()]
    team_clusters = {cid: idx for idx, cid in enumerate(ranked_clusters[:2])}

    foot_x = (tracks["x1"] + tracks["x2"]) * 0.5
    x_min = float(foot_x.min())
    x_span = float(foot_x.max() - x_min)
    n_total = len(all_track_ids)

    rows = []
    for tid in all_track_ids:
        cid = cluster_by_track.get(tid)
        if cid is None:
            rows.append((tid, TEAM_UNKNOWN, ROLE_UNKNOWN))
        elif cid in team_clusters:
            rows.append((tid, team_clusters[cid], ROLE_PLAYER))
        else:
            role = _classify_outlier_role(tid, tracks, x_min, x_span, n_total)
            rows.append((tid, TEAM_UNKNOWN, role))
    return pd.DataFrame(rows, columns=list(TEAM_LABEL_COLUMNS))
