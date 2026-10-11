"""Team and role classification from jersey colors (stage P6)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from football_vision.config import TeamConfig


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

    def fit(self, features: npt.NDArray[np.float64]) -> None:
        """Learns the color clusters.

        Args:
            features: Array of shape ``(N, D)`` with one feature per player sample.

        Raises:
            NotImplementedError: Stub; to be implemented by member C.
        """
        # TODO(C): sklearn.cluster.KMeans(n_clusters=cfg.n_clusters, n_init=10,
        #   random_state=seed).
        raise NotImplementedError("TODO(C): implement TeamClassifier.fit")

    def predict(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """Assigns each feature to its nearest color cluster.

        Args:
            features: Array of shape ``(N, D)``.

        Returns:
            Integer cluster ids of shape ``(N,)``.

        Raises:
            NotImplementedError: Stub; to be implemented by member C.
        """
        # TODO(C): delegate to the fitted KMeans model.
        raise NotImplementedError("TODO(C): implement TeamClassifier.predict")


def vote_per_track(samples: pd.DataFrame) -> pd.DataFrame:
    """Stabilizes cluster labels with a per-track majority vote.

    A single frame can be mislabeled by occlusion or lighting; a track keeps the
    cluster that most of its samples received.

    Args:
        samples: Table with the columns ``track_id`` and ``cluster``, one row per
            player sample.

    Returns:
        Table with one row per track and the columns ``track_id`` and ``cluster``.

    Raises:
        NotImplementedError: Stub; to be implemented by member C.
    """
    # TODO(C): groupby('track_id')['cluster'].agg(lambda s: s.mode().iloc[0]).
    raise NotImplementedError("TODO(C): implement vote_per_track")


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

    Raises:
        NotImplementedError: Stub; to be implemented by member C.
    """
    # TODO(C): use constants TEAM_UNKNOWN and ROLE_* for undecided tracks.
    raise NotImplementedError("TODO(C): implement assign_teams_and_roles")
