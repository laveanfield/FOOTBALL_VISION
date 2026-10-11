"""Tracking metrics: MOTA, IDF1, HOTA and ID switches (stage P9).

* ``MOTA = 1 - (FN + FP + IDSW) / GT`` is dominated by detection quality.
* ``IDF1 = 2 IDTP / (2 IDTP + IDFP + IDFN)`` measures identity consistency over the
  whole sequence.
* ``HOTA = sqrt(DetA * AssA)`` balances detection and association.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_mot_ground_truth(
    path: str | Path, one_based_frames: bool = True
) -> pd.DataFrame:
    """Loads a MOTChallenge-style ground-truth file.

    Each line is ``frame, id, left, top, width, height, ...`` (comma separated).

    Args:
        path: Text file in MOT format, for example exported from CVAT.
        one_based_frames: Whether frame numbers start at 1 (MOTChallenge) and must
            be shifted to this project's 0-based ``frame_id``.

    Returns:
        Table with the columns ``frame_id``, ``track_id``, ``x1``, ``y1``, ``x2``
        and ``y2``.
    """
    raw = pd.read_csv(path, header=None, usecols=range(6))
    raw.columns = ["frame_id", "track_id", "left", "top", "width", "height"]
    if one_based_frames:
        raw["frame_id"] = raw["frame_id"] - 1
    raw["x1"] = raw["left"]
    raw["y1"] = raw["top"]
    raw["x2"] = raw["left"] + raw["width"]
    raw["y2"] = raw["top"] + raw["height"]
    return raw[["frame_id", "track_id", "x1", "y1", "x2", "y2"]]


def evaluate_tracking(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Computes tracking metrics against annotated tracks.

    Wrap ``py-motmetrics`` (MOTA, IDF1, ID switches, fragmentation) and, for HOTA,
    TrackEval. Evaluate only the annotated frame range.

    Args:
        predictions: ``tracks.csv`` table.
        ground_truth: Output of :func:`load_mot_ground_truth`.
        iou_threshold: Minimum IoU for a prediction to match a ground-truth box.

    Returns:
        Mapping with at least ``"mota"``, ``"idf1"``, ``"hota"``, ``"id_switches"``
        and ``"fragmentations"``.

    Raises:
        NotImplementedError: Stub; to be implemented by member C.
    """
    # TODO(C): build per-frame distance matrices with 1 - iou_matrix and feed
    #   them to motmetrics.
    raise NotImplementedError("TODO(C): implement evaluate_tracking")
