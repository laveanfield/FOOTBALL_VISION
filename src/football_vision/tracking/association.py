"""Data association between predicted tracks and new detections."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from football_vision.boxes import iou_matrix


def build_iou_cost(
    track_boxes: npt.ArrayLike,
    detection_boxes: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Builds the association cost ``1 - IoU`` between tracks and detections.

    Args:
        track_boxes: Predicted track boxes, shape ``(T, 4)``.
        detection_boxes: Detection boxes, shape ``(D, 4)``.

    Returns:
        Cost matrix of shape ``(T, D)`` with values in ``[0, 1]``.
    """
    return 1.0 - iou_matrix(track_boxes, detection_boxes)


def build_pitch_distance_cost(
    track_points: npt.ArrayLike,
    detection_points: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Builds the Euclidean distance cost matrix in pitch meters.

    Args:
        track_points: Predicted track positions ``(X, Y)``, shape ``(T, 2)``.
        detection_points: Projected foot positions ``(X, Y)``, shape ``(D, 2)``.

    Returns:
        Distance matrix of shape ``(T, D)`` in meters; non-finite entries are ``inf``.
    """
    t_pts = np.asarray(track_points, dtype=np.float64).reshape(-1, 2)
    d_pts = np.asarray(detection_points, dtype=np.float64).reshape(-1, 2)
    if len(t_pts) == 0 or len(d_pts) == 0:
        return np.zeros((len(t_pts), len(d_pts)), dtype=np.float64)
    diff = t_pts[:, None, :] - d_pts[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return np.where(np.isfinite(dist), dist, np.inf)


def solve_assignment(
    cost: npt.NDArray[np.float64],
    max_cost: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Solves the track-to-detection assignment with gating.

    Solve ``min sum C[i, pi(i)]`` over injective matchings with the Hungarian
    algorithm (``scipy.optimize.linear_sum_assignment``), then reject matched pairs
    whose cost exceeds ``max_cost``; those rows and columns become unmatched.

    Args:
        cost: Cost matrix of shape ``(T, D)``; it may be empty.
        max_cost: Gate; pairs with a larger cost are not matched.

    Returns:
        Tuple ``(matches, unmatched_rows, unmatched_cols)`` where ``matches`` lists
        ``(row, col)`` pairs, sorted by row.

    Raises:
        ValueError: If ``cost`` is not 2D or contains ``NaN`` entries.
    """
    cost_mat = np.asarray(cost, dtype=np.float64)
    if cost_mat.ndim != 2:
        raise ValueError(f"Expected 2D cost matrix, got shape {cost_mat.shape}")
    n_rows, n_cols = cost_mat.shape
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))
    if np.isnan(cost_mat).any():
        raise ValueError("Cost matrix must not contain NaN values")

    work_cost = cost_mat
    if np.isposinf(cost_mat).any() and np.isfinite(max_cost):
        # Replace +inf with a finite sentinel above max_cost so Hungarian stays valid.
        finite = cost_mat[np.isfinite(cost_mat)]
        max_abs = float(np.max(np.abs(finite))) if finite.size else 0.0
        sentinel = max(float(max_cost), 0.0) + max(n_rows, n_cols) * max_abs + 1e6
        work_cost = np.where(np.isposinf(cost_mat), sentinel, cost_mat)

    row_ind, col_ind = linear_sum_assignment(work_cost)
    matches = [
        (int(r), int(c))
        for r, c in zip(row_ind, col_ind, strict=True)
        if cost_mat[r, c] <= max_cost
    ]
    matches.sort(key=lambda pair: pair[0])
    matched_rows = {r for r, _ in matches}
    matched_cols = {c for _, c in matches}
    unmatched_rows = [r for r in range(n_rows) if r not in matched_rows]
    unmatched_cols = [c for c in range(n_cols) if c not in matched_cols]
    return matches, unmatched_rows, unmatched_cols
