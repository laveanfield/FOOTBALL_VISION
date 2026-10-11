"""Data association between predicted tracks and new detections."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

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
        NotImplementedError: Stub; to be implemented by member C.
    """
    # TODO(C): handle empty matrices before calling linear_sum_assignment.
    raise NotImplementedError("TODO(C): implement solve_assignment")
