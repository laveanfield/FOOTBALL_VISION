"""Bounding-box utilities shared by detection, tracking, analytics and evaluation.

All boxes are ``(x1, y1, x2, y2)`` in pixels unless a function says otherwise.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def xyxy_to_xywh(boxes: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Converts corner boxes to ``(x, y, w, h)`` with ``(x, y)`` the top-left corner.

    Args:
        boxes: Array of shape ``(N, 4)`` in ``(x1, y1, x2, y2)`` format.

    Returns:
        Array of shape ``(N, 4)`` in ``(x, y, w, h)`` format.
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1)


def xywh_to_xyxy(boxes: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Converts ``(x, y, w, h)`` boxes to corner format.

    Args:
        boxes: Array of shape ``(N, 4)`` in ``(x, y, w, h)`` format.

    Returns:
        Array of shape ``(N, 4)`` in ``(x1, y1, x2, y2)`` format.
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], axis=1)


def box_areas(boxes: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Computes the area of each box.

    Args:
        boxes: Array of shape ``(N, 4)`` in ``(x1, y1, x2, y2)`` format.

    Returns:
        Array of shape ``(N,)``; degenerate boxes have area zero.
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def iou_matrix(
    boxes_a: npt.ArrayLike, boxes_b: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Computes pairwise intersection-over-union between two box sets.

    Args:
        boxes_a: Array of shape ``(N, 4)``.
        boxes_b: Array of shape ``(M, 4)``.

    Returns:
        Array of shape ``(N, M)`` where entry ``(i, j)`` is the IoU of
        ``boxes_a[i]`` and ``boxes_b[j]``.
    """
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = box_areas(a)[:, None] + box_areas(b)[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def foot_points(boxes_xyxy: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Returns the ground-contact point of each box (bottom-center).

    The homography is only valid for points on the pitch plane, so the bottom
    center of a player box approximates where the feet touch the ground. The box
    center must not be projected.

    Args:
        boxes_xyxy: Array of shape ``(N, 4)`` in ``(x1, y1, x2, y2)`` format.

    Returns:
        Array of shape ``(N, 2)`` with ``((x1 + x2) / 2, y2)`` for every box.
    """
    b = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    return np.stack([(b[:, 0] + b[:, 2]) / 2.0, b[:, 3]], axis=1)


def clip_boxes(
    boxes: npt.ArrayLike, width: int, height: int
) -> npt.NDArray[np.float64]:
    """Clips boxes to the image rectangle.

    Args:
        boxes: Array of shape ``(N, 4)`` in ``(x1, y1, x2, y2)`` format.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Array of shape ``(N, 4)`` with all coordinates inside the image.
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, width)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, height)
    return b
