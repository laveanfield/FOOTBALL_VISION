"""Tiled inference for small objects such as the ball (stage P4, optional).

Running the detector on overlapping crops (the idea behind SAHI) increases the
effective resolution, at the cost of speed and duplicate detections at tile borders.
"""

from __future__ import annotations

import numpy as np

from football_vision.boxes import iou_matrix
from football_vision.datatypes import Detections, Frame
from football_vision.detection.base import Detector


def _axis_starts(length: int, tile_len: int, overlap: float) -> list[int]:
    """Computes monotonically increasing tile start coordinates along one axis."""
    if tile_len >= length:
        return [0]
    step = max(1, int(round(tile_len * (1.0 - overlap))))
    starts = [0]
    pos = 0
    while pos + tile_len < length:
        pos = min(pos + step, length - tile_len)
        if pos != starts[-1]:
            starts.append(pos)
    return starts


def _nms_single_class(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> list[int]:
    """Applies greedy NMS to boxes of a single class, returning kept indices."""
    order = np.argsort(-scores, kind="mergesort")
    kept: list[int] = []
    while len(order) > 0:
        best = int(order[0])
        kept.append(best)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = iou_matrix(boxes[best : best + 1], boxes[rest])[0]
        order = rest[ious <= iou_threshold]
    return kept


def tile_image(
    frame: Frame, tile_size: tuple[int, int], overlap: float
) -> list[tuple[Frame, tuple[int, int]]]:
    """Splits a frame into overlapping tiles.

    Args:
        frame: BGR image.
        tile_size: Tile ``(width, height)`` in pixels.
        overlap: Fraction of overlap between neighboring tiles, in ``[0, 1)``.

    Returns:
        List of ``(tile, (x_offset, y_offset))`` where the offset is the top-left
        corner of the tile in the full frame.

    Raises:
        ValueError: If ``frame`` is empty, ``tile_size`` is non-positive, or
            ``overlap`` is outside ``[0, 1)``.
    """
    if frame.ndim < 2 or frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError(f"Expected non-empty frame, got shape {frame.shape}")
    tile_w, tile_h = int(tile_size[0]), int(tile_size[1])
    if tile_w <= 0 or tile_h <= 0:
        raise ValueError(f"tile_size dimensions must be positive, got {tile_size}")
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    img_h, img_w = int(frame.shape[0]), int(frame.shape[1])
    x_starts = _axis_starts(img_w, tile_w, overlap)
    y_starts = _axis_starts(img_h, tile_h, overlap)

    tiles: list[tuple[Frame, tuple[int, int]]] = []
    for y0 in y_starts:
        y1 = min(y0 + tile_h, img_h)
        for x0 in x_starts:
            x1 = min(x0 + tile_w, img_w)
            tiles.append((frame[y0:y1, x0:x1], (x0, y0)))
    return tiles


def merge_tile_detections(
    per_tile: list[Detections],
    offsets: list[tuple[int, int]],
    iou_threshold: float = 0.5,
) -> Detections:
    """Merges tile detections into full-frame coordinates and removes duplicates.

    Args:
        per_tile: Detections of each tile in tile coordinates.
        offsets: Top-left corner ``(x, y)`` of each tile in the full frame.
        iou_threshold: Boxes overlapping more than this are merged (NMS).

    Returns:
        The merged detections in full-frame coordinates.

    Raises:
        ValueError: If ``per_tile`` and ``offsets`` have different lengths, or if
            ``iou_threshold`` is outside ``[0, 1]``.
    """
    if len(per_tile) != len(offsets):
        raise ValueError(
            f"per_tile ({len(per_tile)}) and offsets ({len(offsets)}) length mismatch"
        )
    if not (0.0 <= iou_threshold <= 1.0):
        raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")

    boxes_parts: list[np.ndarray] = []
    scores_parts: list[np.ndarray] = []
    class_parts: list[np.ndarray] = []
    for det, (ox, oy) in zip(per_tile, offsets, strict=True):
        if len(det) == 0:
            continue
        shift = np.array([ox, oy, ox, oy], dtype=np.float32)
        boxes_parts.append(det.boxes_xyxy.astype(np.float32) + shift)
        scores_parts.append(det.scores.astype(np.float32))
        class_parts.append(det.class_ids.astype(np.int64))

    if not boxes_parts:
        return Detections.empty()

    all_boxes = np.vstack(boxes_parts)
    all_scores = np.concatenate(scores_parts)
    all_classes = np.concatenate(class_parts)

    kept_global: list[int] = []
    for cls_id in np.unique(all_classes):
        cls_indices = np.flatnonzero(all_classes == cls_id)
        local_kept = _nms_single_class(
            all_boxes[cls_indices], all_scores[cls_indices], iou_threshold
        )
        kept_global.extend(int(cls_indices[idx]) for idx in local_kept)

    kept_arr = np.asarray(kept_global, dtype=np.int64)
    final_order = kept_arr[np.argsort(-all_scores[kept_arr], kind="mergesort")]
    return Detections(
        boxes_xyxy=all_boxes[final_order],
        scores=all_scores[final_order],
        class_ids=all_classes[final_order],
    )


class TiledDetector(Detector):
    """Runs a base detector on overlapping tiles and merges the results.

    Attributes:
        base: Detector applied to every tile.
        tile_size: Tile ``(width, height)`` in pixels.
        overlap: Fraction of overlap between tiles.
        iou_threshold: IoU threshold used when merging tile detections.
    """

    def __init__(
        self,
        base: Detector,
        tile_size: tuple[int, int] = (640, 640),
        overlap: float = 0.2,
        iou_threshold: float = 0.5,
    ) -> None:
        """Creates a tiled detector.

        Args:
            base: Detector applied to every tile.
            tile_size: Tile ``(width, height)`` in pixels.
            overlap: Fraction of overlap between tiles.
            iou_threshold: IoU threshold used when merging tile detections.
        """
        self.base = base
        self.tile_size = tile_size
        self.overlap = overlap
        self.iou_threshold = iou_threshold

    def detect(self, frame: Frame) -> Detections:
        """Detects objects tile by tile.

        Args:
            frame: BGR image.

        Returns:
            The merged detections in full-frame coordinates.
        """
        tiles = tile_image(frame, self.tile_size, self.overlap)
        per_tile = [self.base.detect(tile) for tile, _ in tiles]
        offsets = [offset for _, offset in tiles]
        return merge_tile_detections(per_tile, offsets, self.iou_threshold)
