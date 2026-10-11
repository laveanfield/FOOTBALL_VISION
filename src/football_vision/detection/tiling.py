"""Tiled inference for small objects such as the ball (stage P4, optional).

Running the detector on overlapping crops (the idea behind SAHI) increases the
effective resolution, at the cost of speed and duplicate detections at tile borders.
"""

from __future__ import annotations

from football_vision.datatypes import Detections, Frame
from football_vision.detection.base import Detector


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
        NotImplementedError: Stub; to be implemented by member C.
    """
    # TODO(C): step = tile * (1 - overlap); make sure the last tile reaches the border.
    raise NotImplementedError("TODO(C): implement tile_image")


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
        NotImplementedError: Stub; to be implemented by member C.
    """
    # TODO(C): shift boxes by the offsets, concatenate, apply class-aware NMS.
    raise NotImplementedError("TODO(C): implement merge_tile_detections")


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
