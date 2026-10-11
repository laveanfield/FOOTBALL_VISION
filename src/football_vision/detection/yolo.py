"""Ultralytics YOLO detector wrapper (stage P4).

The COCO-pretrained model only knows ``person`` and ``sports ball``. It cannot tell
players from goalkeepers or referees; that is done in stage P6 (or by fine-tuning).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from football_vision.config import DetectionConfig
from football_vision.datatypes import Detections, Frame
from football_vision.detection.base import Detector


class YoloDetector(Detector):
    """Detector backed by an Ultralytics YOLO model.

    Attributes:
        cfg: Detection settings used for every prediction.
    """

    def __init__(self, cfg: DetectionConfig) -> None:
        """Loads the YOLO weights.

        Args:
            cfg: Detection settings (weights, resolution, thresholds, classes).

        Raises:
            ImportError: If the optional ``ultralytics`` package is not installed.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for YoloDetector; "
                "install it with `pip install -e .[dl]`."
            ) from exc
        self.cfg = cfg
        self._model = YOLO(cfg.weights)

    def _predict(self, source: Any) -> Any:
        """Runs the model with the configured settings."""
        return self._model.predict(
            source=source,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            classes=list(self.cfg.classes),
            device=self.cfg.device,
            half=self.cfg.half,
            verbose=False,
        )

    @staticmethod
    def _to_detections(result: Any) -> Detections:
        """Converts one Ultralytics result into :class:`Detections`."""
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return Detections.empty()
        return Detections(
            boxes_xyxy=boxes.xyxy.cpu().numpy().astype(np.float32),
            scores=boxes.conf.cpu().numpy().astype(np.float32),
            class_ids=boxes.cls.cpu().numpy().astype(np.int64),
        )

    def detect(self, frame: Frame) -> Detections:
        """Detects objects in one frame.

        Args:
            frame: BGR image of shape ``(H, W, 3)``.

        Returns:
            Detections of the configured classes above the confidence threshold.
        """
        return self._to_detections(self._predict(frame)[0])

    def detect_batch(self, frames: Sequence[Frame]) -> list[Detections]:
        """Detects objects in several frames with one batched forward pass.

        Args:
            frames: BGR images.

        Returns:
            One :class:`~football_vision.datatypes.Detections` per input frame.
        """
        return [self._to_detections(r) for r in self._predict(list(frames))]
