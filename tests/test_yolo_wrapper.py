"""Tests of the YOLO wrapper with a fake ``ultralytics`` module (no weights needed)."""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from football_vision.config import DetectionConfig
from football_vision.detection.yolo import YoloDetector


class FakeTensor:
    """Minimal stand-in for a torch tensor."""

    def __init__(self, array: np.ndarray) -> None:
        """Wraps an array."""
        self.array = array

    def cpu(self) -> FakeTensor:
        """Returns the tensor itself."""
        return self

    def numpy(self) -> np.ndarray:
        """Returns the wrapped array."""
        return self.array


class FakeBoxes:
    """Stand-in for ``ultralytics`` boxes with two detections."""

    xyxy = FakeTensor(np.array([[10.0, 20.0, 30.0, 80.0], [50.0, 60.0, 54.0, 64.0]]))
    conf = FakeTensor(np.array([0.9, 0.4]))
    cls = FakeTensor(np.array([0.0, 32.0]))

    def __len__(self) -> int:
        """Returns the number of boxes."""
        return 2


class FakeResult:
    """Stand-in for one ``ultralytics`` result."""

    def __init__(self, boxes: Any) -> None:
        """Stores the boxes."""
        self.boxes = boxes


class FakeYolo:
    """Stand-in for ``ultralytics.YOLO`` that records the prediction arguments."""

    last_kwargs: dict[str, Any] = {}
    empty = False

    def __init__(self, weights: str) -> None:
        """Remembers the weights name."""
        self.weights = weights

    def predict(self, source: Any, **kwargs: Any) -> list[FakeResult]:
        """Returns one fake result per input image."""
        FakeYolo.last_kwargs = kwargs
        count = len(source) if isinstance(source, list) else 1
        return [
            FakeResult(None if FakeYolo.empty else FakeBoxes()) for _ in range(count)
        ]


@pytest.fixture()
def fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> type[FakeYolo]:
    """Installs the fake ``ultralytics`` module for one test."""
    FakeYolo.empty = False
    monkeypatch.setitem(
        sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYolo)
    )
    return FakeYolo


def test_detect_converts_results_and_forwards_settings(
    fake_ultralytics: type[FakeYolo],
) -> None:
    """Checks the conversion to Detections and the arguments given to the model."""
    detector = YoloDetector(
        DetectionConfig(weights="w.pt", imgsz=640, conf=0.3, classes=(0, 32))
    )
    dets = detector.detect(np.zeros((90, 160, 3), dtype=np.uint8))
    assert dets.boxes_xyxy.dtype == np.float32 and dets.boxes_xyxy.shape == (2, 4)
    assert dets.class_ids.tolist() == [0, 32] and dets.class_ids.dtype == np.int64
    assert dets.scores.tolist() == pytest.approx([0.9, 0.4])
    sent = fake_ultralytics.last_kwargs
    assert sent["imgsz"] == 640 and sent["conf"] == 0.3 and sent["classes"] == [0, 32]
    assert sent["verbose"] is False


def test_detect_batch_returns_one_result_per_frame(
    fake_ultralytics: type[FakeYolo],
) -> None:
    """Checks that batched inference returns a Detections object per frame."""
    detector = YoloDetector(DetectionConfig())
    frames = [np.zeros((90, 160, 3), dtype=np.uint8)] * 3
    assert [len(d) for d in detector.detect_batch(frames)] == [2, 2, 2]


def test_no_boxes_gives_empty_detections(fake_ultralytics: type[FakeYolo]) -> None:
    """Checks that a result without boxes becomes an empty Detections."""
    fake_ultralytics.empty = True
    assert (
        len(YoloDetector(DetectionConfig()).detect(np.zeros((9, 9, 3), dtype=np.uint8)))
        == 0
    )


def test_missing_dependency_has_an_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks the error raised when ultralytics is not installed."""
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    with pytest.raises(ImportError, match="pip install"):
        YoloDetector(DetectionConfig())
