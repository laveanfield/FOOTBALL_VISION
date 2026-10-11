"""Detection metrics: precision, recall and mAP (stage P9)."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def evaluate_detections(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    iou_threshold: float = 0.5,
    classes: Sequence[int] | None = None,
) -> dict[str, float]:
    """Evaluates detections against annotated frames.

    Report precision, recall, ``mAP@0.5`` and ``mAP@0.5:0.95`` per class. Match
    predictions to ground truth greedily by decreasing score and IoU (see
    ``football_vision.boxes.iou_matrix``), build the precision-recall curve, and
    integrate it (COCO uses 101-point interpolation). ``pycocotools`` or the
    Ultralytics validator can serve as a cross-check.

    Args:
        predictions: ``detections.csv`` table restricted to annotated frames.
        ground_truth: Table with the columns ``frame_id``, ``x1``, ``y1``, ``x2``,
            ``y2`` and ``cls``.
        iou_threshold: IoU above which a prediction counts as a true positive.
        classes: Class ids to evaluate; ``None`` evaluates all classes.

    Returns:
        Mapping from metric name (for example ``"map50"`` or
        ``"precision_cls0"``) to value.

    Raises:
        NotImplementedError: Stub; to be implemented by member C.
    """
    # TODO(C): evaluate only on frames that have ground truth and keep the
    #   test split untouched.
    raise NotImplementedError("TODO(C): implement evaluate_detections")
