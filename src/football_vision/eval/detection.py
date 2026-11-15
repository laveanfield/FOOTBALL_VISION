"""Detection metrics: precision, recall and mAP (stage P9)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from football_vision.boxes import iou_matrix

_COCO_IOU_THRESHOLDS = np.round(np.arange(0.50, 0.96, 0.05), 2)
_RECALL_GRID = np.linspace(0.0, 1.0, 101)
_BOX_COLS = ["x1", "y1", "x2", "y2"]


def _compute_ap_101(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Computes COCO 101-point interpolated average precision."""
    if len(recalls) == 0:
        return 0.0
    m_rec = np.concatenate(([0.0], recalls, [1.0]))
    m_pre = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(len(m_pre) - 2, -1, -1):
        m_pre[idx] = max(m_pre[idx], m_pre[idx + 1])
    grid_idx = np.searchsorted(m_rec[:-1], _RECALL_GRID, side="left")
    grid_idx = np.clip(grid_idx, 0, len(m_pre) - 1)
    return float(np.mean(m_pre[grid_idx]))


def _evaluate_class(
    preds_cls: pd.DataFrame,
    gt_cls: pd.DataFrame,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """Matches predictions to GT greedily across thresholds for one class."""
    n_gt = len(gt_cls)
    n_thr = len(thresholds)
    if n_gt == 0 or len(preds_cls) == 0:
        return np.zeros(n_thr, dtype=np.float64), n_gt, np.zeros(n_thr), np.zeros(n_thr)

    gt_by_frame: dict[int, np.ndarray] = {}
    matched_by_frame: dict[int, np.ndarray] = {}
    for fid, group in gt_cls.groupby("frame_id"):
        boxes = group[_BOX_COLS].to_numpy(dtype=np.float64)
        gt_by_frame[int(fid)] = boxes
        matched_by_frame[int(fid)] = np.zeros((n_thr, len(boxes)), dtype=bool)

    sorted_preds = preds_cls.sort_values("conf", ascending=False, kind="mergesort")
    n_preds = len(sorted_preds)
    tp = np.zeros((n_thr, n_preds), dtype=np.float64)
    fp = np.zeros((n_thr, n_preds), dtype=np.float64)

    for pred_idx, row in enumerate(sorted_preds.itertuples(index=False)):
        fid = int(row.frame_id)
        gt_boxes = gt_by_frame.get(fid)
        if gt_boxes is None or len(gt_boxes) == 0:
            fp[:, pred_idx] = 1.0
            continue
        box = np.array([[row.x1, row.y1, row.x2, row.y2]], dtype=np.float64)
        ious = iou_matrix(box, gt_boxes)[0]
        matched = matched_by_frame[fid]
        for thr_idx, thr in enumerate(thresholds):
            available = np.where(~matched[thr_idx])[0]
            if len(available) == 0:
                fp[thr_idx, pred_idx] = 1.0
                continue
            avail_ious = ious[available]
            best_pos = int(np.argmax(avail_ious))
            if avail_ious[best_pos] >= thr:
                gt_idx = int(available[best_pos])
                matched[thr_idx, gt_idx] = True
                tp[thr_idx, pred_idx] = 1.0
            else:
                fp[thr_idx, pred_idx] = 1.0

    tp_cum = np.cumsum(tp, axis=1)
    fp_cum = np.cumsum(fp, axis=1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    recalls = tp_cum / float(n_gt)

    aps = np.array(
        [_compute_ap_101(recalls[i], precisions[i]) for i in range(n_thr)],
        dtype=np.float64,
    )
    tp_totals = tp_cum[:, -1]
    fp_totals = fp_cum[:, -1]
    return aps, n_gt, tp_totals, fp_totals


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
        ValueError: If ``iou_threshold`` is outside ``(0, 1]``.
    """
    if not (0.0 < iou_threshold <= 1.0):
        raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")

    gt_frames = (
        set(int(f) for f in ground_truth["frame_id"].unique())
        if len(ground_truth) > 0
        else set()
    )
    preds = (
        predictions[predictions["frame_id"].isin(gt_frames)]
        if gt_frames and len(predictions) > 0
        else predictions.iloc[0:0]
    )

    if classes is not None:
        eval_classes = sorted({int(c) for c in classes})
    elif len(ground_truth) > 0:
        eval_classes = sorted(int(c) for c in ground_truth["cls"].unique())
    else:
        eval_classes = []

    if not eval_classes:
        return {"precision": 0.0, "recall": 0.0, "map50": 0.0, "map50_95": 0.0}

    thresholds = np.concatenate(([float(iou_threshold)], _COCO_IOU_THRESHOLDS))
    results: dict[str, float] = {}
    ap50_list: list[float] = []
    ap50_95_list: list[float] = []
    total_tp = 0.0
    total_fp = 0.0
    total_gt = 0

    for cls_id in eval_classes:
        preds_cls = preds[preds["cls"] == cls_id]
        gt_cls = ground_truth[ground_truth["cls"] == cls_id]
        aps, n_gt, tp_totals, fp_totals = _evaluate_class(preds_cls, gt_cls, thresholds)

        tp_req = float(tp_totals[0])
        fp_req = float(fp_totals[0])
        n_pred_cls = len(preds_cls)
        prec_cls = tp_req / float(n_pred_cls) if n_pred_cls > 0 else 0.0
        rec_cls = tp_req / float(n_gt) if n_gt > 0 else 0.0
        ap50 = float(aps[1])  # _COCO_IOU_THRESHOLDS[0] is 0.50
        ap50_95 = float(np.mean(aps[1:]))

        results[f"precision_cls{cls_id}"] = prec_cls
        results[f"recall_cls{cls_id}"] = rec_cls
        results[f"ap50_cls{cls_id}"] = ap50
        results[f"ap50_95_cls{cls_id}"] = ap50_95

        ap50_list.append(ap50)
        ap50_95_list.append(ap50_95)
        total_tp += tp_req
        total_fp += fp_req
        total_gt += n_gt

    total_preds = sum(len(preds[preds["cls"] == c]) for c in eval_classes)
    overall_prec = total_tp / float(total_preds) if total_preds > 0 else 0.0
    overall_rec = total_tp / float(total_gt) if total_gt > 0 else 0.0
    return {
        "precision": overall_prec,
        "recall": overall_rec,
        "map50": float(np.mean(ap50_list)),
        "map50_95": float(np.mean(ap50_95_list)),
        **results,
    }
