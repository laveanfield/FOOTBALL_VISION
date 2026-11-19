"""Tracking metrics: MOTA, IDF1, HOTA and ID switches (stage P9).

* ``MOTA = 1 - (FN + FP + IDSW) / GT`` is dominated by detection quality.
* ``IDF1 = 2 IDTP / (2 IDTP + IDFP + IDFN)`` measures identity consistency over the
  whole sequence.
* ``HOTA = sqrt(DetA * AssA)`` balances detection and association.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import motmetrics as mm
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from football_vision.boxes import iou_matrix

_BOX_COLS = ["x1", "y1", "x2", "y2"]
_HOTA_ALPHAS = np.round(np.arange(0.05, 0.96, 0.05), 2)


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


def _group_frame_boxes(
    table: pd.DataFrame,
) -> dict[int, tuple[list[int], np.ndarray]]:
    """Groups track ids and xyxy boxes by integer frame_id."""
    grouped: dict[int, tuple[list[int], np.ndarray]] = {}
    if len(table) == 0:
        return grouped
    for fid, group in table.groupby("frame_id", sort=True):
        ids = [int(tid) for tid in group["track_id"].tolist()]
        boxes = group[_BOX_COLS].to_numpy(dtype=np.float64)
        grouped[int(fid)] = (ids, boxes)
    return grouped


def _compute_hota(
    gt_by_frame: dict[int, tuple[list[int], np.ndarray]],
    pred_by_frame: dict[int, tuple[list[int], np.ndarray]],
    frame_ids: list[int],
) -> float:
    """Computes HOTA integrated over alpha in 0.05..0.95 (Luiten et al.)."""
    gt_counts = Counter(gid for f in frame_ids for gid in gt_by_frame.get(f, ([],))[0])
    pred_counts = Counter(
        pid for f in frame_ids for pid in pred_by_frame.get(f, ([],))[0]
    )
    total_gt = sum(gt_counts.values())
    total_pred = sum(pred_counts.values())
    if total_gt == 0 or total_pred == 0:
        return 0.0

    # Precompute per-frame IoU matrices and ID lists once across all alpha thresholds.
    frame_ious: list[tuple[list[int], list[int], np.ndarray]] = []
    for fid in frame_ids:
        g_ids, g_boxes = gt_by_frame.get(fid, ([], np.empty((0, 4), dtype=np.float64)))
        p_ids, p_boxes = pred_by_frame.get(
            fid, ([], np.empty((0, 4), dtype=np.float64))
        )
        ious = iou_matrix(g_boxes, p_boxes)
        frame_ious.append((g_ids, p_ids, ious))

    hota_scores: list[float] = []
    for alpha in _HOTA_ALPHAS:
        potential_tpa: Counter[tuple[int, int]] = Counter()
        for g_ids, p_ids, ious in frame_ious:
            if ious.size == 0:
                continue
            rows, cols = np.where(ious >= alpha)
            for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
                potential_tpa[(g_ids[r], p_ids[c])] += 1

        matched_pairs: list[tuple[int, int]] = []
        tpa: Counter[tuple[int, int]] = Counter()
        for g_ids, p_ids, ious in frame_ious:
            if ious.size == 0:
                continue
            n_g, n_p = len(g_ids), len(p_ids)
            score = np.full((n_g, n_p), -1.0, dtype=np.float64)
            for r in range(n_g):
                gid = g_ids[r]
                for c in range(n_p):
                    if ious[r, c] < alpha:
                        continue
                    pid = p_ids[c]
                    pt = potential_tpa[(gid, pid)]
                    denom = gt_counts[gid] + pred_counts[pid] - pt
                    assoc_sim = float(pt) / float(denom) if denom > 0 else 0.0
                    score[r, c] = assoc_sim + 1e-6 * float(ious[r, c])
            # Maximize score via minimum-cost assignment on valid entries.
            cost = np.where(score >= 0.0, -score, 1e6)
            r_ind, c_ind = linear_sum_assignment(cost)
            for r, c in zip(r_ind.tolist(), c_ind.tolist(), strict=True):
                if score[r, c] >= 0.0:
                    pair = (g_ids[r], p_ids[c])
                    matched_pairs.append(pair)
                    tpa[pair] += 1

        tp = len(matched_pairs)
        if tp == 0:
            hota_scores.append(0.0)
            continue
        det_a = float(tp) / float(total_gt + total_pred - tp)
        ass_iou_sum = sum(
            float(tpa[g, p]) / float(gt_counts[g] + pred_counts[p] - tpa[g, p])
            for g, p in matched_pairs
        )
        ass_a = ass_iou_sum / float(tp)
        hota_scores.append(math.sqrt(det_a * ass_a))

    return float(np.mean(hota_scores))


def _safe_float(value: float, default: float = 0.0) -> float:
    """Converts NaN/inf motmetrics values to a finite fallback."""
    val = float(value)
    return val if math.isfinite(val) else default


def evaluate_tracking(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Computes tracking metrics against annotated tracks.

    Computes ``mota``, ``idf1``, ``id_switches`` and ``fragmentations`` via
    ``py-motmetrics``, plus ``hota`` integrated over IoU thresholds ``0.05..0.95``.
    Evaluates only frames inside the annotated ground-truth frame range.

    Args:
        predictions: ``tracks.csv`` table.
        ground_truth: Output of :func:`load_mot_ground_truth`.
        iou_threshold: Minimum IoU for a prediction to match a ground-truth box.

    Returns:
        Mapping with ``"mota"``, ``"idf1"``, ``"hota"``, ``"id_switches"`` and
        ``"fragmentations"``.

    Raises:
        ValueError: If ``iou_threshold`` is outside ``(0, 1]``.
    """
    if not (0.0 < iou_threshold <= 1.0):
        raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")

    if len(ground_truth) == 0:
        return {
            "mota": 0.0,
            "idf1": 0.0,
            "hota": 0.0,
            "id_switches": 0.0,
            "fragmentations": 0.0,
        }

    gt_min = int(ground_truth["frame_id"].min())
    gt_max = int(ground_truth["frame_id"].max())
    preds = (
        predictions[predictions["frame_id"].between(gt_min, gt_max)]
        if len(predictions) > 0
        else predictions.iloc[0:0]
    )

    gt_by_frame = _group_frame_boxes(ground_truth)
    pred_by_frame = _group_frame_boxes(preds)
    frame_ids = sorted(set(gt_by_frame) | set(pred_by_frame))

    acc = mm.MOTAccumulator(auto_id=False)
    empty_boxes = np.empty((0, 4), dtype=np.float64)
    for fid in frame_ids:
        g_ids, g_boxes = gt_by_frame.get(fid, ([], empty_boxes))
        p_ids, p_boxes = pred_by_frame.get(fid, ([], empty_boxes))
        ious = iou_matrix(g_boxes, p_boxes)
        dist = 1.0 - ious
        dist[ious < iou_threshold] = np.nan
        acc.update(g_ids, p_ids, dist, frameid=fid)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["mota", "idf1", "num_switches", "num_fragmentations"],
        name="eval",
    )
    row = summary.iloc[0]
    hota = _compute_hota(gt_by_frame, pred_by_frame, frame_ids)
    return {
        "mota": _safe_float(row["mota"]),
        "idf1": _safe_float(row["idf1"]),
        "hota": float(hota),
        "id_switches": _safe_float(row["num_switches"]),
        "fragmentations": _safe_float(row["num_fragmentations"]),
    }
