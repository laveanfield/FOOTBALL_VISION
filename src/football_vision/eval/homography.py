"""Homography metrics: reprojection error in meters and temporal jitter (stage P9).

Errors are measured in *meters on the pitch*: one pixel is a few centimeters near
the camera and several decimeters far away, so pixel errors are not comparable
across the image.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd

from football_vision.artifacts import HomographySequence
from football_vision.datatypes import Homography, Points2D
from football_vision.homography.dlt import reprojection_error
from football_vision.homography.geometry import apply_homography, invert_homography
from football_vision.pitch.template import PitchTemplate


def reprojection_error_m(
    homography: Homography, image_points: Points2D, pitch_points: Points2D
) -> npt.NDArray[np.float64]:
    """Measures landmark errors in meters on the pitch.

    Args:
        homography: Matrix mapping image to pitch.
        image_points: Clicked landmark pixels, shape ``(N, 2)``.
        pitch_points: True landmark positions in meters, shape ``(N, 2)``.

    Returns:
        Array of shape ``(N,)`` with errors in meters.
    """
    return reprojection_error(homography, image_points, pitch_points)


def summarize_errors(errors: npt.ArrayLike) -> dict[str, float]:
    """Summarizes an error sample with robust statistics.

    Report the median and the 95th percentile, not only the mean: a few broken
    frames inflate the mean.

    Args:
        errors: Error values.

    Returns:
        Mapping with ``n``, ``mean``, ``median``, ``p95`` and ``max`` (NaN if empty).
    """
    values = np.asarray(errors, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "n": 0.0,
            "mean": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }
    return {
        "n": float(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def evaluate_homography_sequence(
    sequence: HomographySequence,
    annotations: Mapping[int, Mapping[str, tuple[float, float]]],
    template: PitchTemplate | None = None,
) -> pd.DataFrame:
    """Evaluates stored homographies on held-out landmark annotations.

    Annotated landmarks must not have been used to estimate the homographies.

    Args:
        sequence: Per-frame homographies.
        annotations: Mapping from frame id to landmark name to clicked pixel.
        template: Pitch model; defaults to the 105 x 68 m template.

    Returns:
        Table with the columns ``frame_id``, ``landmark`` and ``error_m``; frames
        without a valid homography are skipped.
    """
    model = template or PitchTemplate()
    known = model.landmarks()
    records = []
    for frame_id, landmarks in annotations.items():
        if not sequence.is_valid(frame_id):
            continue
        names = [n for n in landmarks if n in known]
        if not names:
            continue
        image_points = np.array([landmarks[n] for n in names], dtype=np.float64)
        errors = reprojection_error_m(
            sequence.matrix_for(frame_id), image_points, model.landmark_array(names)
        )
        records.extend(
            {"frame_id": frame_id, "landmark": n, "error_m": float(e)}
            for n, e in zip(names, errors, strict=True)
        )
    return pd.DataFrame(records, columns=["frame_id", "landmark", "error_m"])


def reference_point_jitter_px(
    matrices: npt.NDArray[np.float64], reference_points_pitch: Points2D
) -> npt.NDArray[np.float64]:
    """Measures frame-to-frame motion of projected pitch points in the image.

    Project fixed pitch points into the image with every ``H_t^-1`` and measure how
    far they move between consecutive frames. For a static camera this is pure jitter;
    for a moving camera compare the raw and the smoothed sequence (ablation A3).

    Args:
        matrices: Array of shape ``(T, 3, 3)`` mapping image to pitch.
        reference_points_pitch: Array of shape ``(K, 2)`` in meters.

    Returns:
        Array of shape ``(T - 1,)`` with the mean displacement in pixels per step.
    """
    projected = np.stack(
        [
            apply_homography(invert_homography(m), reference_points_pitch)
            for m in matrices
        ]
    )
    steps = np.linalg.norm(np.diff(projected, axis=0), axis=2)
    return steps.mean(axis=1)
