"""Homography of a keyframe, from clicked landmarks or automatically."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from football_vision.datatypes import Frame, Homography, Points2D
from football_vision.errors import HomographyError
from football_vision.homography.geometry import invert_homography
from football_vision.pitch.template import PitchTemplate

MIN_CORRESPONDENCES = 4


def homography_from_landmarks(
    image_points: Points2D,
    pitch_points: Points2D,
    ransac_threshold_px: float = 3.0,
) -> tuple[Homography, int]:
    """Estimates the image-to-pitch homography from landmark correspondences.

    This is the OpenCV baseline (semi-automatic mode): ``cv2.findHomography`` with
    RANSAC. The model is fitted from pitch to image so that the RANSAC threshold is
    measured in pixels, then inverted. It is the reference to compare against the
    team's own DLT (``homography.dlt``).

    Args:
        image_points: Array of shape ``(N, 2)`` with landmark pixels, ``N >= 4``.
        pitch_points: Array of shape ``(N, 2)`` with the matching pitch meters.
        ransac_threshold_px: Inlier threshold in pixels.

    Returns:
        Tuple ``(homography, n_inliers)`` where the matrix maps image to pitch.

    Raises:
        HomographyError: If fewer than four correspondences are given or OpenCV
            cannot fit a model.
    """
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    pit = np.asarray(pitch_points, dtype=np.float64).reshape(-1, 2)
    if len(img) != len(pit) or len(img) < MIN_CORRESPONDENCES:
        raise HomographyError(
            f"Need at least {MIN_CORRESPONDENCES} matching landmarks, got {len(img)}"
        )
    h_pitch_to_image, mask = cv2.findHomography(
        pit, img, cv2.RANSAC, ransac_threshold_px
    )
    if h_pitch_to_image is None:
        raise HomographyError("cv2.findHomography failed; check the landmarks")
    n_inliers = int(mask.sum()) if mask is not None else len(img)
    return invert_homography(h_pitch_to_image), n_inliers


def homography_from_landmark_dict(
    landmarks_image: Mapping[str, Sequence[float]],
    template: PitchTemplate | None = None,
    ransac_threshold_px: float = 3.0,
) -> tuple[Homography, int]:
    """Estimates the homography from named landmarks clicked on one frame.

    Args:
        landmarks_image: Mapping from landmark name (see ``PitchTemplate.landmarks``)
            to the clicked ``(x, y)`` pixel. Unknown names are ignored.
        template: Pitch model; defaults to the 105 x 68 m template.
        ransac_threshold_px: Inlier threshold in pixels.

    Returns:
        Tuple ``(homography, n_inliers)`` where the matrix maps image to pitch.

    Raises:
        HomographyError: If fewer than four known landmarks are given.
    """
    model = template or PitchTemplate()
    known = model.landmarks()
    names = [name for name in landmarks_image if name in known]
    if len(names) < MIN_CORRESPONDENCES:
        raise HomographyError(
            f"Need at least {MIN_CORRESPONDENCES} known landmarks, got {len(names)}"
        )
    image_points = np.array([landmarks_image[name] for name in names], dtype=np.float64)
    return homography_from_landmarks(
        image_points, model.landmark_array(names), ransac_threshold_px
    )


def estimate_keyframe_homography_auto(
    frame: Frame,
    segments: np.ndarray,
    template: PitchTemplate,
) -> Homography:
    """Estimates a keyframe homography without any clicks (hypothesize-and-verify).

    Choose two lines of each family from the detected segments, pair them with two
    lines of each family of the template, compute ``H`` from the four resulting
    intersections, and score each hypothesis by how well the remaining projected
    template lines match the edge map (see ``homography.quality.line_overlay_score``).
    Keep the best-scoring hypothesis.

    Args:
        frame: BGR keyframe.
        segments: Merged line segments ``(N, 4)`` of the frame.
        template: Pitch model providing the template lines.

    Returns:
        The ``(3, 3)`` homography mapping image to pitch coordinates.

    Raises:
        NotImplementedError: Stub; to be implemented by member B.
    """
    # TODO(B): enumerate line pairings (prune with orientation/order constraints),
    #   build the 4 correspondences, call the DLT, and rank by overlay score.
    raise NotImplementedError("TODO(B): implement estimate_keyframe_homography_auto")
