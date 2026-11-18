"""Frame-to-frame alignment and homography propagation for moving cameras."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from football_vision.datatypes import Frame, Homography
from football_vision.errors import HomographyError
from football_vision.homography.geometry import normalize_homography


def propagate_homography(h_previous: Homography, g_current: Homography) -> Homography:
    """Chains the previous frame's homography with the inter-frame alignment.

    If ``g_current`` maps pixels of frame ``t`` to pixels of frame ``t - 1`` and
    ``h_previous`` maps frame ``t - 1`` to the pitch, then ``H_t = H_{t-1} @ G_t``.
    Errors multiply along the chain, which is why drift must be monitored.

    Args:
        h_previous: Homography of frame ``t - 1`` (image to pitch).
        g_current: Alignment of frame ``t`` onto frame ``t - 1``.

    Returns:
        The homography of frame ``t``, scaled so that ``H[2, 2] = 1``.
    """
    return normalize_homography(np.asarray(h_previous) @ np.asarray(g_current))


def estimate_frame_alignment(
    frame_previous: Frame,
    frame_current: Frame,
    exclude_boxes: np.ndarray | None = None,
    feature: Literal["orb", "sift", "akaze"] = "sift",
    ratio_test: float = 0.75,
    ransac_threshold_px: float = 3.0,
    min_inliers: int = 15,
) -> tuple[Homography, int]:
    """Estimates the homography that aligns the current frame onto the previous one.

    Detect features on static background (mask out player boxes), match them with a
    ratio test, and fit a homography with RANSAC. For a camera that only pans, tilts
    and zooms, this homography is exact for all background points (Appendix A3).

    The same ``exclude_boxes`` are masked in both frames, since players move little
    between two consecutive frames.

    Args:
        frame_previous: BGR frame ``t - 1``.
        frame_current: BGR frame ``t``.
        exclude_boxes: Optional ``(N, 4)`` boxes (``x1, y1, x2, y2``) whose pixels
            are ignored, typically the detected players.
        feature: Feature detector to use.
        ratio_test: Lowe ratio-test threshold.
        ransac_threshold_px: RANSAC inlier threshold in pixels.
        min_inliers: Minimum number of inliers for an accepted alignment.

    Returns:
        Tuple ``(g_current, n_inliers)`` where ``g_current`` maps frame ``t`` to
        frame ``t - 1``.

    Raises:
        HomographyError: If too few features are matched or the RANSAC fit has fewer
            than ``min_inliers`` inliers.
        ValueError: If ``feature`` is not ``"sift"``, ``"akaze"``, or ``"orb"``.
    """
    gray_previous = cv2.cvtColor(frame_previous, cv2.COLOR_BGR2GRAY)
    gray_current = cv2.cvtColor(frame_current, cv2.COLOR_BGR2GRAY)
    # Detection mask: 255 on background pixels, 0 inside the excluded boxes.
    mask = np.full(gray_current.shape, 255, dtype=np.uint8)
    if exclude_boxes is not None:
        for x1, y1, x2, y2 in np.asarray(exclude_boxes).reshape(-1, 4):
            mask[
                max(int(y1), 0) : max(int(y2), 0), max(int(x1), 0) : max(int(x2), 0)
            ] = 0
    if feature == "sift":
        detector = cv2.SIFT_create()
        norm = cv2.NORM_L2
    elif feature == "akaze":
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    elif feature == "orb":
        detector = cv2.ORB_create(nfeatures=2000)
        norm = cv2.NORM_HAMMING
    else:
        raise ValueError(f"Unsupported feature detector: {feature!r}")
    keypoints_previous, descriptors_previous = detector.detectAndCompute(
        gray_previous, mask
    )
    keypoints_current, descriptors_current = detector.detectAndCompute(
        gray_current, mask
    )
    if (
        descriptors_previous is None
        or descriptors_current is None
        or len(keypoints_previous) < 2
        or len(keypoints_current) < 2
    ):
        raise HomographyError("Too few features to align the frames")
    # For each current descriptor, its two nearest previous descriptors (Lowe test).
    pairs = cv2.BFMatcher(norm).knnMatch(descriptors_current, descriptors_previous, k=2)
    good = [
        p[0]
        for p in pairs
        if len(p) == 2 and p[0].distance < ratio_test * p[1].distance
    ]
    if len(good) < max(4, min_inliers):
        raise HomographyError(f"Only {len(good)} good matches, need {min_inliers}")
    points_current = np.float32([keypoints_current[m.queryIdx].pt for m in good])
    points_previous = np.float32([keypoints_previous[m.trainIdx].pt for m in good])
    g_current, inlier_mask = cv2.findHomography(
        points_current, points_previous, cv2.RANSAC, ransac_threshold_px
    )
    n_inliers = 0 if inlier_mask is None else int(inlier_mask.sum())
    if g_current is None or n_inliers < min_inliers:
        raise HomographyError(f"Only {n_inliers} inliers, need {min_inliers}")
    return normalize_homography(g_current), n_inliers
