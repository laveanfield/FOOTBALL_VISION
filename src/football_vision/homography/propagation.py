"""Frame-to-frame alignment and homography propagation for moving cameras."""

from __future__ import annotations

from typing import Literal

import numpy as np

from football_vision.datatypes import Frame, Homography
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
        NotImplementedError: Stub; to be implemented by member B.
    """
    # TODO(B): cv2.SIFT_create / cv2.ORB_create / cv2.AKAZE_create; BFMatcher with kNN;
    #   ratio test; cv2.findHomography(RANSAC). Optionally refine with
    #   cv2.findTransformECC.
    raise NotImplementedError("TODO(B): implement estimate_frame_alignment")
