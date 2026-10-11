"""Robust homography estimation with RANSAC."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from football_vision.datatypes import Homography, Points2D


def ransac_num_iterations(
    confidence: float, outlier_ratio: float, sample_size: int = 4
) -> int:
    """Computes how many RANSAC iterations are needed.

    The probability that one random sample of ``sample_size`` points is outlier-free
    is ``(1 - outlier_ratio) ** sample_size``. To see at least one clean sample with
    probability ``confidence``, ``N = ln(1 - p) / ln(1 - (1 - eps) ** s)`` iterations
    are required. For ``p = 0.99`` and ``s = 4`` this gives 72 iterations at 50%
    outliers and 567 at 70%.

    Args:
        confidence: Desired probability of drawing at least one clean sample, in (0, 1).
        outlier_ratio: Expected fraction of outliers, in [0, 1).
        sample_size: Number of points per sample (4 for a homography).

    Returns:
        The number of iterations, rounded up (at least 1).

    Raises:
        ValueError: If ``confidence`` or ``outlier_ratio`` is out of range.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if not 0.0 <= outlier_ratio < 1.0:
        raise ValueError("outlier_ratio must be in [0, 1)")
    clean_probability = (1.0 - outlier_ratio) ** sample_size
    if clean_probability >= 1.0:
        return 1
    return max(
        1, math.ceil(math.log(1.0 - confidence) / math.log(1.0 - clean_probability))
    )


def estimate_homography_ransac(
    src_points: Points2D,
    dst_points: Points2D,
    threshold: float,
    confidence: float = 0.99,
    max_iterations: int = 2000,
    seed: int | None = None,
) -> tuple[Homography, npt.NDArray[np.bool_]]:
    """Estimates a homography robustly against outlier correspondences.

    Loop: draw four random correspondences, estimate ``H`` with DLT, count inliers
    whose reprojection error is below ``threshold``, keep the best model, optionally
    adapt the iteration count with :func:`ransac_num_iterations`, and finally
    re-estimate ``H`` on all inliers (and refine it with Levenberg-Marquardt).

    Args:
        src_points: Array of shape ``(N, 2)``.
        dst_points: Array of shape ``(N, 2)``.
        threshold: Inlier threshold in the units of ``dst_points``.
        confidence: Desired probability of drawing one clean sample.
        max_iterations: Upper bound on the number of iterations.
        seed: Seed of the sampling generator for reproducible runs.

    Returns:
        Tuple ``(homography, inlier_mask)`` where ``inlier_mask`` has shape ``(N,)``.

    Raises:
        NotImplementedError: Stub; to be implemented by member B.
    """
    # TODO(B): use homography.dlt.estimate_homography_dlt; compare with
    #   cv2.findHomography(..., cv2.RANSAC) and cv2.USAC_MAGSAC in ablation A2.
    raise NotImplementedError("TODO(B): implement estimate_homography_ransac")
