"""Robust homography estimation with RANSAC."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from football_vision.datatypes import Homography, Points2D
from football_vision.errors import HomographyError
from football_vision.homography.dlt import estimate_homography_dlt, reprojection_error


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
        HomographyError: If fewer than four correspondences are given, the two
            point sets differ in length, or no non-degenerate model is found.
    """
    src = np.asarray(src_points, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_points, dtype=np.float64).reshape(-1, 2)
    count = len(src)  # number of correspondences N
    if count != len(dst) or count < 4:
        raise HomographyError(
            f"RANSAC needs at least 4 matching points, got {count} and {len(dst)}"
        )
    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(count, dtype=bool)  # inlier mask of the best model
    best_homography: Homography | None = None
    budget = max_iterations  # shrinks as the inlier ratio of the best model grows
    iteration = 0
    while iteration < budget:
        iteration += 1
        sample = rng.choice(count, size=4, replace=False)
        # Skip degenerate samples: any three of the four source or target points
        # being (nearly) collinear leaves the homography undetermined.
        degenerate = False
        for points in (src[sample], dst[sample]):
            for skip in range(4):
                a, b, c = np.delete(points, skip, axis=0)
                area = abs(
                    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                )
                scale = np.linalg.norm(b - a) * np.linalg.norm(c - a)
                degenerate = degenerate or area <= 1e-6 * max(scale, 1e-12)
        if degenerate:
            continue
        try:
            candidate = estimate_homography_dlt(src[sample], dst[sample])
        except (HomographyError, np.linalg.LinAlgError):
            continue
        error = reprojection_error(candidate, src, dst)
        inliers = error < threshold  # NaN (point at infinity) compares as False
        if inliers.sum() > best_inliers.sum() or best_homography is None:
            best_inliers, best_homography = inliers, candidate
            outlier_ratio = 1.0 - max(int(inliers.sum()), 4) / count
            budget = min(
                max_iterations,
                ransac_num_iterations(confidence, min(outlier_ratio, 0.999)),
            )
    if best_homography is None or best_inliers.sum() < 4:
        raise HomographyError("RANSAC found no valid homography")
    # Final least-squares fits on all inliers; keep a refit only if it does not
    # lose inliers.
    for _ in range(3):
        try:
            refit = estimate_homography_dlt(src[best_inliers], dst[best_inliers])
        except (HomographyError, np.linalg.LinAlgError):
            break
        refit_inliers = reprojection_error(refit, src, dst) < threshold
        if refit_inliers.sum() < best_inliers.sum():
            break
        done = np.array_equal(refit_inliers, best_inliers)
        best_homography, best_inliers = refit, refit_inliers
        if done:
            break
    return best_homography, best_inliers
