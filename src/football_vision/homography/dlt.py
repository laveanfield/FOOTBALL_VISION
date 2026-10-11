"""Direct Linear Transform (DLT) estimation of a homography.

Derivation (see Appendix A4 of the project plan): ``x' ~ H x`` is equivalent to
``x' x (H x) = 0``. Each correspondence yields two independent linear equations in
the nine entries of ``H``, so four non-degenerate correspondences determine ``H``.
With more points, the solution of ``min ||A h||`` subject to ``||h|| = 1`` is the
right singular vector of ``A`` with the smallest singular value.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from football_vision.datatypes import Homography, Points2D
from football_vision.homography.geometry import apply_homography


def hartley_normalization(points: Points2D) -> tuple[Points2D, npt.NDArray[np.float64]]:
    """Normalizes points for a well-conditioned DLT.

    The transform translates the centroid to the origin and scales the points so
    that their mean distance from the origin is ``sqrt(2)``.

    Args:
        points: Array of shape ``(N, 2)``.

    Returns:
        Tuple ``(normalized_points, transform)`` where ``transform`` is the
        ``(3, 3)`` similarity matrix that produced the normalized points.

    Raises:
        NotImplementedError: Stub; to be implemented by member B.
    """
    # TODO(B): centroid -> mean distance -> scale = sqrt(2) / mean_distance;
    #   build T = [[s, 0, -s*cx], [0, s, -s*cy], [0, 0, 1]].
    raise NotImplementedError("TODO(B): implement hartley_normalization")


def estimate_homography_dlt(
    src_points: Points2D, dst_points: Points2D, normalize: bool = True
) -> Homography:
    """Estimates the homography mapping ``src_points`` to ``dst_points`` with DLT.

    Steps: optionally normalize both point sets, stack the ``2N x 9`` matrix ``A``
    built from the two independent rows of ``x' x (H x) = 0``, take the right
    singular vector of the smallest singular value, reshape it to ``3 x 3``,
    undo the normalization (``H = T_dst^-1 @ H_hat @ T_src``) and scale so that
    ``H[2, 2] = 1``.

    Args:
        src_points: Array of shape ``(N, 2)`` with ``N >= 4`` source points.
        dst_points: Array of shape ``(N, 2)`` with the matching target points.
        normalize: Whether to apply Hartley normalization first.

    Returns:
        The ``(3, 3)`` homography with ``H[2, 2] = 1``.

    Raises:
        NotImplementedError: Stub; to be implemented by member B.
    """
    # TODO(B): validate N >= 4 and equal lengths (raise ValueError / HomographyError),
    #   build A, run np.linalg.svd, reshape the last row of Vt. Compare with
    #   cv2.findHomography (see tests/test_dlt_contract.py).
    raise NotImplementedError("TODO(B): implement estimate_homography_dlt")


def reprojection_error(
    homography: Homography, src_points: Points2D, dst_points: Points2D
) -> npt.NDArray[np.float64]:
    """Measures how far mapped source points land from their targets.

    Args:
        homography: Matrix mapping source to target coordinates.
        src_points: Array of shape ``(N, 2)``.
        dst_points: Array of shape ``(N, 2)`` with the expected targets.

    Returns:
        Array of shape ``(N,)`` with Euclidean distances in the units of
        ``dst_points`` (meters when mapping image to pitch).
    """
    mapped = apply_homography(homography, src_points)
    return np.linalg.norm(mapped - np.asarray(dst_points, dtype=np.float64), axis=1)
