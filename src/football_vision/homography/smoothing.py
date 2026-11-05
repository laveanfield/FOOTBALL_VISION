"""Temporal smoothing of homography sequences."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from football_vision.datatypes import Points2D
from football_vision.errors import HomographyError
from football_vision.homography.dlt import estimate_homography_dlt
from football_vision.homography.geometry import apply_homography


def smooth_homography_sequence(
    matrices: npt.NDArray[np.float64],
    reference_points_pitch: Points2D,
    window: int,
    polyorder: int = 2,
) -> npt.NDArray[np.float64]:
    """Reduces frame-to-frame jitter of a homography sequence.

    Averaging the nine matrix entries directly is not geometrically meaningful.
    Instead, project four reference points (for example the corners of a pitch
    rectangle) into the image with ``H_t^-1``, smooth each point's trajectory
    (Savitzky-Golay or a moving median), and rebuild every ``H_t`` from the four
    smoothed correspondences (``cv2.getPerspectiveTransform``). Smooth within a shot
    only, never across cuts.

    The Savitzky-Golay filter is evaluated as a local polynomial fit; near both
    ends of the sequence the polynomial of the first or last window is used, so the
    output has the same length as the input. A sequence shorter than ``window`` uses
    the largest odd window that fits.

    Args:
        matrices: Array of shape ``(T, 3, 3)`` mapping image to pitch.
        reference_points_pitch: Array of shape ``(4, 2)`` in pitch meters.
        window: Odd smoothing window in frames.
        polyorder: Polynomial order of the Savitzky-Golay filter.

    Returns:
        Array of shape ``(T, 3, 3)`` with the smoothed homographies. A frame whose
        smoothed correspondences are degenerate keeps its original matrix.
    """
    # TODO(B): see ablation A3 (no smoothing vs. matrix entries vs. reference points).
    mats = np.asarray(matrices, dtype=np.float64)
    reference = np.asarray(reference_points_pitch, dtype=np.float64).reshape(-1, 2)
    length = len(mats)  # number of frames T
    size = min(window, length if length % 2 else length - 1)  # odd, fits in T
    if size < 3 or window < 3:
        return mats.copy()
    order = min(polyorder, size - 1)  # polynomial order must be below the window
    # Trajectories of the reference points in the image, shape (T, 4, 2).
    trajectory = np.stack([apply_homography(np.linalg.inv(m), reference) for m in mats])
    half = size // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64)  # frame offsets in window
    design = np.vander(offsets, order + 1, increasing=True)  # (size, order + 1)
    # Row 0 of the pseudo-inverse gives the fitted value at the window center.
    center_weights = np.linalg.pinv(design)[0]
    flat = trajectory.reshape(length, -1)  # one column per point coordinate
    smooth = flat.copy()
    # Interior frames: centered convolution with the Savitzky-Golay weights.
    for t in range(half, length - half):
        smooth[t] = center_weights @ flat[t - half : t + half + 1]
    # Edges: evaluate the polynomial fitted on the first / last window.
    for start, targets in (
        (0, range(half)),
        (length - size, range(length - half, length)),
    ):
        coefficients = np.linalg.pinv(design) @ flat[start : start + size]
        for t in targets:
            smooth[t] = (
                np.vander([t - start - half], order + 1, increasing=True) @ coefficients
            )
    smooth = smooth.reshape(length, -1, 2)
    result = mats.copy()
    for t in range(length):
        try:
            result[t] = estimate_homography_dlt(smooth[t], reference)
        except (HomographyError, np.linalg.LinAlgError):
            pass
    return result
