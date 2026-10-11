"""Temporal smoothing of homography sequences."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from football_vision.datatypes import Points2D


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

    Args:
        matrices: Array of shape ``(T, 3, 3)`` mapping image to pitch.
        reference_points_pitch: Array of shape ``(4, 2)`` in pitch meters.
        window: Odd smoothing window in frames.
        polyorder: Polynomial order of the Savitzky-Golay filter.

    Returns:
        Array of shape ``(T, 3, 3)`` with the smoothed homographies.

    Raises:
        NotImplementedError: Stub; to be implemented by member B.
    """
    # TODO(B): see ablation A3 (no smoothing vs. matrix entries vs. reference points).
    raise NotImplementedError("TODO(B): implement smooth_homography_sequence")
