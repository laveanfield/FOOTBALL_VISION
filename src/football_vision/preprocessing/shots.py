"""Shot boundary detection for broadcast video (stage P1).

Broadcast footage contains cuts, replays and close-ups. The homography is only
estimated inside "wide" shots, so the video is first split into shots.
"""

from __future__ import annotations

from collections.abc import Iterable

from football_vision.datatypes import Frame


def histogram_distance(
    frame_a: Frame, frame_b: Frame, bins: tuple[int, int, int] = (8, 8, 8)
) -> float:
    """Computes the chi-square distance between the HSV histograms of two frames.

    With histograms ``h_a`` and ``h_b`` normalized to sum one, the distance is
    ``0.5 * sum((h_a - h_b)**2 / (h_a + h_b + eps))`` and lies in ``[0, 1]``.
    A hard cut produces a spike; slow camera motion does not.

    Args:
        frame_a: First BGR frame.
        frame_b: Second BGR frame.
        bins: Number of histogram bins for the H, S and V channels.

    Returns:
        The chi-square distance between the two histograms.

    Raises:
        NotImplementedError: Stub; to be implemented by member A.
    """
    # TODO(A): convert both frames to HSV, build 3D histograms with cv2.calcHist,
    #   normalize, and evaluate the chi-square formula above.
    raise NotImplementedError("TODO(A): implement histogram_distance")


def detect_shot_boundaries(
    frames: Iterable[tuple[int, Frame]], threshold: float
) -> list[tuple[int, int]]:
    """Splits a frame sequence into shots.

    A cut is declared where the histogram distance between consecutive frames
    exceeds ``threshold``. Consecutive cuts closer than a few frames can be merged
    to ignore flashes.

    Args:
        frames: Iterable of ``(frame_id, frame)`` pairs in temporal order.
        threshold: Distance above which a cut is declared.

    Returns:
        Inclusive ``(first_frame_id, last_frame_id)`` ranges, one per shot.

    Raises:
        NotImplementedError: Stub; to be implemented by member A.
    """
    # TODO(A): iterate over consecutive frame pairs, call histogram_distance, and
    #   collect shot ranges. Plot the distance curve to choose the threshold.
    raise NotImplementedError("TODO(A): implement detect_shot_boundaries")
