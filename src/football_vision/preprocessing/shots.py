"""Shot boundary detection for broadcast video (stage P1).

Broadcast footage contains cuts, replays and close-ups. The homography is only
estimated inside "wide" shots, so the video is first split into shots.
"""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from football_vision.datatypes import Frame

_EPS = 1e-10
# Two cuts closer than this many processed frames are treated as a flash (the
# image jumps to white and back) and both are ignored.
_FLASH_MAX_GAP_SAMPLES = 5
_DEFAULT_BINS = (8, 8, 8)


def _normalized_hsv_histogram(frame: Frame, bins: tuple[int, int, int]) -> np.ndarray:
    """Builds a flattened 3D HSV histogram that sums to one."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256]
    ).astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist.ravel()


def _chi_square(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Computes ``0.5 * sum((a - b)^2 / (a + b + eps))`` for two histograms."""
    return float(0.5 * np.sum((hist_a - hist_b) ** 2 / (hist_a + hist_b + _EPS)))


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
    """
    # TODO(A): convert both frames to HSV, build 3D histograms with cv2.calcHist,
    #   normalize, and evaluate the chi-square formula above.
    return _chi_square(
        _normalized_hsv_histogram(frame_a, bins),
        _normalized_hsv_histogram(frame_b, bins),
    )


def _drop_flashes(
    cuts: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    """Removes pairs of cuts that are closer than ``_FLASH_MAX_GAP_SAMPLES``.

    A camera flash makes the image jump away and back, so it produces two cuts in
    quick succession, and the content before and after belongs to the same shot.
    Cuts are ``(last_id_of_old_shot, first_id_of_new_shot, sample_index)``.
    """
    kept: list[tuple[int, int, int]] = []
    i = 0
    while i < len(cuts):
        if i + 1 < len(cuts) and cuts[i + 1][2] - cuts[i][2] < _FLASH_MAX_GAP_SAMPLES:
            i += 2
            continue
        kept.append(cuts[i])
        i += 1
    return kept


def detect_shot_boundaries(
    frames: Iterable[tuple[int, Frame]], threshold: float
) -> list[tuple[int, int]]:
    """Splits a frame sequence into shots.

    A cut is declared where the histogram distance between consecutive frames
    exceeds ``threshold``. Consecutive cuts closer than a few frames can be merged
    to ignore flashes: here, two cuts less than five processed frames apart are both
    discarded.

    Only the histogram of the previous frame is kept, so the iterable can be a
    generator over a long video.

    Args:
        frames: Iterable of ``(frame_id, frame)`` pairs in temporal order.
        threshold: Distance above which a cut is declared.

    Returns:
        Inclusive ``(first_frame_id, last_frame_id)`` ranges, one per shot.
        The list is empty when ``frames`` is empty.
    """
    # TODO(A): iterate over consecutive frame pairs, call histogram_distance, and
    #   collect shot ranges. Plot the distance curve to choose the threshold.
    first_id: int | None = None
    previous_id = 0
    previous_hist: np.ndarray | None = None
    cuts: list[tuple[int, int, int]] = []
    for sample_index, (frame_id, frame) in enumerate(frames):
        hist = _normalized_hsv_histogram(frame, _DEFAULT_BINS)
        if previous_hist is None:
            first_id = frame_id
        elif _chi_square(previous_hist, hist) > threshold:
            cuts.append((previous_id, frame_id, sample_index))
        previous_hist, previous_id = hist, frame_id
    if first_id is None:
        return []
    shots: list[tuple[int, int]] = []
    start = first_id
    for last_of_shot, first_of_next, _ in _drop_flashes(cuts):
        shots.append((int(start), int(last_of_shot)))
        start = first_of_next
    shots.append((int(start), int(previous_id)))
    return shots
