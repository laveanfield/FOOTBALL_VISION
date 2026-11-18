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
    return _chi_square(
        _normalized_hsv_histogram(frame_a, bins),
        _normalized_hsv_histogram(frame_b, bins),
    )


def _drop_flashes(
    cuts: list[tuple[int, int, int, np.ndarray, np.ndarray]],
    threshold: float,
) -> list[tuple[int, int]]:
    """Removes pairs of cuts closer than ``_FLASH_MAX_GAP_SAMPLES`` if content matches.

    A camera flash makes the image jump away and back, producing two cuts in quick
    succession where the frame before the first cut and the frame after the second
    cut belong to the same shot (their histogram distance does not exceed
    ``threshold``). A short insert between two different shots is preserved.
    """
    kept: list[tuple[int, int]] = []
    i = 0
    while i < len(cuts):
        if (
            i + 1 < len(cuts)
            and cuts[i + 1][2] - cuts[i][2] < _FLASH_MAX_GAP_SAMPLES
            and _chi_square(cuts[i][3], cuts[i + 1][4]) <= threshold
        ):
            i += 2
            continue
        kept.append((cuts[i][0], cuts[i][1]))
        i += 1
    return kept


def detect_shot_boundaries(
    frames: Iterable[tuple[int, Frame]], threshold: float
) -> list[tuple[int, int]]:
    """Splits a frame sequence into shots.

    A cut is declared where the histogram distance between consecutive frames
    exceeds ``threshold``. Two cuts less than five processed frames apart are
    discarded as a flash only when the frames before and after the interruption have
    histogram distance at most ``threshold``.

    Only the histogram of the previous frame (and cut-boundary histograms) is kept,
    so the iterable can be a generator over a long video.

    Args:
        frames: Iterable of ``(frame_id, frame)`` pairs in temporal order.
        threshold: Distance above which a cut is declared.

    Returns:
        Inclusive ``(first_frame_id, last_frame_id)`` ranges, one per shot.
        The list is empty when ``frames`` is empty.
    """
    # TODO(A): calibrate threshold by plotting the distance curve on real clips.
    first_id: int | None = None
    previous_id = 0
    previous_hist: np.ndarray | None = None
    cuts: list[tuple[int, int, int, np.ndarray, np.ndarray]] = []
    for sample_index, (frame_id, frame) in enumerate(frames):
        hist = _normalized_hsv_histogram(frame, _DEFAULT_BINS)
        if previous_hist is None:
            first_id = frame_id
        elif _chi_square(previous_hist, hist) > threshold:
            cuts.append((previous_id, frame_id, sample_index, previous_hist, hist))
        previous_hist, previous_id = hist, frame_id
    if first_id is None:
        return []
    shots: list[tuple[int, int]] = []
    start = first_id
    for last_of_shot, first_of_next in _drop_flashes(cuts, threshold):
        shots.append((int(start), int(last_of_shot)))
        start = first_of_next
    shots.append((int(start), int(previous_id)))
    return shots
