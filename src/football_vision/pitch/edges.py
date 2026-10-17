"""Edge and white-line extraction (stage P2)."""

from __future__ import annotations

import cv2
import numpy as np

from football_vision.config import PitchConfig
from football_vision.datatypes import Frame, Mask
from football_vision.preprocessing.frames import convert_color

# The mask is eroded by this many pixels so that the grass/stand boundary itself
# is not reported as a pitch line.
_MASK_ERODE_PX = 5
# Floor for the top-hat threshold; Otsu alone would turn noise into lines in a
# frame without any line.
_MIN_TOPHAT_LEVEL = 20


def _restrict_to_mask(binary: Mask, mask: Mask | None) -> Mask:
    """Keeps only the pixels of ``binary`` inside the eroded pitch mask."""
    if mask is None:
        return binary
    if mask.shape != binary.shape:
        raise ValueError(
            f"mask shape {mask.shape} does not match image shape {binary.shape}"
        )
    kernel = np.ones((_MASK_ERODE_PX, _MASK_ERODE_PX), dtype=np.uint8)
    return np.asarray(cv2.bitwise_and(binary, cv2.erode(mask, kernel)), dtype=np.uint8)


def detect_edges_canny(
    gray: np.ndarray,
    mask: Mask | None,
    low: int,
    high: int,
    blur_sigma: float,
) -> Mask:
    """Detects edges with the Canny operator inside the pitch mask.

    Pipeline: Gaussian smoothing, gradient magnitude and direction, non-maximum
    suppression, then double thresholding with hysteresis. Use a high-to-low
    threshold ratio between 2:1 and 3:1.

    Args:
        gray: Grayscale image of shape ``(H, W)``.
        mask: Optional pitch mask; edges outside it are removed. A mask whose
            shape differs from ``gray`` makes the call fail with ``ValueError``.
        low: Lower hysteresis threshold.
        high: Upper hysteresis threshold.
        blur_sigma: Standard deviation of the Gaussian smoothing.

    Returns:
        ``uint8`` edge map of shape ``(H, W)`` with values 0 or 255.
    """
    # TODO(A): cv2.GaussianBlur -> cv2.Canny -> bitwise_and with the eroded mask
    #   (erode so the grass/stand boundary is not reported as a pitch line).
    smoothed = cv2.GaussianBlur(gray, (0, 0), blur_sigma) if blur_sigma > 0 else gray
    edges = np.asarray(cv2.Canny(smoothed, low, high), dtype=np.uint8)
    return _restrict_to_mask(edges, mask)


def extract_white_lines_tophat(
    gray: np.ndarray, mask: Mask | None, kernel_size: int
) -> Mask:
    """Extracts thin bright structures (pitch lines) with a white top-hat transform.

    The white top-hat is ``image - opening(image)``. With a structuring element
    wider than a line, it keeps bright structures narrower than the element and
    suppresses mowing stripes, which are wide. Compare it against Canny (ablation A1).

    The top-hat response is thresholded with Otsu's method (computed only over the
    pitch mask when one is given), with a minimum level of 20 so that a frame
    without lines gives an empty result. The result is a map of line pixels with the
    thickness of the lines, not a one-pixel edge.

    Args:
        gray: Grayscale image of shape ``(H, W)``.
        mask: Optional pitch mask restricting the result. A mask whose shape
            differs from ``gray`` makes the call fail with ``ValueError``.
        kernel_size: Size of the structuring element in pixels.

    Returns:
        ``uint8`` binary map of line pixels with values 0 or 255.
    """
    # TODO(A): cv2.morphologyEx(MORPH_TOPHAT) -> Otsu or adaptive threshold -> mask.
    size = max(3, int(kernel_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    sample = tophat if mask is None else tophat[mask > 0]
    if sample.size == 0:
        return np.zeros(gray.shape, dtype=np.uint8)
    otsu, _ = cv2.threshold(
        sample.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    level = max(float(otsu), float(_MIN_TOPHAT_LEVEL))
    binary = np.where(tophat > level, 255, 0).astype(np.uint8)
    return _restrict_to_mask(binary, mask)


def extract_edges(frame: Frame, mask: Mask | None, cfg: PitchConfig) -> Mask:
    """Extracts a line/edge map with the method chosen in the configuration.

    Args:
        frame: BGR frame.
        mask: Optional pitch mask.
        cfg: Pitch settings; ``cfg.edge_method`` selects Canny or top-hat.

    Returns:
        ``uint8`` edge map with values 0 or 255.

    Raises:
        ValueError: If ``cfg.edge_method`` is not supported.
    """
    gray = convert_color(frame, "gray")
    if cfg.edge_method == "canny":
        return detect_edges_canny(
            gray, mask, cfg.canny_low, cfg.canny_high, cfg.blur_sigma
        )
    if cfg.edge_method == "tophat":
        return extract_white_lines_tophat(gray, mask, cfg.tophat_kernel)
    raise ValueError(f"Unsupported edge method '{cfg.edge_method}'")
