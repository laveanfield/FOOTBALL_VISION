"""Edge and white-line extraction (stage P2)."""

from __future__ import annotations

import numpy as np

from football_vision.config import PitchConfig
from football_vision.datatypes import Frame, Mask
from football_vision.preprocessing.frames import convert_color


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
        mask: Optional pitch mask; edges outside it are removed.
        low: Lower hysteresis threshold.
        high: Upper hysteresis threshold.
        blur_sigma: Standard deviation of the Gaussian smoothing.

    Returns:
        ``uint8`` edge map of shape ``(H, W)`` with values 0 or 255.

    Raises:
        NotImplementedError: Stub; to be implemented by member A.
    """
    # TODO(A): cv2.GaussianBlur -> cv2.Canny -> bitwise_and with the eroded mask
    #   (erode so the grass/stand boundary is not reported as a pitch line).
    raise NotImplementedError("TODO(A): implement detect_edges_canny")


def extract_white_lines_tophat(
    gray: np.ndarray, mask: Mask | None, kernel_size: int
) -> Mask:
    """Extracts thin bright structures (pitch lines) with a white top-hat transform.

    The white top-hat is ``image - opening(image)``. With a structuring element
    wider than a line, it keeps bright structures narrower than the element and
    suppresses mowing stripes, which are wide. Compare it against Canny (ablation A1).

    Args:
        gray: Grayscale image of shape ``(H, W)``.
        mask: Optional pitch mask restricting the result.
        kernel_size: Size of the structuring element in pixels.

    Returns:
        ``uint8`` binary map of line pixels with values 0 or 255.

    Raises:
        NotImplementedError: Stub; to be implemented by member A.
    """
    # TODO(A): cv2.morphologyEx(MORPH_TOPHAT) -> Otsu or adaptive threshold -> mask.
    raise NotImplementedError("TODO(A): implement extract_white_lines_tophat")


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
