
"""Image-processing primitives shared by the football vision pipeline.

This module contains small OpenCV/NumPy helpers used by preprocessing stages.
Images are represented as NumPy arrays; color images use OpenCV's BGR convention.
"""

from __future__ import annotations

import cv2
import numpy as np


def bgr_to_hsv(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR image to OpenCV HSV color space.

    Args:
        frame: Input image with shape ``(H, W, 3)`` in BGR channel order.

    Returns:
        HSV image with the same height and width and ``uint8`` channels.

    Raises:
        ValueError: If ``frame`` is not a three-channel image.
    """
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected a BGR image with shape (H, W, 3), got {image.shape}"
        )
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def _odd_kernel_size(kernel_size: int) -> int:
    """Normalize a morphology kernel size to a positive odd integer."""
    size = max(1, int(kernel_size))
    return size if size % 2 == 1 else size + 1


def opening(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply morphological opening to a binary or grayscale image.

    Opening removes small foreground specks while preserving larger regions.

    Args:
        mask: Two-dimensional input mask.
        kernel_size: Width and height of the square structuring element.

    Returns:
        The opened image as a ``uint8`` array.

    Raises:
        ValueError: If ``mask`` is not a two-dimensional array.
    """
    image = np.asarray(mask)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {image.shape}")
    size = _odd_kernel_size(kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(image.astype(np.uint8), cv2.MORPH_OPEN, kernel)


def closing(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply morphological closing to a binary or grayscale image.

    Closing bridges small gaps and narrow breaks in foreground regions.

    Args:
        mask: Two-dimensional input mask.
        kernel_size: Width and height of the square structuring element.

    Returns:
        The closed image as a ``uint8`` array.

    Raises:
        ValueError: If ``mask`` is not a two-dimensional array.
    """
    image = np.asarray(mask)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {image.shape}")
    size = _odd_kernel_size(kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(image.astype(np.uint8), cv2.MORPH_CLOSE, kernel)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected foreground component.

    All nonzero pixels are treated as foreground. If no foreground pixels are
    present, an all-zero mask is returned.

    Args:
        mask: Two-dimensional binary mask.

    Returns:
        A ``uint8`` mask containing the largest component with values 0 or 255.

    Raises:
        ValueError: If ``mask`` is not a two-dimensional array.
    """
    image = np.asarray(mask)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {image.shape}")
    binary = (image != 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if count <= 1:
        return np.zeros(binary.shape, dtype=np.uint8)
    # Label zero is the background, so search labels 1..count-1.
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes in a binary foreground mask.

    The mask is padded with a background border, then the background connected
    to the image exterior is flood-filled. Any remaining zero-valued regions
    are enclosed holes and are filled.

    Args:
        mask: Two-dimensional binary mask; nonzero pixels are foreground.

    Returns:
        A ``uint8`` mask with values 0 or 255 and enclosed holes filled.

    Raises:
        ValueError: If ``mask`` is not a two-dimensional array.
    """
    image = np.asarray(mask)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {image.shape}")
    binary = np.where(image != 0, 255, 0).astype(np.uint8)

    # Padding ensures the flood-fill seed is always outside the foreground.
    padded = cv2.copyMakeBorder(
        binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
    )
    flood_filled = padded.copy()
    flood_mask = np.zeros(
        (flood_filled.shape[0] + 2, flood_filled.shape[1] + 2),
        dtype=np.uint8,
    )
    cv2.floodFill(flood_filled, flood_mask, (0, 0), 255)

    exterior = flood_filled[1:-1, 1:-1]
    holes = cv2.bitwise_not(exterior)
    return cv2.bitwise_or(binary, holes)
