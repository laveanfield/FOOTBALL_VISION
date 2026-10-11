"""Per-frame preprocessing: resizing, color spaces, denoising and lighting.

All functions take and return BGR ``uint8`` images unless stated otherwise, and
never modify their input.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from football_vision.config import PreprocessConfig
from football_vision.datatypes import Frame

ColorSpace = Literal["rgb", "gray", "hsv", "lab"]


def resize_frame(frame: Frame, size: tuple[int, int]) -> Frame:
    """Resizes a frame, using area interpolation when shrinking.

    Args:
        frame: BGR image of shape ``(H, W, 3)``.
        size: Target ``(width, height)`` in pixels.

    Returns:
        The resized image; the input itself is returned if the size already matches.
    """
    width, height = size
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    shrinking = width * height < frame.shape[1] * frame.shape[0]
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def convert_color(frame: Frame, space: ColorSpace) -> np.ndarray:
    """Converts a BGR frame to another color space.

    OpenCV decodes video as BGR, so ``"rgb"`` is needed before handing images to
    libraries that expect RGB (Matplotlib, PyTorch models).

    Args:
        frame: BGR image of shape ``(H, W, 3)``.
        space: Target color space: ``"rgb"``, ``"gray"``, ``"hsv"`` or ``"lab"``.

    Returns:
        The converted image. ``"gray"`` has shape ``(H, W)``.

    Raises:
        ValueError: If ``space`` is not supported.
    """
    codes = {
        "rgb": cv2.COLOR_BGR2RGB,
        "gray": cv2.COLOR_BGR2GRAY,
        "hsv": cv2.COLOR_BGR2HSV,
        "lab": cv2.COLOR_BGR2LAB,
    }
    if space not in codes:
        raise ValueError(
            f"Unsupported color space '{space}', choose from {sorted(codes)}"
        )
    return cv2.cvtColor(frame, codes[space])


def denoise_frame(
    frame: Frame, method: Literal["none", "gaussian", "bilateral"] = "gaussian"
) -> Frame:
    """Reduces image noise before gradient-based processing.

    Args:
        frame: BGR image.
        method: ``"gaussian"`` (fast), ``"bilateral"`` (edge preserving, slower)
            or ``"none"``.

    Returns:
        The filtered image.

    Raises:
        ValueError: If ``method`` is not supported.
    """
    if method == "none":
        return frame
    if method == "gaussian":
        return cv2.GaussianBlur(frame, (5, 5), 0)
    if method == "bilateral":
        return cv2.bilateralFilter(frame, d=7, sigmaColor=50, sigmaSpace=50)
    raise ValueError(f"Unsupported denoise method '{method}'")


def equalize_lighting(
    frame: Frame, clip_limit: float = 2.0, tile_grid: tuple[int, int] = (8, 8)
) -> Frame:
    """Equalizes uneven lighting with CLAHE on the lightness channel.

    Args:
        frame: BGR image.
        clip_limit: CLAHE contrast limit.
        tile_grid: Number of CLAHE tiles as ``(columns, rows)``.

    Returns:
        The BGR image with equalized lightness.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_frame(frame: Frame, cfg: PreprocessConfig) -> Frame:
    """Applies the configured denoising and lighting steps.

    Args:
        frame: BGR image that is already resized.
        cfg: Preprocessing settings.

    Returns:
        The preprocessed BGR image.
    """
    out = denoise_frame(frame, cfg.denoise)
    if cfg.use_clahe:
        out = equalize_lighting(out)
    return out
