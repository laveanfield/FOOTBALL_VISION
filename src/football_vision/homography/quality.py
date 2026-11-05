"""Quality checks for homographies: drift monitoring and validity tests."""

from __future__ import annotations

import numpy as np

from football_vision.datatypes import Homography, Mask
from football_vision.errors import HomographyError
from football_vision.homography.geometry import apply_homography, invert_homography
from football_vision.pitch.template import PitchTemplate


def line_overlay_score(
    edge_map: Mask,
    homography: Homography,
    template: PitchTemplate,
    tolerance_px: float = 3.0,
    samples_per_meter: float = 2.0,
) -> float:
    """Scores how well the projected pitch template matches the image edges.

    Sample points along every template line, project them into the image with the
    inverse homography, and look up the distance to the nearest edge pixel in the
    distance transform of the edge map. The score is the fraction of samples whose
    distance is below ``tolerance_px``. It needs no ground truth, so it detects drift
    and triggers a re-anchor when it falls below a threshold.

    Instead of a full distance transform, each sample checks whether any edge pixel
    lies within a disc of radius ``tolerance_px`` around its rounded position; this
    gives the same fraction up to a sub-pixel rounding of the sample position.

    Args:
        edge_map: Binary edge or white-line map of shape ``(H, W)``.
        homography: Matrix mapping image to pitch.
        template: Pitch model providing the lines.
        tolerance_px: Distance below which a sample counts as matched.
        samples_per_meter: Sampling density along the lines.

    Returns:
        A score in ``[0, 1]``; higher is better. It is ``0.0`` when the matrix is
        singular or no sample projects inside the image.
    """
    # TODO(B): cv2.distanceTransform on the inverted edge map; ignore samples that
    #   project outside the image.
    edges = np.asarray(edge_map) > 0  # (H, W) bool, True on edge pixels
    height, width = edges.shape[:2]
    try:
        h_pitch_to_image = invert_homography(homography)
    except HomographyError:
        return 0.0
    # Sample every marking at ``samples_per_meter`` along its arc length (meters).
    samples = []
    for line in template.polylines():
        steps = np.linalg.norm(np.diff(line, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(steps)])  # arc length at each vertex
        count = max(2, int(np.ceil(arc[-1] * samples_per_meter)))
        positions = np.linspace(0.0, arc[-1], count)
        samples.append(
            np.stack(
                [
                    np.interp(positions, arc, line[:, 0]),
                    np.interp(positions, arc, line[:, 1]),
                ],
                axis=1,
            )
        )
    pixels = apply_homography(h_pitch_to_image, np.concatenate(samples, axis=0))
    pixels = pixels[np.isfinite(pixels).all(axis=1)]
    cols = np.rint(pixels[:, 0]).astype(np.int64)  # sample column (x) in pixels
    rows = np.rint(pixels[:, 1]).astype(np.int64)  # sample row (y) in pixels
    inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    if not inside.any():
        return 0.0
    cols, rows = cols[inside], rows[inside]
    # Offsets of all pixels within ``tolerance_px`` of a sample (a disc).
    radius = int(np.floor(tolerance_px))
    d_row, d_col = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    disc = d_row**2 + d_col**2 <= tolerance_px**2
    padded = np.pad(edges, radius, constant_values=False)
    matched = np.zeros(len(cols), dtype=bool)  # True if an edge is within tolerance
    for shift_row, shift_col in zip(d_row[disc], d_col[disc], strict=True):
        matched |= padded[rows + radius + shift_row, cols + radius + shift_col]
    return float(matched.mean())


def check_homography_validity(
    homography: Homography,
    frame_size: tuple[int, int],
    template: PitchTemplate,
) -> bool:
    """Checks that a homography is geometrically plausible.

    Typical tests: finite entries, sensible determinant sign and condition number,
    and the four image corners projecting to a quadrilateral that is convex and keeps
    its orientation (no mirroring), has a plausible area and lies near the pitch
    rectangle.

    The quadrilateral is taken from the lower half of the image (its four
    corners), because the upper half of a broadcast frame often shows sky or
    stands above the horizon, which has no counterpart on the pitch plane. The
    thresholds are starting values (condition number below ``1e7``, area between
    30 m^2 and three pitch areas, centroid at most 20 m outside the pitch).

    Args:
        homography: Matrix mapping image to pitch.
        frame_size: ``(width, height)`` of the image.
        template: Pitch model providing the expected extent.

    Returns:
        ``True`` if the homography passes all tests.
    """
    # TODO(B): choose thresholds from the dev clip, then store them in HomographyConfig.
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return False
    if abs(matrix[2, 2]) < 1e-12 or abs(np.linalg.det(matrix)) < 1e-18:
        return False
    if np.linalg.cond(matrix / matrix[2, 2]) > 1e7:
        return False
    width, height = frame_size
    corners = np.array(
        [[0.0, height / 2], [width, height / 2], [width, height], [0.0, height]]
    )
    # Homogeneous scale of each corner: all positive (or all negative) means the
    # quadrilateral does not cross the horizon of the pitch plane.
    scale = np.hstack([corners, np.ones((4, 1))]) @ matrix[2]
    if not (np.all(scale > 1e-12) or np.all(scale < -1e-12)):
        return False
    quad = apply_homography(matrix, corners)  # the corners in pitch meters
    if not np.isfinite(quad).all():
        return False
    edges = np.roll(quad, -1, axis=0) - quad
    turns = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(
        edges[:, 0], -1
    )
    # Convex and not mirrored: a real camera never reverses the orientation of the
    # pitch, so the corners must keep their clockwise order (image y points down,
    # and so does pitch Y).
    if not np.all(turns > 0):
        return False
    area = 0.5 * abs(
        np.sum(
            quad[:, 0] * np.roll(quad[:, 1], -1) - np.roll(quad[:, 0], -1) * quad[:, 1]
        )
    )
    if not 30.0 <= area <= 3.0 * template.length * template.width:
        return False
    centroid = quad.mean(axis=0)
    margin = 20.0
    return bool(
        -margin <= centroid[0] <= template.length + margin
        and -margin <= centroid[1] <= template.width + margin
    )
