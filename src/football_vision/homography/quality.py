"""Quality checks for homographies: drift monitoring and validity tests."""

from __future__ import annotations

from football_vision.datatypes import Homography, Mask
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

    Args:
        edge_map: Binary edge or white-line map of shape ``(H, W)``.
        homography: Matrix mapping image to pitch.
        template: Pitch model providing the lines.
        tolerance_px: Distance below which a sample counts as matched.
        samples_per_meter: Sampling density along the lines.

    Returns:
        A score in ``[0, 1]``; higher is better.

    Raises:
        NotImplementedError: Stub; to be implemented by member B.
    """
    # TODO(B): cv2.distanceTransform on the inverted edge map; ignore samples that
    #   project outside the image.
    raise NotImplementedError("TODO(B): implement line_overlay_score")


def check_homography_validity(
    homography: Homography,
    frame_size: tuple[int, int],
    template: PitchTemplate,
) -> bool:
    """Checks that a homography is geometrically plausible.

    Typical tests: finite entries, sensible determinant sign and condition number,
    and the four image corners projecting to a quadrilateral that is convex, has a
    plausible area and lies near the pitch rectangle.

    Args:
        homography: Matrix mapping image to pitch.
        frame_size: ``(width, height)`` of the image.
        template: Pitch model providing the expected extent.

    Returns:
        ``True`` if the homography passes all tests.

    Raises:
        NotImplementedError: Stub; to be implemented by member B.
    """
    # TODO(B): choose thresholds from the dev clip, then store them in HomographyConfig.
    raise NotImplementedError("TODO(B): implement check_homography_validity")
