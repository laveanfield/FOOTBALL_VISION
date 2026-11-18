"""Homography of a keyframe, from clicked landmarks or automatically."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from football_vision.constants import GOAL_AREA_DEPTH_M, PENALTY_AREA_DEPTH_M
from football_vision.datatypes import Frame, Homography, Points2D
from football_vision.errors import HomographyError
from football_vision.homography.geometry import (
    apply_homography,
    invert_homography,
    normalize_homography,
)
from football_vision.homography.quality import (
    check_homography_validity,
    line_overlay_score,
)
from football_vision.pitch.hough import cluster_line_families
from football_vision.pitch.template import PitchTemplate

MIN_CORRESPONDENCES = 4


def homography_from_landmarks(
    image_points: Points2D,
    pitch_points: Points2D,
    ransac_threshold_px: float = 3.0,
) -> tuple[Homography, int]:
    """Estimates the image-to-pitch homography from landmark correspondences.

    This is the OpenCV baseline (semi-automatic mode): ``cv2.findHomography`` with
    RANSAC. The model is fitted from pitch to image so that the RANSAC threshold is
    measured in pixels, then inverted. It is the reference to compare against the
    team's own DLT (``homography.dlt``).

    Args:
        image_points: Array of shape ``(N, 2)`` with landmark pixels, ``N >= 4``.
        pitch_points: Array of shape ``(N, 2)`` with the matching pitch meters.
        ransac_threshold_px: Inlier threshold in pixels.

    Returns:
        Tuple ``(homography, n_inliers)`` where the matrix maps image to pitch.

    Raises:
        HomographyError: If fewer than four correspondences are given or OpenCV
            cannot fit a model.
    """
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    pit = np.asarray(pitch_points, dtype=np.float64).reshape(-1, 2)
    if len(img) != len(pit) or len(img) < MIN_CORRESPONDENCES:
        raise HomographyError(
            f"Need at least {MIN_CORRESPONDENCES} matching landmarks, got {len(img)}"
        )
    h_pitch_to_image, mask = cv2.findHomography(
        pit, img, cv2.RANSAC, ransac_threshold_px
    )
    if h_pitch_to_image is None:
        raise HomographyError("cv2.findHomography failed; check the landmarks")
    n_inliers = int(mask.sum()) if mask is not None else len(img)
    return invert_homography(h_pitch_to_image), n_inliers


def homography_from_landmark_dict(
    landmarks_image: Mapping[str, Sequence[float]],
    template: PitchTemplate | None = None,
    ransac_threshold_px: float = 3.0,
) -> tuple[Homography, int]:
    """Estimates the homography from named landmarks clicked on one frame.

    Args:
        landmarks_image: Mapping from landmark name (see ``PitchTemplate.landmarks``)
            to the clicked ``(x, y)`` pixel. Unknown names are ignored.
        template: Pitch model; defaults to the 105 x 68 m template.
        ransac_threshold_px: Inlier threshold in pixels.

    Returns:
        Tuple ``(homography, n_inliers)`` where the matrix maps image to pitch.

    Raises:
        HomographyError: If fewer than four known landmarks are given.
    """
    model = template or PitchTemplate()
    known = model.landmarks()
    names = [name for name in landmarks_image if name in known]
    if len(names) < MIN_CORRESPONDENCES:
        raise HomographyError(
            f"Need at least {MIN_CORRESPONDENCES} known landmarks, got {len(names)}"
        )
    image_points = np.array([landmarks_image[name] for name in names], dtype=np.float64)
    return homography_from_landmarks(
        image_points, model.landmark_array(names), ransac_threshold_px
    )


def estimate_keyframe_homography_auto(
    frame: Frame,
    segments: np.ndarray,
    template: PitchTemplate,
) -> Homography:
    """Estimates a keyframe homography without any clicks (hypothesize-and-verify).

    Choose two lines of each family from the detected segments, pair them with two
    lines of each family of the template, compute ``H`` from the four resulting
    intersections, and score each hypothesis by how well the remaining projected
    template lines match the edge map (see ``homography.quality.line_overlay_score``).
    Keep the best-scoring hypothesis.

    The edge map is rasterized from ``segments`` themselves. Hypotheses are first
    solved and filtered in vectorized batches (plausible projection of the lower
    image half), ranked by a coarse overlay score on the straight template lines,
    and the best 100 are re-ranked with ``line_overlay_score`` and
    ``check_homography_validity``. At most the 5 longest segments of each family
    are used.

    The pitch markings are identical after a 180-degree rotation, so the lines
    alone cannot tell the two halves apart: the result may be the true homography
    rotated by 180 degrees (``X -> 105 - X``, ``Y -> 68 - Y``). Mirrored solutions
    are excluded.

    Args:
        frame: BGR keyframe.
        segments: Merged line segments ``(N, 4)`` of the frame.
        template: Pitch model providing the template lines.

    Returns:
        The ``(3, 3)`` homography mapping image to pitch coordinates.

    Raises:
        HomographyError: If a family has fewer than two usable lines or no
            plausible hypothesis is found.
    """
    height, width = frame.shape[:2]
    seg = np.asarray(segments, dtype=np.float64).reshape(-1, 4)
    labels = cluster_line_families(seg)
    # Homogeneous lines (a, b, c) of the longest 5 segments of each image family.
    family_lines = []
    for family in (0, 1):
        members = np.flatnonzero(labels == family)
        lengths = np.hypot(
            seg[members, 2] - seg[members, 0], seg[members, 3] - seg[members, 1]
        )
        members = members[np.argsort(-lengths)[:5]]
        p1 = np.hstack([seg[members, :2], np.ones((len(members), 1))])
        p2 = np.hstack([seg[members, 2:], np.ones((len(members), 1))])
        lines = np.cross(p1, p2)
        family_lines.append(
            lines / np.maximum(np.hypot(lines[:, 0], lines[:, 1]), 1e-12)[:, None]
        )
    if min(len(family_lines[0]), len(family_lines[1])) < 2:
        raise HomographyError("Need at least two lines in each family")
    # Image intersections of every line of family 0 with every line of family 1.
    crossings = np.cross(family_lines[0][:, None, :], family_lines[1][None, :, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        crossings = crossings[..., :2] / crossings[..., 2:3]  # shape (K0, K1, 2)
    # Ordered pairs of distinct lines of each family, as indices (K*(K-1), 2).
    pairs0 = np.array(
        [(a, b) for a in range(len(crossings)) for b in range(len(crossings)) if a != b]
    )
    pairs1 = np.array(
        [
            (a, b)
            for a in range(crossings.shape[1])
            for b in range(crossings.shape[1])
            if a != b
        ]
    )
    # The four intersections of a pair of family-0 lines with a pair of family-1 lines.
    image_quads = np.stack(
        [
            crossings[pairs0[:, None, 0], pairs1[None, :, 0]],
            crossings[pairs0[:, None, 0], pairs1[None, :, 1]],
            crossings[pairs0[:, None, 1], pairs1[None, :, 0]],
            crossings[pairs0[:, None, 1], pairs1[None, :, 1]],
        ],
        axis=2,
    ).reshape(-1, 4, 2)
    image_quads = image_quads[np.isfinite(image_quads).all(axis=(1, 2))]
    # Template lines: constant Y (along the pitch) and constant X (across it).
    length_m, width_m = template.length, template.width
    y_lines = np.array([0.0, *template.goal_area_y, *template.penalty_area_y, width_m])
    y_lines = np.sort(y_lines)
    x_lines = np.array(
        [
            0.0,
            GOAL_AREA_DEPTH_M,
            PENALTY_AREA_DEPTH_M,
            length_m / 2,
            length_m - PENALTY_AREA_DEPTH_M,
            length_m - GOAL_AREA_DEPTH_M,
            length_m,
        ]
    )
    # Unordered pairs (a < b) of template lines of each kind.
    y_pairs = np.array(
        [(a, b) for a in range(len(y_lines)) for b in range(a + 1, len(y_lines))]
    )
    x_pairs = np.array(
        [(a, b) for a in range(len(x_lines)) for b in range(a + 1, len(x_lines))]
    )
    # Template quads for both assignments of image families to template kinds:
    # option 0 maps family 0 to constant-Y lines, option 1 to constant-X lines.
    ya, yb = y_lines[y_pairs[:, 0]][:, None], y_lines[y_pairs[:, 1]][:, None]
    xa, xb = x_lines[x_pairs[:, 0]][None, :], x_lines[x_pairs[:, 1]][None, :]
    ya, yb = (
        np.broadcast_to(ya, (len(y_pairs), len(x_pairs))),
        np.broadcast_to(yb, (len(y_pairs), len(x_pairs))),
    )
    xa, xb = np.broadcast_to(xa, ya.shape), np.broadcast_to(xb, ya.shape)
    option_0 = np.stack(
        [np.stack(p, axis=-1) for p in ((xa, ya), (xb, ya), (xa, yb), (xb, yb))], axis=2
    )
    ya_t, yb_t, xa_t, xb_t = ya.T, yb.T, xa.T, xb.T
    option_1 = np.stack(
        [
            np.stack(p, axis=-1)
            for p in ((xa_t, ya_t), (xa_t, yb_t), (xb_t, ya_t), (xb_t, yb_t))
        ],
        axis=2,
    )
    template_quads = np.concatenate(
        [option_0.reshape(-1, 4, 2), option_1.reshape(-1, 4, 2)]
    )
    # Edge map rasterized from the segments (one sample per pixel of length).
    edges = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in seg:
        steps = max(2, int(np.ceil(np.hypot(x2 - x1, y2 - y1))) + 1)
        cols = np.rint(np.linspace(x1, x2, steps)).astype(np.int64)
        rows = np.rint(np.linspace(y1, y2, steps)).astype(np.int64)
        keep = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        edges[rows[keep], cols[keep]] = True
    # ``near``: True within 3 px of an edge (box window summed with an integral image).
    table = np.pad(
        np.pad(edges, 3).astype(np.int32).cumsum(axis=0).cumsum(axis=1),
        ((1, 0), (1, 0)),
    )
    near = (table[7:, 7:] - table[:-7, 7:] - table[7:, :-7] + table[:-7, :-7]) > 0
    # Coarse samples: straight template lines every 2 m, as homogeneous points.
    coarse = []
    for (x0, y0), (x1, y1) in template.segments():
        count = max(2, int(np.ceil(np.hypot(x1 - x0, y1 - y0) / 2.0)) + 1)
        coarse.append(
            np.stack([np.linspace(x0, x1, count), np.linspace(y0, y1, count)], axis=1)
        )
    coarse_h = np.hstack(
        [np.concatenate(coarse), np.ones((sum(len(c) for c in coarse), 1))]
    )
    lower_half = np.array(
        [
            [0.0, height / 2, 1.0],
            [width, height / 2, 1.0],
            [width, height, 1.0],
            [0.0, height, 1.0],
        ]
    )
    best_homographies: list[np.ndarray] = []
    best_scores: list[float] = []
    # Process the hypotheses in chunks of image quads to bound memory.
    chunk = max(1, 20000 // len(template_quads))
    for start in range(0, len(image_quads), chunk):
        src = np.repeat(image_quads[start : start + chunk], len(template_quads), axis=0)
        dst = np.tile(template_quads, (len(src) // len(template_quads), 1, 1))
        # Linear system of the 4-point homography (h22 = 1): 8 equations, 8 unknowns.
        x, y, u, v = src[..., 0], src[..., 1], dst[..., 0], dst[..., 1]
        zeros, ones = np.zeros_like(x), np.ones_like(x)
        rows_u = np.stack([x, y, ones, zeros, zeros, zeros, -u * x, -u * y], axis=-1)
        rows_v = np.stack([zeros, zeros, zeros, x, y, ones, -v * x, -v * y], axis=-1)
        system = np.concatenate([rows_u, rows_v], axis=1)  # (n, 8, 8)
        rhs = np.concatenate([u, v], axis=1)
        # Skip (nearly) singular systems: Hadamard ratio of the determinant to the
        # row norms (tiny in value because pixel and meter scales differ).
        ratio = np.abs(np.linalg.det(system)) / np.maximum(
            np.prod(np.linalg.norm(system, axis=2), axis=1), 1e-300
        )
        solvable = ratio > 1e-12
        if not solvable.any():
            continue
        solved = np.linalg.solve(system[solvable], rhs[solvable][..., None])[..., 0]
        matrices = np.concatenate([solved, np.ones((len(solved), 1))], axis=1).reshape(
            -1, 3, 3
        )
        # Plausibility of the lower half of the image projected to the pitch.
        scale = lower_half @ np.swapaxes(matrices[:, 2:3, :], 1, 2)  # (n, 4, 1)
        scale = scale[..., 0]
        same_side = np.all(scale > 1e-9, axis=1) | np.all(scale < -1e-9, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            quad = (lower_half @ np.swapaxes(matrices, 1, 2))[..., :2] / scale[
                ..., None
            ]
            sides = np.roll(quad, -1, axis=1) - quad
            turns = sides[..., 0] * np.roll(sides[..., 1], -1, axis=1) - sides[
                ..., 1
            ] * np.roll(sides[..., 0], -1, axis=1)
            area = 0.5 * np.abs(
                np.sum(
                    quad[..., 0] * np.roll(quad[..., 1], -1, axis=1)
                    - np.roll(quad[..., 0], -1, axis=1) * quad[..., 1],
                    axis=1,
                )
            )
            centroid = quad.mean(axis=1)
        plausible = (
            same_side
            & np.all(turns > 0, axis=1)  # convex, not mirrored
            & (area >= 30.0)
            & (area <= 3.0 * length_m * width_m)
            & (centroid[:, 0] >= -20.0)
            & (centroid[:, 0] <= length_m + 20.0)
            & (centroid[:, 1] >= -20.0)
            & (centroid[:, 1] <= width_m + 20.0)
        )
        matrices = matrices[plausible]
        if len(matrices) == 0:
            continue
        # Coarse overlay: project the straight template samples into the image.
        inverse = np.linalg.inv(matrices)
        projected = coarse_h @ np.swapaxes(inverse, 1, 2)  # (n, S, 3)
        with np.errstate(divide="ignore", invalid="ignore"):
            px = np.rint(projected[..., 0] / projected[..., 2])
            py = np.rint(projected[..., 1] / projected[..., 2])
        inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        hits = (
            near[
                np.where(inside, py, 0).astype(np.int64),
                np.where(inside, px, 0).astype(np.int64),
            ]
            & inside
        )
        scores = hits.sum(axis=1) / np.maximum(inside.sum(axis=1), 30)
        top = np.argsort(-scores)[:100]
        best_homographies.extend(matrices[top])
        best_scores.extend(scores[top].tolist())
    if not best_homographies:
        raise HomographyError("No plausible line pairing found")
    # Points along the image segments (about every 8 px) and every straight piece
    # of the template markings: used to check that the detected lines are explained.
    seg_steps = np.maximum(
        np.ceil(np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]) / 8.0), 1
    ).astype(int)
    seg_points = np.concatenate(
        [
            np.stack([np.linspace(x1, x2, n + 1), np.linspace(y1, y2, n + 1)], axis=1)
            for (x1, y1, x2, y2), n in zip(seg, seg_steps, strict=True)
        ]
    )
    pieces = np.concatenate(
        [np.stack([line[:-1], line[1:]], axis=1) for line in template.polylines()]
    )
    piece_start, piece_vector = pieces[:, 0], pieces[:, 1] - pieces[:, 0]
    piece_sq = np.maximum((piece_vector**2).sum(axis=1), 1e-12)
    # Re-rank the best coarse candidates with the real overlay score, multiplied by
    # the fraction of detected-line points that lie on a projected template line.
    final_scores = []
    for index in np.argsort(-np.asarray(best_scores))[:100]:
        candidate = best_homographies[index]
        if not check_homography_validity(candidate, (width, height), template):
            continue
        mapped = apply_homography(candidate, seg_points)  # segment points in meters
        one_px = (
            apply_homography(candidate, seg_points + [3.0, 0.0]) - mapped
        )  # 3 px in meters
        tolerance = np.linalg.norm(one_px, axis=1)
        along = np.clip(
            ((mapped[:, None, :] - piece_start[None]) * piece_vector[None]).sum(axis=2)
            / piece_sq,
            0.0,
            1.0,
        )
        nearest = np.linalg.norm(
            mapped[:, None, :]
            - (piece_start[None] + along[..., None] * piece_vector[None]),
            axis=2,
        ).min(axis=1)
        explained = float(np.mean(nearest <= np.maximum(tolerance, 1e-9)))
        overlay = line_overlay_score(edges.astype(np.uint8) * 255, candidate, template)
        final_scores.append((overlay * explained, index))
    if not final_scores:
        raise HomographyError("No hypothesis passed the validity checks")
    index = max(final_scores)[1]
    return normalize_homography(best_homographies[index])
