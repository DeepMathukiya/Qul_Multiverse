"""Shared component analysis + false-positive rules for surface defects.

Classical geometry/photometric filters that reject blobs caused by noise,
reflections, texture and machining marks before anything is reported as a
crack or scratch.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LinearComponent:
    mask: np.ndarray            # component mask (uint8 0/255), full-frame size
    skeleton: np.ndarray        # thinned centerline (uint8 0/255)
    length_px: float            # skeleton length
    max_width_px: float
    avg_width_px: float
    area_px: float
    orientation_deg: float
    branch_count: int
    elongation: float           # length^2 / area — high for thin lines
    straightness: float         # endpoint distance / skeleton length (1 = straight)


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Thin a binary mask to a 1-px centerline (ximgproc if available)."""
    if hasattr(cv2, "ximgproc"):
        return cv2.ximgproc.thinning(mask)

    # Morphological fallback (Zhang-Suen unavailable without contrib).
    skeleton = np.zeros_like(mask)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    work = mask.copy()
    while cv2.countNonZero(work) > 0:
        eroded = cv2.erode(work, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened))
        work = eroded
    return skeleton


def _branch_and_end_points(skeleton: np.ndarray) -> tuple[int, np.ndarray]:
    """Count branch points and return endpoint coordinates of a skeleton."""
    binary = (skeleton > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    neighbor_count = cv2.filter2D(binary, cv2.CV_8U, kernel) - binary

    branch_points = int(((binary == 1) & (neighbor_count >= 4)).sum())
    end_ys, end_xs = np.nonzero((binary == 1) & (neighbor_count == 1))
    endpoints = np.column_stack([end_xs, end_ys])
    return branch_points, endpoints


def analyze_component(component_mask: np.ndarray) -> LinearComponent | None:
    """Measure a single connected component as a potential linear defect."""
    area = float(cv2.countNonZero(component_mask))
    if area < 10:
        return None

    skeleton = skeletonize(component_mask)
    length = float(cv2.countNonZero(skeleton))
    if length < 5:
        return None

    # Width from the distance transform sampled along the centerline:
    # distance to the nearest background pixel = half width.
    dist = cv2.distanceTransform(component_mask, cv2.DIST_L2, 3)
    widths = 2.0 * dist[skeleton > 0]
    max_width = float(widths.max()) if widths.size else 0.0
    avg_width = float(widths.mean()) if widths.size else 0.0

    # Orientation of the principal axis.
    ys, xs = np.nonzero(component_mask)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    orientation = float(np.degrees(np.arctan2(vy, vx))) % 180.0

    branch_count, endpoints = _branch_and_end_points(skeleton)
    if len(endpoints) >= 2:
        # Longest endpoint-to-endpoint span vs skeleton length.
        span = 0.0
        for i in range(len(endpoints)):
            d = np.linalg.norm(endpoints[i] - endpoints, axis=1).max()
            span = max(span, float(d))
        straightness = min(1.0, span / max(length, 1.0))
    else:
        straightness = 0.0

    return LinearComponent(
        mask=component_mask,
        skeleton=skeleton,
        length_px=length,
        max_width_px=round(max_width, 2),
        avg_width_px=round(avg_width, 2),
        area_px=area,
        orientation_deg=round(orientation, 1),
        branch_count=branch_count,
        elongation=round(length * length / area, 2),
        straightness=round(straightness, 3),
    )


def extract_linear_components(
    binary: np.ndarray,
    min_length_px: float,
    min_elongation: float,
    max_area_ratio: float,
) -> list[LinearComponent]:
    """Split a defect-candidate binary image into filtered linear components.

    Each component is analyzed on its bounding-box crop (with padding), then
    the masks are pasted back to full-frame size — thinning/distance
    transforms on full frames per component would be prohibitively slow.
    """
    frame_h, frame_w = binary.shape[:2]
    frame_area = frame_h * frame_w
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    components: list[LinearComponent] = []
    for label in range(1, n_labels):
        x, y, w, h, area = stats[label]

        if area > max_area_ratio * frame_area:
            continue  # huge region: shadow / background, not a defect
        if area < 10 or max(w, h) < min_length_px * 0.5:
            continue  # cannot possibly reach the minimum length

        pad = 3
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(frame_w, x + w + pad), min(frame_h, y + h + pad)
        crop_mask = ((labels[y0:y1, x0:x1] == label) * 255).astype(np.uint8)

        comp = analyze_component(crop_mask)
        if comp is None:
            continue
        if comp.length_px < min_length_px:
            continue  # too short: noise / texture speckle
        if comp.elongation < min_elongation:
            continue  # blob-like: reflection, stain, machining pit

        # Paste crop-space masks back into full-frame coordinates.
        full_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        full_mask[y0:y1, x0:x1] = comp.mask
        full_skeleton = np.zeros((frame_h, frame_w), dtype=np.uint8)
        full_skeleton[y0:y1, x0:x1] = comp.skeleton
        comp.mask = full_mask
        comp.skeleton = full_skeleton

        components.append(comp)
    return components


def suppress_reflections(gray: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Remove candidate pixels sitting inside saturated (specular) regions."""
    _, glare = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    glare = cv2.dilate(glare, np.ones((7, 7), np.uint8))
    return cv2.bitwise_and(candidates, cv2.bitwise_not(glare))
