"""Overlay renderers for the dashboard images."""

from __future__ import annotations

import cv2
import numpy as np

from datascience.dimensions_2d.boundary_extraction import BoundaryResult
from datascience.dimensions_2d.geometric_fitting import fit_min_area_rect
from datascience.surface_defects.defect_filtering import LinearComponent

GREEN = (0, 220, 0)
RED = (0, 0, 230)
YELLOW = (0, 220, 220)
CYAN = (230, 200, 0)


def draw_roi(image: np.ndarray, roi_rect: tuple[int, int, int, int]) -> np.ndarray:
    out = image.copy()
    x, y, w, h = roi_rect
    cv2.rectangle(out, (x, y), (x + w, y + h), CYAN, 2)
    cv2.putText(out, "ROI", (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)
    return out


def draw_boundary(roi_image: np.ndarray, boundary: BoundaryResult) -> np.ndarray:
    out = roi_image.copy()
    cv2.drawContours(out, [boundary.outer_contour], -1, GREEN, 2)
    for hole in boundary.hole_contours:
        cv2.drawContours(out, [hole], -1, YELLOW, 2)
    return out


def draw_dimensions(
    roi_image: np.ndarray,
    boundary: BoundaryResult,
    mm_per_px: float | None,
) -> np.ndarray:
    """Boundary + min-area rectangle with length/width labels."""
    out = draw_boundary(roi_image, boundary)

    rect = fit_min_area_rect(boundary.outer_contour)
    box = rect.box_points.astype(np.int32)
    cv2.polylines(out, [box], True, RED, 2)

    unit = "mm" if mm_per_px else "px"
    k = mm_per_px if mm_per_px else 1.0
    label = f"L={rect.length_px * k:.1f}{unit}  W={rect.width_px * k:.1f}{unit}"
    cx, cy = int(rect.center[0]), int(rect.center[1])
    cv2.putText(out, label, (max(cx - 120, 5), max(cy - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
    return out


def draw_linear_defects(
    roi_image: np.ndarray,
    components: list[LinearComponent],
    color: tuple[int, int, int],
    label: str,
) -> np.ndarray:
    out = roi_image.copy()
    for i, comp in enumerate(components, start=1):
        contours, _ = cv2.findContours(
            comp.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, color, 2)
        ys, xs = np.nonzero(comp.mask)
        if len(xs):
            cv2.putText(out, f"{label}{i}", (int(xs.min()), max(int(ys.min()) - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


def draw_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a binary mask onto the image (dents, pothole segmentation)."""
    out = image.copy()
    colored = np.zeros_like(out)
    colored[mask > 0] = color
    return cv2.addWeighted(out, 1.0, colored, alpha, 0)


def draw_pothole_detections(
    image: np.ndarray,
    masks: list[np.ndarray],
    detections: list,
) -> np.ndarray:
    out = image.copy()
    for mask, det in zip(masks, detections):
        out = draw_mask_overlay(out, mask, RED)
        x1, y1, x2, y2 = (int(v) for v in det.bbox_xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), RED, 2)
        cv2.putText(out, f"pothole {det.confidence:.2f} [{det.severity}]",
                    (x1, max(y1 - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2)
    return out
