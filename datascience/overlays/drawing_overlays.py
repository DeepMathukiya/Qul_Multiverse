"""Overlay renderers for the dashboard images."""

from __future__ import annotations

import cv2
import numpy as np

from datascience.dimensions_2d.boundary_extraction import BoundaryResult
from datascience.dimensions_2d.geometric_fitting import fit_min_area_rect

GREEN = (0, 220, 0)
RED = (0, 0, 230)
YELLOW = (0, 220, 220)
CYAN = (230, 200, 0)
ORANGE = (0, 140, 255)
MAGENTA = (200, 0, 200)

# One color per YOLO defect class (BGR). Unknown labels fall back to RED.
DEFECT_COLORS = {
    "Crack": RED,
    "Dent": ORANGE,
    "Missing-head": MAGENTA,
    "Paint-off": YELLOW,
    "Scratch": CYAN,
}


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


def draw_defect_boxes(image: np.ndarray, detections: list) -> np.ndarray:
    """Draw YOLO defect bounding boxes (color-coded per class) with labels."""
    out = image.copy()
    for det in detections:
        color = DEFECT_COLORS.get(det.label, RED)
        x1, y1, x2, y2 = (int(v) for v in det.bbox_xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{det.label} {det.confidence:.2f}",
                    (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
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


def compose_annotated_frame(
    frame: np.ndarray,
    roi_rect: tuple[int, int, int, int],
    boundary,
    mm_per_px: float | None,
    defect_detections: list,
    pothole_masks: list,
    pothole_detections: list,
    result,
) -> np.ndarray:
    """Burn ALL inspection findings into one live-stream frame:

    PASS/FAIL banner + key numbers on top, boundary + dimensions inside the
    ROI, YOLO defect boxes + pothole overlays on the full frame.
    """
    out = frame.copy()
    x, y, w, h = roi_rect

    # ---- annotate inside the ROI ----
    crop = out[y : y + h, x : x + w]
    if boundary is not None:
        crop = draw_dimensions(crop, boundary, mm_per_px)
    out[y : y + h, x : x + w] = crop
    cv2.rectangle(out, (x, y), (x + w, y + h), CYAN, 2)

    # ---- full-frame overlays (YOLO boxes are in full-frame coords) ----
    if defect_detections:
        out = draw_defect_boxes(out, defect_detections)
    if pothole_masks:
        out = draw_pothole_detections(out, pothole_masks, pothole_detections)

    # ---- status banner ----
    passed = bool(result.quality and result.quality.overall_pass)
    banner_color = (60, 160, 40) if passed else (40, 40, 210)
    frame_w = out.shape[1]
    bar_h = 78

    banner = out.copy()
    cv2.rectangle(banner, (0, 0), (frame_w, bar_h), banner_color, -1)
    out = cv2.addWeighted(banner, 0.8, out, 0.2, 0)

    cv2.putText(out, "PASS" if passed else "FAIL", (12, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

    defects = result.surface_defects
    if defects.counts:
        defect_str = " ".join(f"{lbl}:{n}" for lbl, n in defects.counts.items())
    else:
        defect_str = f"{len(defects.detections)} found" if defects.present else "none"
    info = (
        f"defects[{defect_str}]  QR:{'Y' if result.ocr.qr_present else 'N'}  "
        f"exp:{result.ocr.fields.expiry_date or '-'}"
    )
    dims = {m.name.rsplit("_", 1)[0]: f"{m.value}{m.unit}"
            for m in result.dims_2d.measurements}
    if "length" in dims and "width" in dims:
        info = f"L:{dims['length']}  W:{dims['width']}  " + info
    cv2.putText(out, info, (150, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # First failure reason (if any) on the second banner line.
    if result.quality and result.quality.failure_reasons:
        reason = result.quality.failure_reasons[0]
        if len(reason) > 90:
            reason = reason[:87] + "..."
        cv2.putText(out, reason, (150, 58), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return out
