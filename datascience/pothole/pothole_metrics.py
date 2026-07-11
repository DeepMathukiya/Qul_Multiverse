"""Geometric metrics computed from YOLO pothole segmentation masks."""

from __future__ import annotations

import cv2
import numpy as np

from datascience.config_loader import load_system_config
from datascience.schemas import CheckStatus, PotholeDetection, PotholeResult

from datascience.calibration.scale_reference import get_scale
from datascience.pothole.yolo_pothole_segmentation import segment_potholes


def _severity(area_ratio: float, cfg: dict) -> str:
    if area_ratio >= float(cfg.get("severity_high_ratio", 0.08)):
        return "high"
    if area_ratio >= float(cfg.get("severity_medium_ratio", 0.02)):
        return "medium"
    return "low"


def analyze_potholes(image: np.ndarray) -> tuple[PotholeResult, list[np.ndarray]]:
    """Run YOLO pothole segmentation and compute metrics per detection.

    Real-world (mm) values are attached only when a valid scale exists —
    otherwise metrics stay in pixels (spec: no pixel==mm assumption).
    Returns (result, masks) — masks for overlay drawing.
    """
    cfg = load_system_config().get("pothole", {})
    result = PotholeResult(enabled=bool(cfg.get("enabled", True)))

    if not result.enabled:
        result.status = CheckStatus.SKIPPED
        result.note = "pothole analysis disabled in system_config.yaml"
        return result, []

    detections_raw, error = segment_potholes(image)
    if error:
        result.status = CheckStatus.NOT_AVAILABLE
        result.error = error
        return result, []

    mm_per_px, scale_source = get_scale()
    frame_area = float(image.shape[0] * image.shape[1])
    unit = "mm" if mm_per_px else "px"
    k = mm_per_px if mm_per_px else 1.0

    masks: list[np.ndarray] = []
    for det in detections_raw:
        mask = det["mask"]
        area_px = float(cv2.countNonZero(mask))
        if area_px < 10:
            continue

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        perimeter_px = float(cv2.arcLength(largest, True))

        # Max width = longer side of the minimum-area rectangle.
        (_, _), (rw, rh), _ = cv2.minAreaRect(largest)
        max_width_px = float(max(rw, rh))

        result.detections.append(
            PotholeDetection(
                confidence=round(det["confidence"], 3),
                bbox_xyxy=[round(v, 1) for v in det["bbox_xyxy"]],
                area=round(area_px * k * k, 1),
                perimeter=round(perimeter_px * k, 1),
                max_width=round(max_width_px * k, 1),
                unit=unit,
                severity=_severity(area_px / frame_area, cfg),
            )
        )
        masks.append(mask)

    result.present = len(result.detections) > 0
    result.status = CheckStatus.PASS
    if mm_per_px is None:
        result.note = "metrics in pixels — no calibration/scale for real-world units"
    else:
        result.note = f"real-world units via {scale_source} scale"
    return result, masks
