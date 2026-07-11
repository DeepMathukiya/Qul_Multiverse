"""Build the surface-defect result from YOLO detections.

Geometry is bounding-box derived only (the model outputs boxes, not masks):
width, height and area come straight from the box. Real-world (mm) values are
attached only when a valid scale exists — otherwise metrics stay in pixels
(spec: never assume pixels == millimeters).
"""

from __future__ import annotations

import numpy as np

from datascience.calibration.scale_reference import get_scale
from datascience.config_loader import load_system_config
from datascience.schemas import CheckStatus, DefectDetection, SurfaceDefectResult
from datascience.surface_defects.yolo_defect_detection import detect_defects


def analyze_surface_defects(image: np.ndarray) -> tuple[SurfaceDefectResult, list[DefectDetection]]:
    """Run YOLO defect detection and package the metrics.

    Returns (result, detections) — detections are also returned separately so
    overlay drawing does not have to reach into the result object.
    """
    cfg = load_system_config().get("defect_detection", {})
    result = SurfaceDefectResult(enabled=bool(cfg.get("enabled", True)))

    if not result.enabled:
        result.status = CheckStatus.SKIPPED
        result.note = "defect detection disabled in processing_config.yaml"
        return result, []

    detections_raw, error = detect_defects(image)
    if error:
        result.status = CheckStatus.NOT_AVAILABLE
        result.error = error
        return result, []

    mm_per_px, scale_source = get_scale()
    unit = "mm" if mm_per_px else "px"
    k = mm_per_px if mm_per_px else 1.0

    counts: dict[str, int] = {}
    for det in detections_raw:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        w_px = max(0.0, x2 - x1)
        h_px = max(0.0, y2 - y1)

        result.detections.append(
            DefectDetection(
                label=det["label"],
                confidence=round(det["confidence"], 3),
                bbox_xyxy=[round(v, 1) for v in det["bbox_xyxy"]],
                width=round(w_px * k, 1),
                height=round(h_px * k, 1),
                area=round(w_px * h_px * k * k, 1),
                unit=unit,
            )
        )
        counts[det["label"]] = counts.get(det["label"], 0) + 1

    result.counts = counts
    result.present = len(result.detections) > 0
    result.status = CheckStatus.PASS
    if mm_per_px is None:
        result.note = "metrics in pixels — no calibration/scale for real-world units"
    else:
        result.note = f"real-world units via {scale_source} scale"
    return result, result.detections
