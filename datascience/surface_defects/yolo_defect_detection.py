"""YOLO object detection for metal surface defects.

The primary surface-defect detector: one model (`models/yolo_model_final.pt`)
that localizes all five defect classes as bounding boxes:

    Crack · Dent · Missing-head · Paint-off · Scratch

The model is loaded lazily from the configured weights path. If the weights
file is missing or ultralytics is not installed, the module reports itself as
disabled instead of crashing the pipeline (same contract as the pothole
module). This model is *detection* only — it outputs boxes, not masks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from datascience.config_loader import PACKAGE_ROOT, load_system_config

_model = None
_model_error: str | None = None


def _config() -> dict:
    return load_system_config().get("defect_detection", {})


def _load_model():
    global _model, _model_error
    if _model is not None or _model_error is not None:
        return _model

    cfg = _config()
    weights = Path(cfg.get("weights_path", "models/yolo_model_final.pt"))
    if not weights.is_absolute():
        weights = PACKAGE_ROOT / weights

    if not weights.exists():
        _model_error = f"defect-detection weights not found: {weights}"
        return None

    try:
        from ultralytics import YOLO
    except ImportError:
        _model_error = "ultralytics not installed (pip install ultralytics)"
        return None

    try:
        _model = YOLO(str(weights))
    except Exception as exc:
        _model_error = f"failed to load defect-detection weights: {exc}"
        return None
    return _model


def detect_defects(image: np.ndarray) -> tuple[list[dict], str | None]:
    """Run YOLO defect detection on a full BGR frame.

    Returns (detections, error). Each detection:
    {"label": str, "confidence": float, "bbox_xyxy": [x1, y1, x2, y2]}
    """
    cfg = _config()
    model = _load_model()
    if model is None:
        return [], _model_error

    results = model.predict(
        image,
        conf=float(cfg.get("confidence", 0.25)),
        verbose=False,
    )

    detections: list[dict] = []
    for res in results:
        boxes = res.boxes
        if boxes is None:
            continue
        names = res.names  # {class_id: label}
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            detections.append(
                {
                    "label": str(names.get(cls_id, cls_id)),
                    "confidence": float(boxes.conf[i].item()),
                    "bbox_xyxy": [float(v) for v in boxes.xyxy[i].tolist()],
                }
            )
    return detections, None
