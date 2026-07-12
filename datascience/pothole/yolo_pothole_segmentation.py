"""YOLO segmentation — used ONLY for pothole detection (per spec).

The model is loaded lazily from the configured weights path. If the weights
file is missing or ultralytics is not installed, the module reports itself
as disabled instead of crashing the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from datascience.config_loader import PACKAGE_ROOT, load_system_config

_model = None
_model_error: str | None = None


def _load_model():
    global _model, _model_error
    if _model is not None or _model_error is not None:
        return _model

    cfg = load_system_config().get("pothole", {})
    weights = Path(cfg.get("weights_path", "models/pothole_yolov8_seg.pt"))
    if not weights.is_absolute():
        weights = PACKAGE_ROOT / weights

    if not weights.exists():
        _model_error = f"pothole weights not found: {weights}"
        return None

    try:
        from ultralytics import YOLO
    except ImportError:
        _model_error = "ultralytics not installed (pip install ultralytics)"
        return None

    try:
        _model = YOLO(str(weights))
    except Exception as exc:
        _model_error = f"failed to load YOLO weights: {exc}"
        return None
    return _model


def segment_potholes(
    image: np.ndarray,
) -> tuple[list[dict], str | None]:
    """Run YOLO segmentation.

    Returns (detections, error). Each detection:
    {"confidence": float, "bbox_xyxy": [x1,y1,x2,y2], "mask": HxW uint8}
    """
    cfg = load_system_config().get("pothole", {})
    model = _load_model()
    if model is None:
        return [], _model_error

    results = model.predict(
        image,
        conf=float(cfg.get("confidence", 0.4)),
        verbose=False,device="cuda"
    )

    detections: list[dict] = []
    h, w = image.shape[:2]

    for res in results:
        if res.masks is None:
            continue
        boxes = res.boxes
        for i, mask_data in enumerate(res.masks.data):
            mask = mask_data.cpu().numpy().astype(np.uint8) * 255
            if mask.shape != (h, w):
                import cv2

                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            detections.append(
                {
                    "confidence": float(boxes.conf[i].item()),
                    "bbox_xyxy": [float(v) for v in boxes.xyxy[i].tolist()],
                    "mask": mask,
                }
            )
    return detections, None
