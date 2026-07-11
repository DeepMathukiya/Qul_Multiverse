"""Qwen3-VL-based pothole detection (replaces YOLO segmentation).

No pixel mask is available from a VLM, so detections carry only a
qualitative severity + description — no geometry, no overlay image.
"""

from __future__ import annotations

import numpy as np

from datascience.config_loader import load_system_config
from datascience.schemas import CheckStatus, PotholeDetection, PotholeResult
from datascience.vlm.geniex_client import call_vlm
from datascience.vlm.prompts import pothole_prompt
from datascience.vlm.response_json import parse_json_response

_VALID_SEVERITIES = {"low", "medium", "high"}


def analyze_potholes(
    vertical: np.ndarray, horizontal: np.ndarray
) -> tuple[PotholeResult, list[np.ndarray]]:
    """Run Qwen3-VL pothole detection.

    Returns (result, masks) — masks is always empty (no pixel mask from a
    VLM); kept in the return signature so the pipeline's existing
    overlay-drawing guard (`if masks:`) still works unchanged.
    """
    cfg = load_system_config().get("pothole", {})
    result = PotholeResult(enabled=bool(cfg.get("enabled", True)))

    if not result.enabled:
        result.status = CheckStatus.SKIPPED
        result.note = "pothole analysis disabled in processing_config.yaml"
        return result, []

    text, error = call_vlm([vertical, horizontal], pothole_prompt())
    if error:
        result.status = CheckStatus.NOT_AVAILABLE
        result.error = error
        return result, []

    data = parse_json_response(text)
    if not isinstance(data, list):
        result.status = CheckStatus.NOT_AVAILABLE
        result.error = "VLM did not return a valid JSON array"
        return result, []

    for item in data:
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity", "low")).lower()
        if severity not in _VALID_SEVERITIES:
            severity = "low"
        result.detections.append(
            PotholeDetection(
                severity=severity,
                description=item.get("description") or None,
            )
        )

    result.present = len(result.detections) > 0
    result.status = CheckStatus.PASS
    result.note = "qwen3-vl qualitative detection — no calibrated geometry"
    return result, []
