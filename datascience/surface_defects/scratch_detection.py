"""Scratch detection on metal surfaces — classical CV only.

Scratches are thin, mostly straight lines that can be brighter or darker
than the surrounding metal. Directional top-hat + black-hat enhancement
followed by linear-component filtering with a straightness requirement.
"""

from __future__ import annotations

import cv2
import numpy as np

from datascience.config_loader import load_system_config
from datascience.schemas import ScratchMetrics

from datascience.surface_defects.defect_filtering import (
    LinearComponent,
    extract_linear_components,
    suppress_reflections,
)


def _directional_line_response(gray: np.ndarray) -> np.ndarray:
    """Max top-hat/black-hat response over several line orientations."""
    responses = []
    for angle in (0, 45, 90, 135):
        # Long thin structuring element approximating a line at `angle`.
        kernel = np.zeros((15, 15), np.uint8)
        cv2.line(
            kernel,
            (7 - int(7 * np.cos(np.radians(angle))), 7 - int(7 * np.sin(np.radians(angle)))),
            (7 + int(7 * np.cos(np.radians(angle))), 7 + int(7 * np.sin(np.radians(angle)))),
            1,
            1,
        )
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)    # bright scratches
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)  # dark scratches
        responses.append(cv2.max(tophat, blackhat))

    out = responses[0]
    for r in responses[1:]:
        out = cv2.max(out, r)
    return out


def detect_scratches(
    gray: np.ndarray,
    product_mask: np.ndarray | None = None,
    exclude_mask: np.ndarray | None = None,
) -> tuple[list[ScratchMetrics], list[LinearComponent]]:
    """Return scratch metrics + raw components.

    exclude_mask: pixels already claimed as cracks (avoids double reporting).
    """
    cfg = load_system_config().get("surface_defects", {})

    response = _directional_line_response(gray)
    _, binary = cv2.threshold(response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    binary = suppress_reflections(gray, binary)

    if product_mask is not None:
        binary = cv2.bitwise_and(binary, product_mask)
    if exclude_mask is not None:
        binary = cv2.bitwise_and(binary, cv2.bitwise_not(exclude_mask))

    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    components = extract_linear_components(
        binary,
        min_length_px=float(cfg.get("min_scratch_length_px", 30)),
        min_elongation=float(cfg.get("min_elongation", 3.0)),
        max_area_ratio=float(cfg.get("max_component_area_ratio", 0.2)),
    )

    # Scratches: straight, unbranched lines.
    scratch_components = [
        c for c in components if c.straightness >= 0.85 and c.branch_count <= 1
    ]

    metrics = [
        ScratchMetrics(
            length=c.length_px,
            width=c.avg_width_px,
            area=c.area_px,
            orientation_deg=c.orientation_deg,
            unit="px",
        )
        for c in scratch_components
    ]
    return metrics, scratch_components
