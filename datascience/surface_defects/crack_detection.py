"""Crack detection on metal surfaces — classical CV only.

Cracks appear as thin dark tortuous lines. Pipeline:
CLAHE-normalized gray -> black-hat (dark line enhancement) -> ridge
reinforcement -> threshold -> morphology -> connected components ->
linear-component filtering (crack = elongated, often branched, not straight).
"""

from __future__ import annotations

import cv2
import numpy as np

from datascience.config_loader import load_system_config
from datascience.schemas import CrackMetrics

from datascience.surface_defects.defect_filtering import (
    LinearComponent,
    extract_linear_components,
    suppress_reflections,
)


def _enhance_dark_lines(gray: np.ndarray) -> np.ndarray:
    """Black-hat morphological filtering highlights thin dark structures."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Ridge reinforcement: second-derivative response along both axes keeps
    # line-like structures and suppresses flat texture.
    lap = cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F, ksize=5)
    ridge = np.clip(lap, 0, None)  # positive Laplacian = dark line on bright bg
    ridge = cv2.normalize(ridge, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return cv2.addWeighted(blackhat, 0.7, ridge, 0.3, 0)


def detect_cracks(
    gray: np.ndarray,
    product_mask: np.ndarray | None = None,
) -> tuple[list[CrackMetrics], list[LinearComponent]]:
    """Return crack metrics + raw components (for overlay drawing)."""
    cfg = load_system_config().get("surface_defects", {})

    enhanced = _enhance_dark_lines(gray)

    # Adaptive threshold handles residual illumination gradients.
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    binary = suppress_reflections(gray, binary)

    if product_mask is not None:
        binary = cv2.bitwise_and(binary, product_mask)

    # Connect fragmented crack segments, then drop single-pixel noise.
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    components = extract_linear_components(
        binary,
        min_length_px=float(cfg.get("min_crack_length_px", 40)),
        min_elongation=float(cfg.get("min_elongation", 3.0)),
        max_area_ratio=float(cfg.get("max_component_area_ratio", 0.2)),
    )

    # Cracks are tortuous/branched; near-perfectly straight lines are more
    # likely scratches or machining marks and are handled by scratch_detection.
    crack_components = [
        c for c in components if c.straightness < 0.9 or c.branch_count > 0
    ]

    metrics = [
        CrackMetrics(
            length=c.length_px,
            max_width=c.max_width_px,
            avg_width=c.avg_width_px,
            area=c.area_px,
            orientation_deg=c.orientation_deg,
            branch_count=c.branch_count,
            unit="px",
        )
        for c in crack_components
    ]
    return metrics, crack_components
