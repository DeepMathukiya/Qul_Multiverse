"""Dent detection using stereo depth (metric, calibration-based).

A dent is a region of the product surface lying measurably below the local
surface plane. Requires the stereo depth map — without it the check reports
NOT_AVAILABLE (spec: dent depth must come from stereo, never from pixels).
"""

from __future__ import annotations

import cv2
import numpy as np

from datascience.config_loader import load_system_config
from datascience.schemas import DentMetrics

from datascience.dimensions_3d.depth_reconstruction import DepthData
from datascience.dimensions_3d.measurements_3d import plane_deviation_map


def detect_dents(
    depth: DepthData,
    product_mask: np.ndarray,
) -> tuple[list[DentMetrics], np.ndarray | None]:
    """Return dent metrics (mm) + a binary dent mask for overlays."""
    cfg = load_system_config().get("surface_defects", {})
    depth_threshold_mm = float(cfg.get("dent_depth_threshold_mm", 0.5))

    deviation = plane_deviation_map(depth, product_mask)
    if deviation is None:
        return [], None

    # Fit a *product surface* reference: median deviation of the product
    # itself, so a raised product on a table doesn't read as one huge dent.
    finite = np.isfinite(deviation)
    if finite.sum() < 200:
        return [], None
    surface_level = float(np.nanmedian(deviation))

    # Dent = locally below the product surface by more than the threshold.
    below = np.zeros(deviation.shape, dtype=np.uint8)
    below[finite & (deviation < surface_level - depth_threshold_mm)] = 255

    below = cv2.morphologyEx(below, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    below = cv2.morphologyEx(below, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(below, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dents: list[DentMetrics] = []
    dent_mask = np.zeros_like(below)

    for contour in contours:
        area_px = cv2.contourArea(contour)
        if area_px < 50:
            continue

        region = np.zeros_like(below)
        cv2.drawContours(region, [contour], -1, 255, cv2.FILLED)
        region_dev = deviation[(region > 0) & finite]
        if region_dev.size < 20:
            continue

        max_depth_mm = float(surface_level - np.nanpercentile(region_dev, 2))
        deformation_mm = float(np.nanstd(region_dev))

        # Convert px footprint to mm using the local 3D point spacing.
        ys, xs = np.nonzero(region)
        pts = depth.points_3d[ys, xs]
        span_x = float(np.nanpercentile(pts[:, 0], 98) - np.nanpercentile(pts[:, 0], 2))
        span_y = float(np.nanpercentile(pts[:, 1], 98) - np.nanpercentile(pts[:, 1], 2))
        diameter_mm = max(span_x, span_y)
        area_mm2 = span_x * span_y * (area_px / max(region.sum() / 255, 1))

        dents.append(
            DentMetrics(
                area=round(area_mm2, 1),
                diameter=round(diameter_mm, 2),
                max_depth=round(max_depth_mm, 2),
                deformation=round(deformation_mm, 3),
                unit="mm",
            )
        )
        dent_mask = cv2.bitwise_or(dent_mask, region)

    return dents, (dent_mask if dents else None)
