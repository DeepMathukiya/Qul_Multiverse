"""Compute 2D product measurements from the extracted boundary.

Pixel values are converted to millimeters only when a valid scale exists
(camera calibration reference or configured mm_per_px). Otherwise values are
reported in px and mm-based tolerance checks become NOT_AVAILABLE.
"""

from __future__ import annotations

import cv2

from datascience.schemas import CheckStatus, Dim2DResult, Measurement

from datascience.calibration.scale_reference import get_scale
from datascience.dimensions_2d.boundary_extraction import BoundaryResult
from datascience.dimensions_2d.geometric_fitting import (
    corner_angles,
    fit_enclosing_circle,
    fit_hole_circles,
    fit_min_area_rect,
    hole_center_distances,
    roundness,
)


def measure_product(boundary: BoundaryResult | None) -> Dim2DResult:
    result = Dim2DResult()

    if boundary is None:
        result.status = CheckStatus.NOT_AVAILABLE
        result.error = "no product boundary found in ROI"
        return result

    result.boundary_found = True

    mm_per_px, source = get_scale()
    result.mm_per_px = mm_per_px
    result.scale_source = source

    unit = "mm" if mm_per_px else "px"
    unit_sq = "mm2" if mm_per_px else "px2"
    k = mm_per_px if mm_per_px else 1.0

    contour = boundary.outer_contour
    measurements: list[Measurement] = []

    rect = fit_min_area_rect(contour)
    measurements.append(Measurement(name=f"length_{unit}", value=round(rect.length_px * k, 2), unit=unit))
    measurements.append(Measurement(name=f"width_{unit}", value=round(rect.width_px * k, 2), unit=unit))

    circle = fit_enclosing_circle(contour)
    measurements.append(Measurement(name=f"diameter_{unit}", value=round(2 * circle.radius_px * k, 2), unit=unit))
    measurements.append(Measurement(name=f"radius_{unit}", value=round(circle.radius_px * k, 2), unit=unit))

    area_px = cv2.contourArea(contour)
    perimeter_px = cv2.arcLength(contour, True)
    measurements.append(Measurement(name=f"area_{unit_sq}", value=round(area_px * k * k, 2), unit=unit_sq))
    measurements.append(Measurement(name=f"perimeter_{unit}", value=round(perimeter_px * k, 2), unit=unit))
    measurements.append(Measurement(name="roundness", value=round(roundness(contour), 4), unit="ratio"))

    holes = fit_hole_circles(boundary.hole_contours)
    for i, hole in enumerate(holes, start=1):
        measurements.append(
            Measurement(
                name=f"hole_{i}_diameter_{unit}",
                value=round(2 * hole.radius_px * k, 2),
                unit=unit,
            )
        )
    for i, dist in enumerate(hole_center_distances(holes), start=1):
        measurements.append(
            Measurement(
                name=f"hole_distance_{i}_{unit}",
                value=round(dist * k, 2),
                unit=unit,
            )
        )

    for i, angle in enumerate(corner_angles(contour), start=1):
        measurements.append(
            Measurement(name=f"corner_angle_{i}_deg", value=round(angle, 1), unit="deg")
        )

    result.measurements = measurements
    result.status = CheckStatus.PASS
    return result
