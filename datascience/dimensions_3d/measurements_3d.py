"""Metric 3D measurements from reconstructed stereo depth.

All values here are only produced when a valid stereo calibration exists —
the pipeline never converts raw pixels to millimeters without it.
"""

from __future__ import annotations

import numpy as np

from datascience.schemas import CheckStatus, Dim3DResult, Measurement

from datascience.dimensions_3d.depth_reconstruction import DepthData


def _fit_background_plane(
    points: np.ndarray, sample_mask: np.ndarray
) -> tuple[np.ndarray, float] | None:
    """Least-squares plane z = ax + by + c over the sampled points.

    Returns (coeffs [a, b, c], rms) or None if too few samples.
    """
    ys, xs = np.nonzero(sample_mask)
    if len(xs) < 100:
        return None
    # Subsample for speed.
    idx = np.random.default_rng(0).choice(len(xs), min(len(xs), 5000), replace=False)
    xs, ys = xs[idx], ys[idx]

    pts = points[ys, xs]  # N x 3 (X, Y, Z in mm)
    A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
    z = pts[:, 2]
    coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
    rms = float(np.sqrt(np.mean((A @ coeffs - z) ** 2)))
    return coeffs, rms


def plane_deviation_map(depth: DepthData, product_mask: np.ndarray) -> np.ndarray | None:
    """Signed deviation (mm) of each product pixel from the surrounding
    background plane. Positive = closer to camera than the plane."""
    background = depth.valid_mask & (product_mask == 0)
    fit = _fit_background_plane(depth.points_3d, background)
    if fit is None:
        return None
    coeffs, _ = fit

    h, w = depth.depth_map.shape
    deviation = np.full((h, w), np.nan, dtype=np.float32)

    ys, xs = np.nonzero(depth.valid_mask & (product_mask > 0))
    if len(xs) == 0:
        return None
    pts = depth.points_3d[ys, xs]
    plane_z = coeffs[0] * pts[:, 0] + coeffs[1] * pts[:, 1] + coeffs[2]
    # Camera looks along +Z: smaller Z = closer = higher above the plane.
    deviation[ys, xs] = plane_z - pts[:, 2]
    return deviation


def measure_product_3d(depth: DepthData, product_mask: np.ndarray) -> Dim3DResult:
    """Height/length/width/depth + surface statistics of the masked product."""
    result = Dim3DResult(calibrated=True)

    on_product = depth.valid_mask & (product_mask > 0)
    if on_product.sum() < 200:
        result.status = CheckStatus.NOT_AVAILABLE
        result.note = "too few valid depth pixels on the product"
        return result

    ys, xs = np.nonzero(on_product)
    pts = depth.points_3d[ys, xs]  # mm

    # Robust extents (percentiles reject disparity outliers).
    def robust_range(v: np.ndarray) -> float:
        return float(np.percentile(v, 98) - np.percentile(v, 2))

    length_mm = robust_range(pts[:, 0])
    width_mm = robust_range(pts[:, 1])
    depth_extent_mm = robust_range(pts[:, 2])

    measurements = [
        Measurement(name="length_mm", value=round(length_mm, 2), unit="mm"),
        Measurement(name="width_mm", value=round(width_mm, 2), unit="mm"),
        Measurement(name="depth_mm", value=round(depth_extent_mm, 2), unit="mm"),
    ]

    # Height above the background plane (e.g. product standing on a table).
    deviation = plane_deviation_map(depth, product_mask)
    if deviation is not None:
        dev_vals = deviation[np.isfinite(deviation)]
        if dev_vals.size:
            height_mm = float(np.percentile(dev_vals, 98))
            surface_range = float(
                np.percentile(dev_vals, 98) - np.percentile(dev_vals, 2)
            )
            measurements.append(
                Measurement(name="height_mm", value=round(height_mm, 2), unit="mm")
            )
            measurements.append(
                Measurement(
                    name="surface_height_range_mm",
                    value=round(surface_range, 2),
                    unit="mm",
                )
            )
            measurements.append(
                Measurement(
                    name="surface_deformation_rms_mm",
                    value=round(float(np.std(dev_vals)), 3),
                    unit="mm",
                )
            )

            # Approximate volume: sum of height above plane x pixel footprint.
            # Footprint from median depth and the point grid spacing.
            positive = dev_vals[dev_vals > 0]
            if positive.size > 100:
                px_area_mm2 = _median_pixel_footprint_mm2(depth, on_product)
                if px_area_mm2 is not None:
                    volume_mm3 = float(positive.sum() * px_area_mm2)
                    measurements.append(
                        Measurement(
                            name="volume_cm3",
                            value=round(volume_mm3 / 1000.0, 2),
                            unit="cm3",
                        )
                    )

    result.measurements = measurements
    result.status = CheckStatus.PASS
    return result


def _median_pixel_footprint_mm2(depth: DepthData, mask: np.ndarray) -> float | None:
    """Median real-world area covered by one pixel on the product surface,
    estimated from horizontal neighbor distances in the 3D point grid."""
    shifted = np.zeros_like(mask)
    shifted[:, :-1] = mask[:, 1:]
    pairs = mask & shifted
    ys, xs = np.nonzero(pairs)
    if len(xs) < 50:
        return None
    p0 = depth.points_3d[ys, xs]
    p1 = depth.points_3d[ys, xs + 1]
    dists = np.linalg.norm((p1 - p0)[:, :2], axis=1)
    dists = dists[np.isfinite(dists) & (dists > 0) & (dists < 50)]
    if dists.size < 50:
        return None
    step = float(np.median(dists))
    return step * step
