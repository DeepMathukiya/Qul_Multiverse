"""Turn disparity into metric 3D points using the calibration Q matrix."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DepthData:
    points_3d: np.ndarray   # HxWx3 float32, mm (X, Y, Z in left-camera frame)
    depth_map: np.ndarray   # HxW float32 Z in mm; NaN where invalid
    valid_mask: np.ndarray  # HxW bool


def reconstruct_depth(disparity: np.ndarray, Q: np.ndarray) -> DepthData:
    """Reproject disparity to 3D. Units follow the calibration (mm if the
    chessboard square size was given in mm)."""
    points_3d = cv2.reprojectImageTo3D(disparity, Q.astype(np.float64))

    depth = points_3d[:, :, 2].astype(np.float32)

    valid = (
        (disparity > 0.5)
        & np.isfinite(depth)
        & (depth > 0)
        & (depth < 10_000)  # reject reprojection artifacts beyond 10 m
    )
    depth = depth.copy()
    depth[~valid] = np.nan

    return DepthData(
        points_3d=points_3d.astype(np.float32),
        depth_map=depth,
        valid_mask=valid,
    )


def colorize_depth(depth_map: np.ndarray) -> np.ndarray:
    """Render the depth map as a color image for the dashboard."""
    valid = np.isfinite(depth_map)
    if not valid.any():
        return np.zeros((*depth_map.shape, 3), dtype=np.uint8)

    lo, hi = np.nanpercentile(depth_map, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((depth_map - lo) / (hi - lo), 0, 1)
    norm = np.nan_to_num(norm, nan=0.0)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[~valid] = (0, 0, 0)
    return colored
