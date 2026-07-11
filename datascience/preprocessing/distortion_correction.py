"""Lens distortion correction using saved single-camera intrinsics."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from datascience.config_loader import calibration_dir


def load_intrinsics(camera_name: str) -> dict | None:
    """Load `<calibration_dir>/<camera_name>_cam.npz` (K, dist) if present."""
    path = calibration_dir() / f"{camera_name}_cam.npz"
    if not path.exists():
        return None
    data = np.load(str(path))
    return {"K": data["K"], "dist": data["dist"]}


def undistort(image: np.ndarray, camera_name: str) -> np.ndarray:
    """Undistort using saved intrinsics; returns the input unchanged if the
    camera has not been calibrated yet."""
    intrinsics = load_intrinsics(camera_name)
    if intrinsics is None:
        return image
    return cv2.undistort(image, intrinsics["K"], intrinsics["dist"])
