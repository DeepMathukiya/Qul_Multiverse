"""Apply precomputed stereo rectification maps to a stereo pair.

Stereo channel convention: the vertical camera feeds the left channel,
the horizontal camera the right channel."""

from __future__ import annotations

import cv2
import numpy as np

from datascience.config_loader import calibration_dir


def load_stereo_calibration() -> dict | None:
    """Load `<calibration_dir>/stereo.npz` written by stereo_calibration.py.

    Contains rectification maps for both cameras plus the Q reprojection
    matrix needed to turn disparity into metric depth.
    """
    path = calibration_dir() / "stereo.npz"
    if not path.exists():
        return None
    data = np.load(str(path))
    required = ["map1x", "map1y", "map2x", "map2y", "Q", "image_size"]
    if any(key not in data for key in required):
        return None
    return {key: data[key] for key in data.files}


def rectify_pair(
    left: np.ndarray,
    right: np.ndarray,
    stereo_calib: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Remap both images into the rectified (row-aligned) stereo geometry."""
    calib_w, calib_h = (int(v) for v in stereo_calib["image_size"])

    # Maps are only valid at the calibration resolution.
    if (left.shape[1], left.shape[0]) != (calib_w, calib_h):
        left = cv2.resize(left, (calib_w, calib_h), interpolation=cv2.INTER_AREA)
    if (right.shape[1], right.shape[0]) != (calib_w, calib_h):
        right = cv2.resize(right, (calib_w, calib_h), interpolation=cv2.INTER_AREA)

    left_rect = cv2.remap(
        left, stereo_calib["map1x"], stereo_calib["map1y"], cv2.INTER_LINEAR
    )
    right_rect = cv2.remap(
        right, stereo_calib["map2x"], stereo_calib["map2y"], cv2.INTER_LINEAR
    )
    return left_rect, right_rect
