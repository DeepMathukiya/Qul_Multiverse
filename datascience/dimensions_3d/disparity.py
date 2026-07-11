"""Disparity computation from a rectified stereo pair (StereoSGBM + WLS)."""

from __future__ import annotations

import cv2
import numpy as np

from datascience.config_loader import load_system_config


def compute_disparity(left_rect: np.ndarray, right_rect: np.ndarray) -> np.ndarray:
    """Return a float32 disparity map (pixels); invalid pixels <= 0."""
    cfg = load_system_config().get("stereo", {})
    num_disp = int(cfg.get("num_disparities", 128))
    num_disp -= num_disp % 16  # SGBM requires a multiple of 16
    block = int(cfg.get("block_size", 5)) | 1  # odd

    gray_l = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block,
        P1=8 * 3 * block * block,
        P2=32 * 3 * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disp_left = left_matcher.compute(gray_l, gray_r)

    if cfg.get("use_wls_filter", True) and hasattr(cv2, "ximgproc"):
        # Weighted-least-squares filtering fills holes and sharpens edges
        # using the right-view disparity as consistency check.
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        disp_right = right_matcher.compute(gray_r, gray_l)

        wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls.setLambda(8000.0)
        wls.setSigmaColor(1.5)
        disp_left = wls.filter(disp_left, gray_l, disparity_map_right=disp_right)

    # SGBM output is fixed-point with 4 fractional bits.
    return disp_left.astype(np.float32) / 16.0
