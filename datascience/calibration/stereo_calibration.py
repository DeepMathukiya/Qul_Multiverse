"""Stereo calibration of the two-phone rig.

Prerequisite: both cameras individually calibrated (camera_calibration.py).
Capture synchronized chessboard pairs (same instant, both phones), name them
identically in two folders, then:

    python -m datascience.calibration.stereo_calibration --vertical calib/stereo/vertical --horizontal calib/stereo/horizontal

Writes `<calibration_dir>/stereo.npz` containing rectification maps
(map1x/y, map2x/y), projection matrices and the Q reprojection matrix used
for metric depth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from datascience.config_loader import calibration_dir

from datascience.calibration.camera_calibration import (
    DEFAULT_BOARD,
    DEFAULT_SQUARE_MM,
    find_chessboard_corners,
)
from datascience.preprocessing.distortion_correction import load_intrinsics


def calibrate_stereo(
    left_dir: str,
    right_dir: str,
    board_size: tuple[int, int] = DEFAULT_BOARD,
    square_mm: float = DEFAULT_SQUARE_MM,
) -> float:
    left_paths = sorted(
        p for p in Path(left_dir).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    right_paths = sorted(
        p for p in Path(right_dir).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    # Keep only pairs present in both folders (matched by file name).
    left_by_name = {p.name: p for p in left_paths}
    right_by_name = {p.name: p for p in right_paths}
    common_names = sorted(set(left_by_name) & set(right_by_name))
    if len(common_names) < 5:
        raise SystemExit("need at least 5 synchronized chessboard pairs")

    obj_l, img_l, size_l = find_chessboard_corners(
        [left_by_name[n] for n in common_names], board_size, square_mm
    )
    obj_r, img_r, _ = find_chessboard_corners(
        [right_by_name[n] for n in common_names], board_size, square_mm
    )
    n = min(len(img_l), len(img_r))
    if n < 5:
        raise SystemExit(f"only {n} usable pairs after corner detection")

    left_intr = load_intrinsics("vertical")
    right_intr = load_intrinsics("horizontal")
    if left_intr is None or right_intr is None:
        raise SystemExit("run camera_calibration.py for both cameras (vertical + horizontal) first")

    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    rms, K1, d1, K2, d2, R, T, _, _ = cv2.stereoCalibrate(
        obj_l[:n], img_l[:n], img_r[:n],
        left_intr["K"], left_intr["dist"],
        right_intr["K"], right_intr["dist"],
        size_l, criteria=criteria, flags=flags,
    )
    print(f"[stereo] RMS reprojection error: {rms:.3f} px")
    print(f"[stereo] baseline: {np.linalg.norm(T):.1f} mm")

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, d1, K2, d2, size_l, R, T, alpha=0
    )

    map1x, map1y = cv2.initUndistortRectifyMap(K1, d1, R1, P1, size_l, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, d2, R2, P2, size_l, cv2.CV_32FC1)

    out = calibration_dir() / "stereo.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out),
        map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y,
        R=R, T=T, Q=Q, P1=P1, P2=P2,
        image_size=np.array(size_l),
    )
    print(f"[stereo] saved {out}")
    return rms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertical", required=True, help="folder of vertical-camera chessboard photos")
    parser.add_argument("--horizontal", required=True, help="folder of horizontal-camera chessboard photos")
    parser.add_argument("--cols", type=int, default=DEFAULT_BOARD[0])
    parser.add_argument("--rows", type=int, default=DEFAULT_BOARD[1])
    parser.add_argument("--square-mm", type=float, default=DEFAULT_SQUARE_MM)
    args = parser.parse_args()

    calibrate_stereo(args.vertical, args.horizontal, (args.cols, args.rows), args.square_mm)
