"""Single-camera chessboard calibration.

Capture 15-25 chessboard photos with one phone, put them in a folder, then:

    python -m datascience.calibration.camera_calibration --images calib/vertical --camera vertical
    python -m datascience.calibration.camera_calibration --images calib/horizontal --camera horizontal

Writes `<calibration_dir>/<camera>_cam.npz` with K (intrinsics) and dist
(distortion coefficients).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from datascience.config_loader import calibration_dir

DEFAULT_BOARD = (9, 6)  # inner corners (cols, rows)
DEFAULT_SQUARE_MM = 24.0


def find_chessboard_corners(
    image_paths: list[Path],
    board_size: tuple[int, int],
    square_mm: float,
) -> tuple[list, list, tuple[int, int]]:
    """Detect chessboard corners in every image; returns object/image points."""
    # 3D coordinates of the board corners in board-local space (Z = 0).
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
    objp *= square_mm

    obj_points, img_points = [], []
    image_size = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"[calib] skip unreadable {path}")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(gray, board_size, None)
        if not found:
            print(f"[calib] no chessboard in {path.name}")
            continue

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners)
        print(f"[calib] ok {path.name}")

    return obj_points, img_points, image_size


def calibrate_camera(
    images_dir: str,
    camera_name: str,
    board_size: tuple[int, int] = DEFAULT_BOARD,
    square_mm: float = DEFAULT_SQUARE_MM,
) -> float:
    paths = sorted(
        p for p in Path(images_dir).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not paths:
        raise SystemExit(f"no images found in {images_dir}")

    obj_points, img_points, image_size = find_chessboard_corners(
        paths, board_size, square_mm
    )
    if len(obj_points) < 5:
        raise SystemExit(
            f"only {len(obj_points)} usable chessboard views — need at least 5"
        )

    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    out = calibration_dir() / f"{camera_name}_cam.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out), K=K, dist=dist, image_size=np.array(image_size))
    print(f"[calib] RMS reprojection error: {rms:.3f} px")
    print(f"[calib] saved {out}")
    return rms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="folder of chessboard photos")
    parser.add_argument("--camera", required=True, choices=["vertical", "horizontal"])
    parser.add_argument("--cols", type=int, default=DEFAULT_BOARD[0])
    parser.add_argument("--rows", type=int, default=DEFAULT_BOARD[1])
    parser.add_argument("--square-mm", type=float, default=DEFAULT_SQUARE_MM)
    args = parser.parse_args()

    calibrate_camera(
        args.images, args.camera, (args.cols, args.rows), args.square_mm
    )
