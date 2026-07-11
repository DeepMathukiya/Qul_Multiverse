"""End-to-end inspection pipeline — the single entry point of the
datascience layer.

    run_inspection(vertical, horizontal) -> InspectionResult

Stages:
1. preprocessing (undistort -> rectify when calibrated, denoise, ROI crop)
2. OCR (Qwen3-VL via GenieX) + local QR decode, and pothole detection
   (Qwen3-VL via GenieX) — run sequentially in one worker thread since both
   share a single local NPU-hosted model (slowest stage)
3. 2D dimensions (classical CV in the known ROI)
4. stereo disparity -> depth -> 3D measurements (calibration required)
5. crack/scratch (classical CV) + dent (stereo depth)
6. quality rules -> explainable PASS/FAIL

CLI smoke test:
    python -m datascience.inspection_pipeline --vertical bottle-dent.png --horizontal bottle-dent.png
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np

from datascience.image_codec import encode_image_b64
from datascience.schemas import (
    CheckStatus,
    Dim3DResult,
    InspectionResult,
    OcrResult,
    PotholeResult,
    SurfaceDefectResult,
)
from datascience.timing import StageTimer

from datascience.dimensions_2d.boundary_extraction import extract_product_boundary
from datascience.dimensions_2d.measurements_2d import measure_product
from datascience.dimensions_3d.depth_reconstruction import (
    colorize_depth,
    reconstruct_depth,
)
from datascience.dimensions_3d.disparity import compute_disparity
from datascience.dimensions_3d.measurements_3d import measure_product_3d
from datascience.ocr.ocr_validation import inspect_product_info
from datascience.overlays.drawing_overlays import (
    GREEN,
    RED,
    YELLOW,
    draw_dimensions,
    draw_linear_defects,
    draw_mask_overlay,
    draw_pothole_detections,
    draw_roi,
)
from datascience.pothole.pothole_metrics import analyze_potholes
from datascience.preprocessing.distortion_correction import undistort
from datascience.preprocessing.image_preprocessing import (
    preprocess_frame,
    preprocess_gray,
)
from datascience.preprocessing.roi_extraction import extract_roi
from datascience.preprocessing.stereo_rectification import (
    load_stereo_calibration,
    rectify_pair,
)
from datascience.quality.decision_report import build_decision
from datascience.quality.quality_rules import evaluate_quality_rules
from datascience.surface_defects.crack_detection import detect_cracks
from datascience.surface_defects.dent_detection import detect_dents
from datascience.surface_defects.scratch_detection import detect_scratches


def run_inspection(
    vertical_raw: np.ndarray,
    horizontal_raw: np.ndarray,
    vertical_device_id: str | None = None,
    horizontal_device_id: str | None = None,
    ocr_enabled: bool | None = None,
) -> InspectionResult:
    timer = StageTimer()
    result = InspectionResult(
        inspection_id=uuid.uuid4().hex[:12],
        created_at=datetime.now(timezone.utc).isoformat(),
        vertical_device_id=vertical_device_id,
        horizontal_device_id=horizontal_device_id,
    )

    # ---------- 1. preprocessing ----------
    with timer.stage("preprocessing"):
        vertical_img = preprocess_frame(undistort(vertical_raw, "vertical"))
        horizontal_img = preprocess_frame(undistort(horizontal_raw, "horizontal"))

        stereo_calib = load_stereo_calibration()
        if stereo_calib is not None:
            vertical_rect, horizontal_rect = rectify_pair(vertical_img, horizontal_img, stereo_calib)
        else:
            vertical_rect, horizontal_rect = vertical_img, horizontal_img

        roi_bgr, roi_rect = extract_roi(vertical_rect)
        roi_gray = preprocess_gray(roi_bgr)

    result.images["vertical"] = encode_image_b64(vertical_rect)
    result.images["horizontal"] = encode_image_b64(horizontal_rect)
    result.images["roi"] = encode_image_b64(draw_roi(vertical_rect, roi_rect))

    # ---------- 2. OCR + pothole (Qwen3-VL via GenieX) in a worker thread --
    # Both calls hit the same local NPU-hosted model, so they run
    # sequentially inside one background thread (avoids two concurrent
    # generation requests against one local server) while still overlapping
    # with the classical CV stages below running on the main thread.
    ocr_holder: dict[str, OcrResult] = {}
    pothole_holder: dict[str, object] = {}

    def _vlm_worker() -> None:
        t0 = __import__("time").perf_counter()
        ocr_holder["result"] = inspect_product_info(vertical_rect, horizontal_rect, ocr_enabled)
        t1 = __import__("time").perf_counter()
        ocr_holder["ms"] = (t1 - t0) * 1000.0

        pothole_result, pothole_masks = analyze_potholes(vertical_rect, horizontal_rect)
        pothole_holder["result"] = pothole_result
        pothole_holder["masks"] = pothole_masks
        pothole_holder["ms"] = (__import__("time").perf_counter() - t1) * 1000.0

    vlm_thread = threading.Thread(target=_vlm_worker, daemon=True)
    vlm_thread.start()

    # ---------- 3. 2D dimensions ----------
    with timer.stage("dimensions_2d"):
        boundary = extract_product_boundary(roi_bgr)
        result.dims_2d = measure_product(boundary)
        if boundary is not None:
            result.images["boundary"] = encode_image_b64(
                draw_dimensions(roi_bgr, boundary, result.dims_2d.mm_per_px)
            )

    # ---------- 4. stereo 3D ----------
    depth_data = None
    with timer.stage("dimensions_3d"):
        if stereo_calib is None:
            result.dims_3d = Dim3DResult(
                status=CheckStatus.NOT_AVAILABLE,
                calibrated=False,
                note=(
                    "stereo calibration missing — run "
                    "datascience/calibration/stereo_calibration.py"
                ),
            )
        else:
            try:
                disparity = compute_disparity(vertical_rect, horizontal_rect)
                depth_data = reconstruct_depth(disparity, stereo_calib["Q"])
                result.images["depth_map"] = encode_image_b64(
                    colorize_depth(depth_data.depth_map)
                )

                # Product mask in full-frame coordinates for 3D measurement.
                full_mask = np.zeros(vertical_rect.shape[:2], dtype=np.uint8)
                if boundary is not None:
                    x, y, w, h = roi_rect
                    full_mask[y : y + h, x : x + w] = boundary.mask
                else:
                    full_mask[:] = 255  # fall back to whole frame
                result.dims_3d = measure_product_3d(depth_data, full_mask)
            except Exception as exc:
                result.dims_3d = Dim3DResult(
                    status=CheckStatus.NOT_AVAILABLE,
                    calibrated=True,
                    error=f"stereo processing failed: {exc}",
                )

    # ---------- 5. surface defects ----------
    with timer.stage("surface_defects"):
        defects = SurfaceDefectResult()
        try:
            product_mask = boundary.mask if boundary is not None else None

            crack_metrics, crack_components = detect_cracks(roi_gray, product_mask)
            defects.cracks = crack_metrics

            crack_mask = np.zeros_like(roi_gray)
            for comp in crack_components:
                crack_mask = cv2.bitwise_or(crack_mask, comp.mask)

            scratch_metrics, scratch_components = detect_scratches(
                roi_gray, product_mask, exclude_mask=crack_mask
            )
            defects.scratches = scratch_metrics

            if crack_components:
                result.images["crack"] = encode_image_b64(
                    draw_linear_defects(roi_bgr, crack_components, RED, "crack ")
                )
            if scratch_components:
                result.images["scratch"] = encode_image_b64(
                    draw_linear_defects(roi_bgr, scratch_components, YELLOW, "scratch ")
                )

            if depth_data is not None:
                x, y, w, h = roi_rect
                full_mask = np.zeros(vertical_rect.shape[:2], dtype=np.uint8)
                if boundary is not None:
                    full_mask[y : y + h, x : x + w] = boundary.mask
                else:
                    full_mask[y : y + h, x : x + w] = 255
                dent_metrics, dent_mask = detect_dents(depth_data, full_mask)
                defects.dents = dent_metrics
                if dent_mask is not None:
                    result.images["dent"] = encode_image_b64(
                        draw_mask_overlay(vertical_rect, dent_mask, GREEN)
                    )
            else:
                defects.dent_note = "dent depth needs stereo calibration"

            defects.status = CheckStatus.PASS
        except Exception as exc:
            defects.status = CheckStatus.NOT_AVAILABLE
            defects.error = f"surface defect analysis failed: {exc}"
        result.surface_defects = defects

    # ---------- wait for OCR + pothole (Qwen3-VL) ----------
    vlm_thread.join()
    result.ocr = ocr_holder.get("result", OcrResult())
    timer.record("ocr", float(ocr_holder.get("ms", 0.0)))

    result.pothole = pothole_holder.get("result", PotholeResult())
    pothole_masks = pothole_holder.get("masks", [])
    timer.record("pothole", float(pothole_holder.get("ms", 0.0)))
    if pothole_masks:
        result.images["pothole"] = encode_image_b64(
            draw_pothole_detections(vertical_rect, pothole_masks, result.pothole.detections)
        )

    # ---------- 6. quality rules -> decision ----------
    with timer.stage("quality_rules"):
        checks = evaluate_quality_rules(
            result.ocr,
            result.dims_2d,
            result.dims_3d,
            result.surface_defects,
            result.pothole,
        )
        result.quality = build_decision(checks)

    result.timings_ms = timer.timings_ms
    result.total_time_ms = timer.total_ms
    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run one inspection from image files")
    parser.add_argument("--vertical", required=True)
    parser.add_argument("--horizontal", required=True)
    parser.add_argument("--save-overlays", action="store_true",
                        help="write overlay images next to the vertical input")
    args = parser.parse_args()

    vertical_cli_img = cv2.imread(args.vertical)
    horizontal_cli_img = cv2.imread(args.horizontal)
    if vertical_cli_img is None or horizontal_cli_img is None:
        raise SystemExit("could not read input images")

    inspection = run_inspection(vertical_cli_img, horizontal_cli_img, "cli-vertical", "cli-horizontal")

    printable = inspection.model_dump()
    printable["images"] = {k: f"<{len(v)} b64 chars>" for k, v in printable["images"].items()}
    print(json.dumps(printable, indent=2))

    if args.save_overlays:
        from pathlib import Path

        from datascience.image_codec import decode_image_b64

        out_dir = Path(args.vertical).parent
        for name, b64 in inspection.images.items():
            path = out_dir / f"overlay_{name}.jpg"
            cv2.imwrite(str(path), decode_image_b64(b64))
            print(f"saved {path}")
