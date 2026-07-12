"""End-to-end inspection pipeline — the single entry point of the
datascience layer.

    run_inspection(vertical, horizontal) -> InspectionResult

Stages:
1. preprocessing (undistort -> rectify when calibrated, denoise, ROI crop)
2. OCR (Sarvam, runs in a worker thread — slowest stage)
3. 2D dimensions (classical CV in the known ROI)
4. stereo disparity -> depth -> 3D measurements (calibration required)
5. surface defects (YOLO detection: crack/dent/missing-head/paint-off/scratch)
6. YOLO pothole segmentation
7. quality rules -> explainable PASS/FAIL

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
    Dim2DResult,
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
    compose_annotated_frame,
    draw_defect_boxes,
    draw_dimensions,
    draw_pothole_detections,
    draw_roi,
)
from datascience.pothole.pothole_metrics import analyze_potholes
from datascience.preprocessing.distortion_correction import undistort
from datascience.preprocessing.image_preprocessing import preprocess_frame
from datascience.preprocessing.roi_extraction import extract_roi
from datascience.preprocessing.stereo_rectification import (
    load_stereo_calibration,
    rectify_pair,
)
from datascience.quality.decision_report import build_decision
from datascience.quality.quality_rules import evaluate_quality_rules
from datascience.quality.reference_area_check import capture_reference_area
from datascience.surface_defects.defect_metrics import analyze_surface_defects


def run_inspection(
    vertical_raw: np.ndarray,
    horizontal_raw: np.ndarray,
    vertical_device_id: str | None = None,
    horizontal_device_id: str | None = None,
    ocr_enabled: bool | None = None,
    is_upload: bool = False,
    area_tolerance_ratio: float | None = None,
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

    result.images["vertical"] = encode_image_b64(vertical_rect)
    result.images["horizontal"] = encode_image_b64(horizontal_rect)
    result.images["roi"] = encode_image_b64(draw_roi(vertical_rect, roi_rect))

    # ---------- 2. OCR in a worker thread (network-bound, slowest) ----------
    ocr_holder: dict[str, OcrResult] = {}

    def _ocr_worker() -> None:
        t0 = __import__("time").perf_counter()
        ocr_holder["result"] = inspect_product_info(vertical_rect, ocr_enabled)
        ocr_holder["ms"] = ( __import__("time").perf_counter() - t0) * 1000.0

    ocr_thread = threading.Thread(target=_ocr_worker, daemon=True)
    ocr_thread.start()

    # ---------- 3. 2D dimensions ----------
    with timer.stage("dimensions_2d"):
        boundary = extract_product_boundary(roi_bgr)
        result.dims_2d = measure_product(boundary)
        if boundary is not None:
            result.images["boundary"] = encode_image_b64(
                draw_dimensions(roi_bgr, boundary, result.dims_2d.mm_per_px)
            )
        if is_upload:
            capture_reference_area(result.dims_2d)

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

    # ---------- 5. surface defects (YOLO detection, all classes) ----------
    defect_detections: list = []
    with timer.stage("surface_defects"):
        try:
            result.surface_defects, defect_detections = analyze_surface_defects(
                vertical_rect
            )
            if defect_detections:
                result.images["defects"] = encode_image_b64(
                    draw_defect_boxes(vertical_rect, defect_detections)
                )
        except Exception as exc:
            result.surface_defects = SurfaceDefectResult(
                status=CheckStatus.NOT_AVAILABLE,
                error=f"surface defect analysis failed: {exc}",
            )

    # ---------- 6. pothole (YOLO segmentation only) ----------
    with timer.stage("pothole"):
        pothole_result, pothole_masks = analyze_potholes(vertical_rect)
        result.pothole = pothole_result
        if pothole_masks:
            result.images["pothole"] = encode_image_b64(
                draw_pothole_detections(vertical_rect, pothole_masks, pothole_result.detections)
            )

    # ---------- wait for OCR ----------
    ocr_thread.join()
    result.ocr = ocr_holder.get("result", OcrResult())
    timer.record("ocr", float(ocr_holder.get("ms", 0.0)))

    # ---------- 7. quality rules -> decision ----------
    with timer.stage("quality_rules"):
        checks = evaluate_quality_rules(
            result.ocr,
            result.dims_2d,
            result.dims_3d,
            result.surface_defects,
            result.pothole,
            area_tolerance_ratio=area_tolerance_ratio,
        )
        result.quality = build_decision(checks)

    # ---------- 8. annotated stream frames (BOTH cameras) ----------
    # Each camera's frame gets all findings burned in, so the dashboard shows
    # the two processed live streams and nothing else. The vertical frame
    # carries the primary detections; the horizontal frame is detected
    # independently for display (both are "processed streams").
    with timer.stage("annotate"):
        annotated_vertical = compose_annotated_frame(
            vertical_rect,
            roi_rect,
            boundary,
            result.dims_2d.mm_per_px,
            defect_detections,
            pothole_masks,
            result.pothole.detections,
            result,
        )
        result.images["annotated_vertical"] = encode_image_b64(annotated_vertical)

        annotated_horizontal = _annotate_secondary_frame(horizontal_rect, result)
        result.images["annotated_horizontal"] = encode_image_b64(annotated_horizontal)

    result.timings_ms = timer.timings_ms
    result.total_time_ms = timer.total_ms
    return result


def _annotate_secondary_frame(frame_rect: np.ndarray, result: InspectionResult) -> np.ndarray:
    """Run independent 2D boundary + YOLO defect detection on the secondary
    (horizontal) camera frame and burn the overlays + shared PASS/FAIL banner
    onto it, so it is displayed as a processed stream too."""
    roi_bgr, roi_rect = extract_roi(frame_rect)

    boundary = extract_product_boundary(roi_bgr)
    dim2d = measure_product(boundary)

    _, defect_detections = analyze_surface_defects(frame_rect)

    return compose_annotated_frame(
        frame_rect,
        roi_rect,
        boundary,
        dim2d.mm_per_px,
        defect_detections,
        [],   # pothole overlay stays on the primary frame
        [],
        result,
    )


def run_single_frame_inspection(
    frame_raw: np.ndarray,
    device_id: str | None = None,
    ocr_enabled: bool | None = None,
) -> InspectionResult:
    """Single-camera fallback — only one phone is streaming, so there is no
    stereo pair to measure area (2D) or volume (3D) from. Dimensions, OCR and
    pothole are all skipped/NOT_AVAILABLE (they either need the pair or are
    out of scope here), and the PASS/FAIL judgment rests solely on YOLO
    surface-defect (crack/dent/...) detection on this one frame.
    """
    timer = StageTimer()
    result = InspectionResult(
        inspection_id=uuid.uuid4().hex[:12],
        created_at=datetime.now(timezone.utc).isoformat(),
        horizontal_device_id=device_id,
    )

    with timer.stage("preprocessing"):
        frame_rect = preprocess_frame(undistort(frame_raw, "horizontal"))

    result.images["horizontal"] = encode_image_b64(frame_rect)

    result.dims_2d = Dim2DResult(
        status=CheckStatus.NOT_AVAILABLE,
        error="dimensions skipped — single-camera stream, no stereo pair",
    )
    result.dims_3d = Dim3DResult(
        status=CheckStatus.NOT_AVAILABLE,
        calibrated=False,
        note="volume skipped — single-camera stream, no stereo pair",
    )
    result.ocr = OcrResult(
        status=CheckStatus.SKIPPED,
        error="OCR skipped — single-camera stream (defect-only judgment)",
    )
    result.pothole = PotholeResult(status=CheckStatus.SKIPPED, enabled=False)

    defect_detections: list = []
    with timer.stage("surface_defects"):
        try:
            result.surface_defects, defect_detections = analyze_surface_defects(frame_rect)
            if defect_detections:
                result.images["defects"] = encode_image_b64(
                    draw_defect_boxes(frame_rect, defect_detections)
                )
        except Exception as exc:
            result.surface_defects = SurfaceDefectResult(
                status=CheckStatus.NOT_AVAILABLE,
                error=f"surface defect analysis failed: {exc}",
            )

    with timer.stage("quality_rules"):
        checks = evaluate_quality_rules(
            result.ocr,
            result.dims_2d,
            result.dims_3d,
            result.surface_defects,
            result.pothole,
        )
        result.quality = build_decision(checks)

    with timer.stage("annotate"):
        annotated = compose_annotated_frame(
            frame_rect, None, None, None, defect_detections, [], [], result,
        )
        result.images["annotated_horizontal"] = encode_image_b64(annotated)

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
