"""POST /process — run the full inspection pipeline on a stereo pair."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from datascience.image_codec import decode_image_b64
from datascience.schemas import ProcessRequest

from datascience.inspection_pipeline import run_inspection, run_single_frame_inspection

router = APIRouter()


@router.post("/process")
def process(request: ProcessRequest):
    try:
        if request.vertical_image_b64 and request.horizontal_image_b64:
            vertical = decode_image_b64(request.vertical_image_b64)
            horizontal = decode_image_b64(request.horizontal_image_b64)
            result = run_inspection(
                vertical,
                horizontal,
                request.vertical_device_id,
                request.horizontal_device_id,
                ocr_enabled=request.ocr_enabled,
                is_upload=request.is_upload,
                area_tolerance_ratio=request.area_tolerance_ratio,
            )
        elif request.vertical_image_b64 or request.horizontal_image_b64:
            # Only one camera available — no stereo pair to measure
            # area/volume from. Fall back to defect-only judgment.
            frame = decode_image_b64(
                request.horizontal_image_b64 or request.vertical_image_b64
            )
            device_id = request.horizontal_device_id or request.vertical_device_id
            result = run_single_frame_inspection(
                frame, device_id, ocr_enabled=request.ocr_enabled
            )
        else:
            raise HTTPException(status_code=400, detail="at least one image is required")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return result.model_dump()
