"""POST /process — run the full inspection pipeline on a stereo pair."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from datascience.image_codec import decode_image_b64
from datascience.schemas import ProcessRequest

from datascience.inspection_pipeline import run_inspection

router = APIRouter()


@router.post("/process")
def process(request: ProcessRequest):
    try:
        vertical = decode_image_b64(request.vertical_image_b64)
        horizontal = decode_image_b64(request.horizontal_image_b64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    result = run_inspection(
        vertical,
        horizontal,
        request.vertical_device_id,
        request.horizontal_device_id,
        ocr_enabled=request.ocr_enabled,
    )
    return result.model_dump()
