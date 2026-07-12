"""Inspection orchestration: trigger a run and fetch results."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.image_codec import decode_image_bytes
from backend.services.datascience_client import (
    DatascienceUnavailable,
    run_inspection_remote,
)
from backend.services.frame_pairing import PairingError, get_latest_pair
from backend.services.result_store import result_store

router = APIRouter()


@router.post("/inspect")
async def inspect(
    vertical: Optional[UploadFile] = File(None),
    horizontal: Optional[UploadFile] = File(None),
    ocr: Optional[bool] = None,
    area_tolerance_ratio: Optional[float] = None,
):
    """Run one inspection.

    - With `vertical` and `horizontal` files: inspect the uploaded pair
      (test mode).
    - Without files: inspect the latest live pair from the two phones.
    - `ocr` query param: true/false overrides the OCR setting for this run.
    - `area_tolerance_ratio`: overrides product_specs.yaml's allowed area
      shortfall (e.g. the dashboard's tolerance slider), 0.02 = 2%.
    """
    is_upload = vertical is not None and horizontal is not None
    if is_upload:
        try:
            vertical_image = decode_image_bytes(await vertical.read())
            horizontal_image = decode_image_bytes(await horizontal.read())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        vertical_id, horizontal_id = "upload-vertical", "upload-horizontal"
    elif vertical is None and horizontal is None:
        try:
            pair = get_latest_pair()
        except PairingError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        vertical_image, horizontal_image = pair.vertical_image, pair.horizontal_image
        vertical_id, horizontal_id = pair.vertical_device_id, pair.horizontal_device_id
    else:
        raise HTTPException(
            status_code=400,
            detail="provide both 'vertical' and 'horizontal' files, or neither for live mode",
        )

    try:
        result = run_inspection_remote(
            vertical_image, horizontal_image, vertical_id, horizontal_id,
            ocr_enabled=ocr, is_upload=is_upload,
            area_tolerance_ratio=area_tolerance_ratio,
        )
    except DatascienceUnavailable as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    result_store.add(result)
    return result


@router.get("/inspections/latest")
def latest_inspection():
    result = result_store.latest()
    if result is None:
        raise HTTPException(status_code=404, detail="no inspections yet")
    return result


@router.get("/inspections/{inspection_id}")
def get_inspection(inspection_id: str):
    result = result_store.get(inspection_id)
    if result is None:
        raise HTTPException(status_code=404, detail="unknown inspection id")
    return result


@router.get("/inspections")
def list_inspections():
    return {"ids": result_store.list_ids()}
