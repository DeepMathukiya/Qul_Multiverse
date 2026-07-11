"""POST /upload — the phone app posts JPEG frames here.

Same contract as the original example_receiver.py Flask endpoint:
multipart form with fields `device_id`, `timestamp` and file `frame`.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.image_codec import decode_image_bytes
from backend.services.frame_store import frame_store

router = APIRouter()


@router.post("/upload")
async def upload_frame(
    frame: UploadFile = File(...),
    device_id: str = Form("unknown"),
    timestamp: str = Form(""),
):
    raw = await frame.read()

    try:
        image = decode_image_bytes(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail="could not decode frame")

    try:
        device_ts_ms = float(timestamp)
    except (TypeError, ValueError):
        device_ts_ms = 0.0

    frame_store.update(device_id, image, device_ts_ms)

    return {"status": "ok", "device_id": device_id}
