"""Live-preview endpoints consumed by the Streamlit frontend."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from typing import Optional

from backend.image_codec import encode_image_b64
from backend.services.continuous_inspection import continuous_inspector
from backend.services.frame_pairing import PairingError, get_latest_pair
from backend.services.frame_store import frame_store
from backend.services.result_store import result_store

import cv2

router = APIRouter()


# ---- continuous streaming mode ----


@router.post("/stream/start")
def start_stream(
    interval_sec: Optional[float] = None,
    ocr_enabled: Optional[bool] = None,
):
    """Start continuous inspection (or update its settings while running)."""
    continuous_inspector.start(interval_sec, ocr_enabled)
    return continuous_inspector.status()


@router.post("/stream/stop")
def stop_stream():
    continuous_inspector.stop()
    return continuous_inspector.status()


@router.get("/stream/status")
def stream_status():
    return continuous_inspector.status()


@router.get("/devices")
def list_devices():
    return {"devices": frame_store.devices()}


@router.get("/frame/{device_id}")
def get_frame(device_id: str):
    stored = frame_store.get(device_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"no frame from '{device_id}'")
    ok, buf = cv2.imencode(".jpg", stored.image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(status_code=500, detail="encode failed")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@router.get("/stream/processed")
def processed_stream():
    """Return ONLY the two processed (annotated) frames from the latest
    inspection, plus a minimal status — this is what the dashboard shows.

    Nothing raw and no heavy report payload is sent to the frontend.
    """
    result = result_store.latest()
    if result is None:
        raise HTTPException(status_code=404, detail="no inspection yet")

    images = result.get("images", {})
    quality = result.get("quality") or {}
    return {
        "inspection_id": result.get("inspection_id"),
        "created_at": result.get("created_at"),
        "overall_pass": quality.get("overall_pass"),
        "failure_reasons": quality.get("failure_reasons", []),
        "total_time_ms": result.get("total_time_ms", 0),
        "vertical_device_id": result.get("vertical_device_id"),
        "horizontal_device_id": result.get("horizontal_device_id"),
        "vertical_fps": frame_store.get_fps(result.get("vertical_device_id") or ""),
        "horizontal_fps": frame_store.get_fps(result.get("horizontal_device_id") or ""),
        "annotated_vertical_b64": images.get("annotated_vertical"),
        "annotated_horizontal_b64": images.get("annotated_horizontal"),
    }


@router.get("/latest_pair")
def latest_pair():
    try:
        pair = get_latest_pair()
    except PairingError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return {
        "vertical_device_id": pair.vertical_device_id,
        "horizontal_device_id": pair.horizontal_device_id,
        "time_delta_ms": pair.time_delta_ms,
        "synchronized": pair.synchronized,
        "vertical_fps": frame_store.get_fps(pair.vertical_device_id),
        "horizontal_fps": frame_store.get_fps(pair.horizontal_device_id),
        "vertical_image_b64": encode_image_b64(pair.vertical_image),
        "horizontal_image_b64": encode_image_b64(pair.horizontal_image),
    }
