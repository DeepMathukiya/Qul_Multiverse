"""ndarray <-> base64 helpers used for JSON transport between services."""

from __future__ import annotations

import base64

import cv2
import numpy as np


def encode_image_b64(image: np.ndarray, ext: str = ".jpg", quality: int = 90) -> str:
    """Encode a BGR image to a base64 string (JPEG by default, PNG for masks)."""
    params = []
    if ext == ".jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise ValueError(f"could not encode image as {ext}")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_image_b64(data: str) -> np.ndarray:
    """Decode a base64 string back to a BGR image."""
    raw = base64.b64decode(data)
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode base64 image")
    return image


def decode_image_bytes(raw: bytes) -> np.ndarray:
    """Decode raw JPEG/PNG bytes to a BGR image."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode image bytes")
    return image
