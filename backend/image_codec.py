"""Backend-owned image byte/base64 helpers (no shared code between layers)."""

from __future__ import annotations

import base64

import cv2
import numpy as np


def encode_image_b64(image: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode image as JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_image_bytes(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode image bytes")
    return image
