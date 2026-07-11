"""Local QR code detection and decoding (independent of the OCR provider)."""

from __future__ import annotations

import cv2
import numpy as np


def decode_qr(image: np.ndarray) -> str | None:
    """Return the decoded QR payload, or None if no QR was found."""
    detector = cv2.QRCodeDetector()

    data, _, _ = detector.detectAndDecode(image)
    if data:
        return data

    # Retry on an upscaled, contrast-boosted grayscale — helps small codes
    # from phone frames.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    scaled = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    data, _, _ = detector.detectAndDecode(scaled)
    return data or None
