"""Crop the known, configured Region of Interest for dimensional inspection."""

from __future__ import annotations

import numpy as np

from datascience.config_loader import load_system_config


def get_roi_rect(image: np.ndarray) -> tuple[int, int, int, int]:
    """Return the configured ROI as absolute pixels (x, y, w, h)."""
    roi = load_system_config().get("roi", {})
    h, w = image.shape[:2]

    x = int(round(float(roi.get("x", 0.0)) * w))
    y = int(round(float(roi.get("y", 0.0)) * h))
    rw = int(round(float(roi.get("w", 1.0)) * w))
    rh = int(round(float(roi.get("h", 1.0)) * h))

    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    rw = max(1, min(rw, w - x))
    rh = max(1, min(rh, h - y))
    return x, y, rw, rh


def extract_roi(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop the configured ROI. Returns (crop, (x, y, w, h))."""
    x, y, rw, rh = get_roi_rect(image)
    return image[y : y + rh, x : x + rw], (x, y, rw, rh)
