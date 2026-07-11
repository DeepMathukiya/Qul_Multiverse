"""Resize, denoise and illumination normalization applied before analysis."""

from __future__ import annotations

import cv2
import numpy as np

from datascience.config_loader import load_system_config


def resize_keep_aspect(image: np.ndarray, target_width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= target_width:
        return image
    scale = target_width / w
    return cv2.resize(
        image,
        (target_width, int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def denoise(image: np.ndarray) -> np.ndarray:
    # Bilateral filter smooths sensor noise while keeping edges sharp,
    # which matters for edge-based dimensional measurement.
    return cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """CLAHE + large-kernel background division to flatten uneven lighting."""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    background = cv2.GaussianBlur(equalized, (0, 0), sigmaX=31)
    background = np.clip(background, 1, 255)
    flattened = cv2.divide(equalized, background, scale=128)
    return cv2.normalize(flattened, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def preprocess_frame(image: np.ndarray) -> np.ndarray:
    """Standard preprocessing for one camera frame (color output)."""
    cfg = load_system_config().get("preprocessing", {})

    out = resize_keep_aspect(image, int(cfg.get("resize_width", 1280)))

    if cfg.get("denoise", True):
        out = denoise(out)

    return out


def preprocess_gray(image: np.ndarray) -> np.ndarray:
    """Grayscale + illumination-normalized version for defect analysis."""
    cfg = load_system_config().get("preprocessing", {})
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if cfg.get("illumination_normalize", True):
        gray = normalize_illumination(gray)
    return gray
