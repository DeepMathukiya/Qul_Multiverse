"""Extract the product's outer boundary inside the known ROI.

Classical CV only: threshold + Canny/Sobel edges + morphology + contours.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BoundaryResult:
    outer_contour: np.ndarray          # (N,1,2) points, ROI coordinates
    hole_contours: list[np.ndarray]    # inner contours (holes) if any
    mask: np.ndarray                   # filled product mask (uint8 0/255)
    edges: np.ndarray                  # edge map used


def _binary_product_mask(gray: np.ndarray) -> np.ndarray:
    """Otsu threshold with automatic polarity so the product is white."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If the image border is mostly white the polarity is inverted
    # (background became foreground).
    border = np.concatenate(
        [binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]]
    )
    if border.mean() > 127:
        binary = cv2.bitwise_not(binary)
    return binary


def _edge_map(gray: np.ndarray) -> np.ndarray:
    """Canny with Sobel-magnitude reinforcement for weak boundaries."""
    median = float(np.median(gray))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    canny = cv2.Canny(gray, lower, upper)

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, strong = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return cv2.bitwise_or(canny, strong)


def extract_product_boundary(roi_bgr: np.ndarray) -> BoundaryResult | None:
    """Return the refined outer boundary of the product in the ROI."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    binary = _binary_product_mask(gray)
    edges = _edge_map(gray)

    # Fuse region and edge evidence, then clean up.
    fused = cv2.bitwise_or(binary, edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, kernel, iterations=2)
    fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN, kernel, iterations=1)

    # RETR_CCOMP: top level = outer boundaries, second level = holes.
    contours, hierarchy = cv2.findContours(
        fused, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    if not contours or hierarchy is None:
        return None

    hierarchy = hierarchy[0]
    outer_ids = [i for i, h in enumerate(hierarchy) if h[3] == -1]
    if not outer_ids:
        return None

    # The product is the largest outer contour.
    best = max(outer_ids, key=lambda i: cv2.contourArea(contours[i]))
    outer = contours[best]

    if cv2.contourArea(outer) < 0.005 * gray.shape[0] * gray.shape[1]:
        return None  # too small — likely noise, no product in ROI

    # Light smoothing of the boundary (edge refinement).
    epsilon = 0.0015 * cv2.arcLength(outer, True)
    outer = cv2.approxPolyDP(outer, epsilon, True)

    # Holes: children of the product contour, big enough to be real.
    min_hole_area = 30.0
    hole_contours = [
        contours[i]
        for i, h in enumerate(hierarchy)
        if h[3] == best and cv2.contourArea(contours[i]) >= min_hole_area
    ]

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [outer], -1, 255, thickness=cv2.FILLED)
    for hole in hole_contours:
        cv2.drawContours(mask, [hole], -1, 0, thickness=cv2.FILLED)

    return BoundaryResult(
        outer_contour=outer,
        hole_contours=hole_contours,
        mask=mask,
        edges=edges,
    )
