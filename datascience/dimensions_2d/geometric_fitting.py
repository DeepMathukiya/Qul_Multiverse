"""Geometric primitives fitted to the extracted product boundary."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RectFit:
    center: tuple[float, float]
    length_px: float   # longer side
    width_px: float    # shorter side
    angle_deg: float
    box_points: np.ndarray  # 4x2 corner points


@dataclass
class CircleFit:
    center: tuple[float, float]
    radius_px: float


def fit_min_area_rect(contour: np.ndarray) -> RectFit:
    (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
    length, width = (h, w) if h >= w else (w, h)
    box = cv2.boxPoints(((cx, cy), (w, h), angle))
    return RectFit(
        center=(cx, cy),
        length_px=float(length),
        width_px=float(width),
        angle_deg=float(angle),
        box_points=box,
    )


def fit_enclosing_circle(contour: np.ndarray) -> CircleFit:
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    return CircleFit(center=(float(cx), float(cy)), radius_px=float(radius))


def fit_hole_circles(hole_contours: list[np.ndarray]) -> list[CircleFit]:
    return [fit_enclosing_circle(c) for c in hole_contours]


def hole_center_distances(holes: list[CircleFit]) -> list[float]:
    """Pairwise distances between hole centers, in pixels."""
    distances = []
    for i in range(len(holes)):
        for j in range(i + 1, len(holes)):
            dx = holes[i].center[0] - holes[j].center[0]
            dy = holes[i].center[1] - holes[j].center[1]
            distances.append(float(np.hypot(dx, dy)))
    return distances


def corner_angles(contour: np.ndarray, max_corners: int = 8) -> list[float]:
    """Interior angles (degrees) at the dominant polygon corners."""
    perimeter = cv2.arcLength(contour, True)
    poly = cv2.approxPolyDP(contour, 0.02 * perimeter, True).reshape(-1, 2)
    n = len(poly)
    if n < 3 or n > max_corners:
        return []

    angles = []
    for i in range(n):
        prev_pt = poly[(i - 1) % n].astype(np.float64)
        cur_pt = poly[i].astype(np.float64)
        next_pt = poly[(i + 1) % n].astype(np.float64)
        v1 = prev_pt - cur_pt
        v2 = next_pt - cur_pt
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom < 1e-9:
            continue
        cos_a = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
        angles.append(float(np.degrees(np.arccos(cos_a))))
    return angles


def roundness(contour: np.ndarray) -> float:
    """4*pi*A / P^2 — 1.0 for a perfect circle."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter * perimeter))
