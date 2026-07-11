"""Millimeters-per-pixel scale for 2D measurement.

Priority:
1. Explicit `scale.mm_per_px` in system_config.yaml (measured once with a
   reference object at the fixed working distance).
2. None — measurements stay in pixels and mm-based tolerance checks report
   NOT_AVAILABLE (the system never pretends pixels are millimeters).

`mm_per_px_from_reference` computes the value from a detected reference
object of known width so it can be pasted into the config.
"""

from __future__ import annotations

from datascience.config_loader import load_system_config


def get_scale() -> tuple[float | None, str]:
    """Return (mm_per_px, source). source in {'config', 'none'}."""
    scale_cfg = load_system_config().get("scale", {}) or {}
    mm_per_px = scale_cfg.get("mm_per_px")
    if mm_per_px:
        return float(mm_per_px), "config"
    return None, "none"


def mm_per_px_from_reference(reference_width_px: float, reference_width_mm: float) -> float:
    if reference_width_px <= 0:
        raise ValueError("reference width in pixels must be positive")
    return reference_width_mm / reference_width_px
