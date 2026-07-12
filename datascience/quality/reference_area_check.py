"""Compare measured 2D area against the reference area captured from the
most recent Upload-mode inspection (see reference_store.py).
"""

from __future__ import annotations

from datascience.quality.reference_store import get_reference_area, set_reference_area
from datascience.schemas import CheckStatus, Dim2DResult, Measurement, QualityCheck

CHECK_NAME = "2D dimension: area vs. reference"


def _find_area(measurements: list[Measurement]) -> Measurement | None:
    return next((m for m in measurements if m.name.startswith("area_")), None)


def capture_reference_area(dims_2d: Dim2DResult) -> None:
    """Called after an Upload-mode inspection — its measured area becomes the
    new baseline for area comparisons on subsequent Live/stream inspections."""
    area = _find_area(dims_2d.measurements)
    if area is not None:
        set_reference_area(area.value, area.unit)


def check_area_against_reference(dims_2d: Dim2DResult, rules: dict) -> list[QualityCheck]:
    """FAIL if the measured area is more than `tolerance_ratio` below the
    reference area. NOT_AVAILABLE when no reference has been captured yet, or
    this frame's area measurement isn't available."""
    if not rules.get("enabled", True):
        return []

    reference = get_reference_area()
    if reference is None:
        return [
            QualityCheck(
                name=CHECK_NAME,
                status=CheckStatus.NOT_AVAILABLE,
                reason="no reference image uploaded yet",
            )
        ]

    area = _find_area(dims_2d.measurements)
    if area is None:
        return [
            QualityCheck(
                name=CHECK_NAME,
                status=CheckStatus.NOT_AVAILABLE,
                expected=f">= {reference['value']} {reference['unit']}",
                reason=(
                    "area measurement unavailable — needs calibration/scale"
                    if dims_2d.status != CheckStatus.NOT_AVAILABLE
                    else "stage did not run"
                ),
            )
        ]

    if area.unit != reference["unit"]:
        return [
            QualityCheck(
                name=CHECK_NAME,
                status=CheckStatus.NOT_AVAILABLE,
                measured=f"{area.value} {area.unit}",
                expected=f">= {reference['value']} {reference['unit']}",
                reason="unit mismatch between reference and current measurement (calibration changed?)",
            )
        ]

    tolerance_ratio = float(rules.get("tolerance_ratio", 0.02))
    min_allowed = reference["value"] * (1 - tolerance_ratio)
    ok = area.value >= min_allowed

    # Positive = area is above the reference, negative = below it (lagging).
    delta_pct = (area.value - reference["value"]) / reference["value"] * 100

    return [
        QualityCheck(
            name=CHECK_NAME,
            status=CheckStatus.PASS if ok else CheckStatus.FAIL,
            measured=f"{area.value} {area.unit} ({delta_pct:+.1f}% vs reference)",
            expected=f">= {min_allowed:.2f} {reference['unit']} (ref {reference['value']} {reference['unit']}, -{tolerance_ratio * 100:.0f}% tolerance)",
            reason=(
                None
                if ok
                else f"area is lagging {abs(delta_pct):.1f}% below the reference "
                f"{reference['value']}{reference['unit']} (tolerance allows up to {tolerance_ratio * 100:.0f}% shortfall)"
            ),
        )
    ]
