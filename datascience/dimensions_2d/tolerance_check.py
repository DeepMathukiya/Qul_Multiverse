"""Compare measured dimensions against configured specs and tolerances."""

from __future__ import annotations

from datascience.schemas import CheckStatus, Dim2DResult, Dim3DResult, QualityCheck


def check_dimensions(
    result: Dim2DResult | Dim3DResult,
    specs: dict,
    prefix: str,
) -> list[QualityCheck]:
    """Build one QualityCheck per configured dimension spec.

    specs: {"length_mm": {"expected": 120.0, "tolerance": 3.0}, ...}
    The spec name must match a Measurement name; mm specs against px-only
    measurements yield NOT_AVAILABLE ("no calibration") instead of a bogus
    comparison.
    """
    checks: list[QualityCheck] = []
    if not specs:
        return checks

    measured_by_name = {m.name: m for m in result.measurements}

    for name, spec in specs.items():
        expected = float(spec["expected"])
        tolerance = float(spec.get("tolerance", 0.0))
        display_expected = f"{expected} ± {tolerance}"
        check_name = f"{prefix}: {name}"

        measurement = measured_by_name.get(name)
        if measurement is None:
            # Requested in mm but only px available (or stage failed).
            checks.append(
                QualityCheck(
                    name=check_name,
                    status=CheckStatus.NOT_AVAILABLE,
                    expected=display_expected,
                    reason=(
                        "measurement unavailable — needs calibration/scale"
                        if result.status != CheckStatus.NOT_AVAILABLE
                        else "stage did not run"
                    ),
                )
            )
            continue

        deviation = abs(measurement.value - expected)
        ok = deviation <= tolerance
        checks.append(
            QualityCheck(
                name=check_name,
                status=CheckStatus.PASS if ok else CheckStatus.FAIL,
                measured=f"{measurement.value} {measurement.unit}",
                expected=display_expected,
                reason=None if ok else f"deviation {deviation:.2f} exceeds tolerance {tolerance}",
            )
        )
    return checks
