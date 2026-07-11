"""Configurable quality-rule evaluation over all inspection results.

Every rule produces a QualityCheck with measured value, expected range and a
human-readable reason, so the final decision is fully explainable.
"""

from __future__ import annotations

from datascience.config_loader import load_product_specs
from datascience.schemas import (
    CheckStatus,
    Dim2DResult,
    Dim3DResult,
    OcrResult,
    PotholeResult,
    QualityCheck,
    SurfaceDefectResult,
)

from datascience.dimensions_2d.tolerance_check import check_dimensions


def _ocr_checks(ocr: OcrResult, rules: dict) -> list[QualityCheck]:
    checks: list[QualityCheck] = []

    if ocr.status in (CheckStatus.NOT_AVAILABLE, CheckStatus.SKIPPED):
        checks.append(
            QualityCheck(
                name="OCR: product information",
                status=ocr.status,
                reason=ocr.error or "OCR did not run",
            )
        )
    else:
        for field_name in rules.get("required_fields", []):
            value = ocr.fields.model_dump().get(field_name)
            checks.append(
                QualityCheck(
                    name=f"OCR: {field_name} present",
                    status=CheckStatus.PASS if value else CheckStatus.FAIL,
                    measured=value or "not found",
                    expected="present",
                    reason=None if value else f"{field_name} missing on label",
                )
            )

        if rules.get("require_valid_expiry", True):
            if ocr.expiry_valid is None:
                status, reason = CheckStatus.NOT_AVAILABLE, "expiry date unreadable"
            elif ocr.expiry_valid:
                status, reason = CheckStatus.PASS, None
            else:
                status, reason = CheckStatus.FAIL, "product expired"
            checks.append(
                QualityCheck(
                    name="OCR: expiry date valid",
                    status=status,
                    measured=ocr.fields.expiry_date or "unreadable",
                    expected="date in the future",
                    reason=reason,
                )
            )

    if rules.get("require_qr", False):
        checks.append(
            QualityCheck(
                name="QR code present",
                status=CheckStatus.PASS if ocr.qr_present else CheckStatus.FAIL,
                measured=(ocr.qr_data[:40] + "…") if ocr.qr_data and len(ocr.qr_data) > 40
                else (ocr.qr_data or "not detected"),
                expected="decodable QR code",
                reason=None if ocr.qr_present else "no QR code detected",
            )
        )
    return checks


def _defect_checks(defects: SurfaceDefectResult, rules: dict) -> list[QualityCheck]:
    checks: list[QualityCheck] = []

    crack_rules = rules.get("crack", {})
    n_cracks = len(defects.cracks)
    if not crack_rules.get("allowed", False):
        checks.append(
            QualityCheck(
                name="Defect: no cracks",
                status=CheckStatus.PASS if n_cracks == 0 else CheckStatus.FAIL,
                measured=f"{n_cracks} crack(s)",
                expected="0 cracks",
                reason=None if n_cracks == 0 else
                f"crack detected (longest {max(c.length for c in defects.cracks):.0f} px)",
            )
        )

    scratch_rules = rules.get("scratch", {})
    n_scratches = len(defects.scratches)
    if not scratch_rules.get("allowed", True):
        checks.append(
            QualityCheck(
                name="Defect: no scratches",
                status=CheckStatus.PASS if n_scratches == 0 else CheckStatus.FAIL,
                measured=f"{n_scratches} scratch(es)",
                expected="0 scratches",
                reason=None if n_scratches == 0 else "scratch detected",
            )
        )
    elif n_scratches and scratch_rules.get("max_length_mm") is not None:
        # Length limit is in mm; scratch metrics may be px if uncalibrated.
        longest = max(defects.scratches, key=lambda s: s.length)
        if longest.unit == "mm":
            limit = float(scratch_rules["max_length_mm"])
            ok = longest.length <= limit
            checks.append(
                QualityCheck(
                    name="Defect: scratch length within limit",
                    status=CheckStatus.PASS if ok else CheckStatus.FAIL,
                    measured=f"{longest.length:.1f} mm",
                    expected=f"<= {limit} mm",
                    reason=None if ok else "scratch exceeds allowed length",
                )
            )
        else:
            checks.append(
                QualityCheck(
                    name="Defect: scratch length within limit",
                    status=CheckStatus.NOT_AVAILABLE,
                    measured=f"{longest.length:.0f} px",
                    expected=f"<= {scratch_rules['max_length_mm']} mm",
                    reason="no calibration — cannot compare px against mm limit",
                )
            )

    dent_rules = rules.get("dent", {})
    if defects.dents:
        deepest = max(defects.dents, key=lambda d: d.max_depth)
        if not dent_rules.get("allowed", True):
            checks.append(
                QualityCheck(
                    name="Defect: no dents",
                    status=CheckStatus.FAIL,
                    measured=f"{len(defects.dents)} dent(s)",
                    expected="0 dents",
                    reason=f"dent detected (max depth {deepest.max_depth:.2f} mm)",
                )
            )
        elif dent_rules.get("max_depth_mm") is not None:
            limit = float(dent_rules["max_depth_mm"])
            ok = deepest.max_depth <= limit
            checks.append(
                QualityCheck(
                    name="Defect: dent depth within limit",
                    status=CheckStatus.PASS if ok else CheckStatus.FAIL,
                    measured=f"{deepest.max_depth:.2f} mm",
                    expected=f"<= {limit} mm",
                    reason=None if ok else "dent deeper than allowed",
                )
            )
    elif defects.dent_note:
        checks.append(
            QualityCheck(
                name="Defect: dent analysis",
                status=CheckStatus.NOT_AVAILABLE,
                reason=defects.dent_note,
            )
        )

    return checks


def _pothole_checks(pothole: PotholeResult, rules: dict) -> list[QualityCheck]:
    if not pothole.enabled or pothole.status == CheckStatus.SKIPPED:
        return []
    if pothole.status == CheckStatus.NOT_AVAILABLE:
        return [
            QualityCheck(
                name="Pothole: analysis",
                status=CheckStatus.NOT_AVAILABLE,
                reason=pothole.error or "pothole model unavailable",
            )
        ]
    if rules.get("fail_if_present", True):
        n = len(pothole.detections)
        worst = max((d.severity for d in pothole.detections), default="none")
        return [
            QualityCheck(
                name="Pothole: none present",
                status=CheckStatus.PASS if not pothole.present else CheckStatus.FAIL,
                measured=f"{n} pothole(s), worst severity: {worst}",
                expected="0 potholes",
                reason=None if not pothole.present else "pothole detected",
            )
        ]
    return []


def evaluate_quality_rules(
    ocr: OcrResult,
    dims_2d: Dim2DResult,
    dims_3d: Dim3DResult,
    defects: SurfaceDefectResult,
    pothole: PotholeResult,
) -> list[QualityCheck]:
    specs = load_product_specs()

    checks: list[QualityCheck] = []
    checks += _ocr_checks(ocr, specs.get("ocr_rules", {}))
    checks += check_dimensions(dims_2d, specs.get("dimensions_2d", {}), "2D dimension")
    checks += check_dimensions(dims_3d, specs.get("dimensions_3d", {}), "3D dimension")
    checks += _defect_checks(defects, specs.get("defect_rules", {}))
    checks += _pothole_checks(pothole, specs.get("pothole_rules", {}))
    return checks
