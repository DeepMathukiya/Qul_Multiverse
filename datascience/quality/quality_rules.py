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
    """One check per configured defect class, evaluated over YOLO detections.

    `allowed: false` fails on any detection; `allowed: true` with `max_count`
    fails only above that count. Detections below `min_confidence` are ignored
    for pass/fail (they still show in the detailed report).
    """
    if defects.status in (CheckStatus.NOT_AVAILABLE, CheckStatus.SKIPPED):
        return [
            QualityCheck(
                name="Surface defects: analysis",
                status=defects.status,
                reason=defects.error or defects.note or "defect model unavailable",
            )
        ]

    min_conf = float(rules.get("min_confidence", 0.0))
    class_rules: dict = rules.get("classes", {})

    checks: list[QualityCheck] = []
    for label, crule in class_rules.items():
        n = sum(
            1 for d in defects.detections
            if d.label == label and d.confidence >= min_conf
        )
        allowed = crule.get("allowed", False)
        max_count = crule.get("max_count")

        if not allowed:
            checks.append(
                QualityCheck(
                    name=f"Defect: no {label}",
                    status=CheckStatus.PASS if n == 0 else CheckStatus.FAIL,
                    measured=f"{n} {label}",
                    expected=f"0 {label}",
                    reason=None if n == 0 else f"{label} detected",
                )
            )
        elif max_count is not None:
            ok = n <= int(max_count)
            checks.append(
                QualityCheck(
                    name=f"Defect: {label} within limit",
                    status=CheckStatus.PASS if ok else CheckStatus.FAIL,
                    measured=f"{n} {label}",
                    expected=f"<= {max_count} {label}",
                    reason=None if ok else f"too many {label} ({n} > {max_count})",
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
