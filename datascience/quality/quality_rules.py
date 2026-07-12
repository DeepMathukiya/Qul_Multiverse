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
from datascience.quality.reference_area_check import check_area_against_reference


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

    return checks


def _defect_checks(defects: SurfaceDefectResult) -> list[QualityCheck]:
    """Any YOLO-detected surface defect fails the product — the class doesn't
    matter for the decision, only whether something was found. Every entry in
    defects.detections already cleared the model's own confidence cutoff
    (defect_detection.confidence), so no second threshold is applied here —
    otherwise a defect that is visibly detected/drawn could still PASS.
    """
    if defects.status in (CheckStatus.NOT_AVAILABLE, CheckStatus.SKIPPED):
        return [
            QualityCheck(
                name="Surface defects: analysis",
                status=defects.status,
                reason=defects.error or defects.note or "defect model unavailable",
            )
        ]

    n = len(defects.detections)

    return [
        QualityCheck(
            name="Surface defects: none present",
            status=CheckStatus.PASS if n == 0 else CheckStatus.FAIL,
            measured=f"{n} defect(s)",
            expected="0 defects",
            reason=None if n == 0 else f"{n} surface defect(s) detected",
        )
    ]


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
    area_tolerance_ratio: float | None = None,
) -> list[QualityCheck]:
    specs = load_product_specs()

    # area_tolerance_ratio: None = use product_specs.yaml; otherwise a
    # per-request override (e.g. the dashboard's tolerance slider).
    area_rules = dict(specs.get("area_reference_rules", {}))
    if area_tolerance_ratio is not None:
        area_rules["tolerance_ratio"] = area_tolerance_ratio

    checks: list[QualityCheck] = []
    checks += _ocr_checks(ocr, specs.get("ocr_rules", {}))
    checks += check_dimensions(dims_2d, specs.get("dimensions_2d", {}), "2D dimension")
    checks += check_dimensions(dims_3d, specs.get("dimensions_3d", {}), "3D dimension")
    checks += check_area_against_reference(dims_2d, area_rules)
    checks += _defect_checks(defects)
    checks += _pothole_checks(pothole, specs.get("pothole_rules", {}))
    return checks
