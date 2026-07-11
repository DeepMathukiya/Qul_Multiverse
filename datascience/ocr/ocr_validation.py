"""Run the full OCR inspection: Qwen3-VL extraction + QR + validation."""

from __future__ import annotations

import numpy as np

from datascience.config_loader import load_product_specs, load_system_config
from datascience.schemas import CheckStatus, OcrResult

from datascience.ocr.field_parsing import is_expiry_valid
from datascience.ocr.qr_decoding import decode_qr
from datascience.ocr.qwen_ocr_client import extract_product_info


def inspect_product_info(
    vertical: np.ndarray,
    horizontal: np.ndarray,
    enabled_override: bool | None = None,
) -> OcrResult:
    """Full OCR inspection. enabled_override (from the API request) wins
    over the ocr.enabled config value; QR decoding always runs (local)."""
    result = OcrResult()

    ocr_cfg = load_system_config().get("ocr", {})
    rules = load_product_specs().get("ocr_rules", {})

    # QR decoding is local and independent of the OCR provider.
    qr_data = decode_qr(vertical)
    result.qr_present = qr_data is not None
    result.qr_data = qr_data

    enabled = (
        enabled_override
        if enabled_override is not None
        else ocr_cfg.get("enabled", True)
    )
    if not enabled:
        result.status = CheckStatus.SKIPPED
        result.error = "OCR turned off for this inspection"
        return result

    result.fields, result.raw_text, error = extract_product_info(vertical, horizontal)
    if error:
        result.status = CheckStatus.NOT_AVAILABLE
        result.error = error
        return result

    result.expiry_valid = is_expiry_valid(result.fields.expiry_date)

    required = rules.get("required_fields", [])
    field_values = result.fields.model_dump()
    result.missing_required_fields = [
        name for name in required if not field_values.get(name)
    ]

    ok = not result.missing_required_fields
    if rules.get("require_qr", False):
        ok = ok and result.qr_present
    if rules.get("require_valid_expiry", False):
        ok = ok and result.expiry_valid is True

    result.status = CheckStatus.PASS if ok else CheckStatus.FAIL
    return result
