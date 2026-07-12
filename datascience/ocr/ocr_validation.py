"""Run the full OCR inspection: Sarvam text + parsing + validation."""

from __future__ import annotations

import numpy as np

from datascience.config_loader import load_product_specs, load_system_config
from datascience.schemas import CheckStatus, OcrResult

from datascience.ocr.field_parsing import is_expiry_valid, parse_fields
from datascience.ocr.sarvam_client import extract_text


def inspect_product_info(
    image: np.ndarray,
    enabled_override: bool | None = None,
) -> OcrResult:
    """Full OCR inspection. enabled_override (from the API request) wins
    over the ocr.enabled config value."""
    result = OcrResult()

    ocr_cfg = load_system_config().get("ocr", {})
    rules = load_product_specs().get("ocr_rules", {})

    enabled = (
        enabled_override
        if enabled_override is not None
        else ocr_cfg.get("enabled", True)
    )
    if not enabled:
        result.status = CheckStatus.SKIPPED
        result.error = "OCR turned off for this inspection"
        return result

    text, error = extract_text(image)
    result.raw_text = text
    if error:
        result.status = CheckStatus.NOT_AVAILABLE
        result.error = error
        return result

    result.fields = parse_fields(text)
    result.expiry_valid = is_expiry_valid(result.fields.expiry_date)

    required = rules.get("required_fields", [])
    field_values = result.fields.model_dump()
    result.missing_required_fields = [
        name for name in required if not field_values.get(name)
    ]

    ok = not result.missing_required_fields
    if rules.get("require_valid_expiry", False):
        ok = ok and result.expiry_valid is True

    result.status = CheckStatus.PASS if ok else CheckStatus.FAIL
    return result
