"""Qwen3-VL-based OCR field extraction (replaces the old Sarvam client).

Flow: vertical + horizontal images -> GenieX-served Qwen3-VL -> strict JSON
-> OcrFields. Any failure (server down, malformed JSON) is reported as an
error string so the pipeline can degrade gracefully.
"""

from __future__ import annotations

import numpy as np

from datascience.schemas import OcrFields
from datascience.vlm.geniex_client import call_vlm
from datascience.vlm.prompts import ocr_prompt
from datascience.vlm.response_json import parse_json_response


def extract_product_info(
    vertical: np.ndarray, horizontal: np.ndarray
) -> tuple[OcrFields, str, str | None]:
    """Run Qwen3-VL OCR extraction. Returns (fields, raw_text, error)."""
    text, error = call_vlm([vertical, horizontal], ocr_prompt())
    if error:
        return OcrFields(), "", error

    data = parse_json_response(text)
    if not isinstance(data, dict):
        return OcrFields(), text or "", "VLM did not return a valid JSON object"

    fields = OcrFields(
        expiry_date=data.get("expiry_date") or None,
        manufacturing_date=data.get("manufacturing_date") or None,
        serial_number=data.get("serial_number") or None,
        batch_number=data.get("batch_number") or None,
        product_id=data.get("product_id") or None,
    )
    raw_text = data.get("raw_text") or ""
    return fields, raw_text, None
