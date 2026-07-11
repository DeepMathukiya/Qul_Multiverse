"""Prompt templates for the Qwen3-VL calls (OCR field extraction, pothole
detection). Pure string-building — no I/O.
"""

from __future__ import annotations


def ocr_prompt() -> str:
    return (
        "You are inspecting a product's packaging label using two photos of "
        "the same physical unit: the first image is the VERTICAL camera "
        "view, the second image is the HORIZONTAL camera view. Read every "
        "piece of printed or embossed text you can find on the label across "
        "both images (they show the same product from different angles, so "
        "use whichever image shows a field more clearly).\n\n"
        "Return ONLY a single JSON object, with no markdown code fences and "
        "no extra commentary before or after it, in exactly this shape:\n"
        "{\n"
        '  "expiry_date": "<string exactly as printed, e.g. \'12/03/2027\' or \'MAR 2027\', or null if not visible>",\n'
        '  "manufacturing_date": "<string exactly as printed, or null>",\n'
        '  "serial_number": "<string exactly as printed, or null>",\n'
        '  "batch_number": "<string exactly as printed, or null>",\n'
        '  "product_id": "<string exactly as printed, or null>",\n'
        '  "raw_text": "<all legible text on the label, newline separated>"\n'
        "}\n\n"
        "Rules: use null for any field that is not clearly legible in either "
        "image — never guess, infer, or fabricate a value. Preserve dates "
        "exactly in the format printed on the label, do not reformat them. "
        "If no text is legible at all, return null for every field and an "
        "empty string for raw_text."
    )


def pothole_prompt() -> str:
    return (
        "You are inspecting a surface for potholes using two photos of the "
        "same physical area: the first image is the VERTICAL camera view, "
        "the second image is the HORIZONTAL camera view. Identify every "
        "distinct pothole or surface cavity visible in either image "
        "(cross-reference both views of the same area, do not double-count "
        "the same pothole seen from two angles).\n\n"
        "Severity rubric:\n"
        "- low: shallow and small, minor surface blemish\n"
        "- medium: noticeably deep or wide, a clear defect\n"
        "- high: deep and/or large, affects a wide area or looks structural\n\n"
        "Return ONLY a single JSON array, with no markdown code fences and "
        "no extra commentary before or after it, in exactly this shape "
        "(empty array if no potholes are visible):\n"
        "[\n"
        "  {\n"
        '    "severity": "low" | "medium" | "high",\n'
        '    "description": "<one sentence: rough location in frame and approximate size>"\n'
        "  }\n"
        "]\n\n"
        "Rules: only report potholes you can actually see — never guess or "
        "fabricate a detection. If you are not confident something is a "
        "pothole (as opposed to a shadow, stain, or surface texture), leave "
        "it out."
    )
