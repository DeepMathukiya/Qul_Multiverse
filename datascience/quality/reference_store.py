"""In-memory reference-area baseline, captured from the most recent
Upload-mode inspection and read back by area comparisons on later
Live/stream inspections. Cleared on process restart — nothing is persisted.
"""

from __future__ import annotations

_reference: dict | None = None


def set_reference_area(value: float, unit: str) -> None:
    global _reference
    _reference = {"value": value, "unit": unit}


def get_reference_area() -> dict | None:
    return _reference
