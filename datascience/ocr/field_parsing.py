"""Deterministic validation of OCR-extracted date strings."""

from __future__ import annotations

import re
from datetime import date, datetime


def parse_date(raw: str | None) -> date | None:
    """Best-effort parse of an OCR'd date string."""
    if not raw:
        return None
    cleaned = re.sub(r"\s+", " ", raw.strip().upper()).replace(",", " ")

    formats = [
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
        "%d/%m/%y", "%d-%m-%y",
        "%m/%d/%Y",
        "%Y-%m-%d", "%Y/%m/%d",
        "%m/%Y", "%m-%Y", "%m.%Y",
        "%b %Y", "%B %Y", "%b/%Y", "%b-%Y", "%b.%Y",
        "%d %b %Y", "%d %B %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def is_expiry_valid(expiry_raw: str | None, today: date | None = None) -> bool | None:
    """True if expiry parses and is in the future; None if unparseable."""
    expiry = parse_date(expiry_raw)
    if expiry is None:
        return None
    return expiry >= (today or date.today())
