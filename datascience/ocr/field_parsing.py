"""Parse product information fields out of raw OCR text."""

from __future__ import annotations

import re
from datetime import date, datetime

from datascience.schemas import OcrFields

# dd/mm/yyyy, dd-mm-yyyy, dd.mm.yyyy, yyyy-mm-dd, mm/yyyy, "MAR 2027", "12 MAR 2027"
_MONTHS = "JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC"
_DATE_PATTERNS = [
    r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}",
    r"\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}",
    r"\d{1,2}[/\-.]\d{4}",
    rf"(?:\d{{1,2}}\s+)?(?:{_MONTHS})[a-z]*\s*[,\s]\s*\d{{4}}",
    rf"(?:{_MONTHS})[a-z]*\s*[/\-.]\s*\d{{2,4}}",
]
_DATE_RE = "(" + "|".join(_DATE_PATTERNS) + ")"

_EXPIRY_KEYS = r"(?:EXP(?:IRY|\.)?(?:\s*DATE)?|USE\s*BY|BEST\s*BEFORE|BB|EXPIRES?)"
_MFG_KEYS = r"(?:MFG|MFD|MANUFACTUR(?:ED|ING)(?:\s*DATE)?|PKD|PACKED)"

_FIELD_PATTERNS = {
    "expiry_date": rf"{_EXPIRY_KEYS}\s*[:.\-]?\s*{_DATE_RE}",
    "manufacturing_date": rf"{_MFG_KEYS}\s*[:.\-]?\s*{_DATE_RE}",
    "serial_number": r"(?:S/?N|SERIAL(?:\s*(?:NO|NUMBER))?)\s*[:.\-]?\s*([A-Z0-9][A-Z0-9\-]{3,})",
    "batch_number": r"(?:BATCH(?:\s*(?:NO|NUMBER))?|LOT(?:\s*NO)?|B\.?\s*NO)\s*[:.\-]?\s*([A-Z0-9][A-Z0-9\-]{2,})",
    "product_id": r"(?:PRODUCT\s*ID|P/?N|PROD\.?\s*(?:ID|NO)|ITEM\s*(?:ID|NO))\s*[:.\-]?\s*([A-Z0-9][A-Z0-9\-]{2,})",
}


def parse_fields(text: str) -> OcrFields:
    upper = text.upper()
    values: dict[str, str | None] = {}
    for field_name, pattern in _FIELD_PATTERNS.items():
        match = re.search(pattern, upper, flags=re.IGNORECASE)
        values[field_name] = match.group(1).strip() if match else None
    return OcrFields(**values)


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
