"""Defensive JSON extraction from a VLM's free-form text response."""

from __future__ import annotations

import json
import re


def parse_json_response(raw_text: str) -> dict | list | None:
    """Best-effort parse of a JSON object/array out of model output.

    Models occasionally wrap JSON in markdown code fences or add stray
    prose before/after it. Never raises — returns None if nothing parses.
    """
    if not raw_text:
        return None

    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    return None
