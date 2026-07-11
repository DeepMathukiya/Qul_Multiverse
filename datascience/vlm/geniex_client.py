"""Thin client for Qwen3-VL-4B-Instruct served locally via Qualcomm GenieX's
OpenAI-compatible server (`geniex serve`, default
http://127.0.0.1:18181/v1).

Any failure (server not running, timeout, malformed response) is reported
as an error string so callers can degrade gracefully instead of crashing
the pipeline — mirrors the pattern the old Sarvam client used.
"""

from __future__ import annotations

import numpy as np

from datascience.config_loader import load_system_config
from datascience.image_codec import encode_image_b64


def call_vlm(images: list[np.ndarray], prompt: str) -> tuple[str | None, str | None]:
    """Send one or more images + a text prompt to the configured VLM.

    Returns (raw_response_text, error).
    """
    cfg = load_system_config().get("vlm", {})
    base_url = cfg.get("base_url", "http://127.0.0.1:18181/v1")
    model = cfg.get("model", "ai-hub-models/Qwen3-VL-4B-Instruct")
    api_key = cfg.get("api_key") or "not-needed-local"
    timeout_sec = float(cfg.get("timeout_sec", 90))
    max_tokens = int(cfg.get("max_tokens", 800))
    temperature = float(cfg.get("temperature", 0.1))

    try:
        from openai import OpenAI
    except ImportError:
        return None, "openai package not installed (pip install openai)"

    content: list[dict] = [{"type": "text", "text": prompt}]
    for image in images:
        b64 = encode_image_b64(image)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )

    try:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_sec)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = response.choices[0].message.content
        if not text:
            return None, "VLM returned an empty response"
        return text, None
    except Exception as exc:  # server down / timeout / SDK errors must not kill the pipeline
        return None, f"GenieX VLM call failed ({base_url}): {exc}"
