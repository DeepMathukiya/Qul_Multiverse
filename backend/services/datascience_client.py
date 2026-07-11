"""HTTP client that sends stereo pairs to the datascience /process service.

The backend treats the inspection result as an opaque JSON document — the
result schema is owned entirely by the datascience layer.
"""

from __future__ import annotations

import httpx
import numpy as np

from backend.config_loader import load_backend_config
from backend.image_codec import encode_image_b64


class DatascienceUnavailable(Exception):
    pass


def run_inspection_remote(
    vertical_image: np.ndarray,
    horizontal_image: np.ndarray,
    vertical_device_id: str | None = None,
    horizontal_device_id: str | None = None,
) -> dict:
    cfg = load_backend_config()["datascience"]
    url = cfg["url"].rstrip("/") + "/process"
    timeout = float(cfg.get("process_timeout_sec", 120))

    payload = {
        "vertical_image_b64": encode_image_b64(vertical_image),
        "horizontal_image_b64": encode_image_b64(horizontal_image),
        "vertical_device_id": vertical_device_id,
        "horizontal_device_id": horizontal_device_id,
    }

    try:
        response = httpx.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise DatascienceUnavailable(
            f"datascience service call failed ({url}): {exc}"
        ) from exc

    return response.json()
