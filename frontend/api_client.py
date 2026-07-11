"""HTTP client for the backend REST API (frontend never touches CV code)."""

from __future__ import annotations

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load frontend/.env (real environment wins over file).
load_dotenv(Path(__file__).resolve().parent / ".env")

DEFAULT_BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:5000")


class BackendClient:
    def __init__(self, base_url: str = DEFAULT_BACKEND_URL) -> None:
        self.base_url = base_url.rstrip("/")

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=3)
            return r.ok
        except requests.RequestException:
            return False

    def devices(self) -> list[str]:
        r = requests.get(f"{self.base_url}/devices", timeout=5)
        r.raise_for_status()
        return r.json().get("devices", [])

    def latest_pair(self) -> dict | None:
        r = requests.get(f"{self.base_url}/latest_pair", timeout=10)
        if r.status_code == 409:
            return None  # not enough devices streaming yet
        r.raise_for_status()
        return r.json()

    def inspect_live(
        self, ocr_enabled: bool | None = None, timeout_sec: float = 180
    ) -> dict:
        params = {} if ocr_enabled is None else {"ocr": ocr_enabled}
        r = requests.post(f"{self.base_url}/inspect", params=params, timeout=timeout_sec)
        r.raise_for_status()
        return r.json()

    def inspect_upload(
        self,
        vertical_bytes: bytes,
        horizontal_bytes: bytes,
        ocr_enabled: bool | None = None,
        timeout_sec: float = 180,
    ) -> dict:
        files = {
            "vertical": ("vertical.jpg", vertical_bytes, "image/jpeg"),
            "horizontal": ("horizontal.jpg", horizontal_bytes, "image/jpeg"),
        }
        params = {} if ocr_enabled is None else {"ocr": ocr_enabled}
        r = requests.post(
            f"{self.base_url}/inspect", files=files, params=params, timeout=timeout_sec
        )
        r.raise_for_status()
        return r.json()

    def latest_inspection(self) -> dict | None:
        r = requests.get(f"{self.base_url}/inspections/latest", timeout=10)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def processed_stream(self) -> dict | None:
        """Only the two processed frames + minimal status (no heavy report)."""
        r = requests.get(f"{self.base_url}/stream/processed", timeout=10)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    # ---- continuous streaming mode ----

    def start_stream(
        self,
        interval_sec: float | None = None,
        ocr_enabled: bool | None = None,
    ) -> dict:
        params: dict = {}
        if interval_sec:
            params["interval_sec"] = interval_sec
        if ocr_enabled is not None:
            params["ocr_enabled"] = ocr_enabled
        r = requests.post(f"{self.base_url}/stream/start", params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def stop_stream(self) -> dict:
        r = requests.post(f"{self.base_url}/stream/stop", timeout=10)
        r.raise_for_status()
        return r.json()

    def stream_status(self) -> dict:
        r = requests.get(f"{self.base_url}/stream/status", timeout=10)
        r.raise_for_status()
        return r.json()
