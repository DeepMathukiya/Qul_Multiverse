"""Load the backend layer's configuration (YAML + .env overrides)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parent  # backend/

# Environment variables from backend/.env (real environment wins over file).
load_dotenv(PACKAGE_ROOT / ".env")


@lru_cache(maxsize=None)
def load_backend_config() -> dict:
    with open(PACKAGE_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Environment variables override config.yaml values.
    server = config.setdefault("server", {})
    server["host"] = os.environ.get("BACKEND_HOST", server.get("host", "0.0.0.0"))
    server["port"] = int(os.environ.get("BACKEND_PORT", server.get("port", 5000)))

    ds = config.setdefault("datascience", {})
    ds["url"] = os.environ.get("DATASCIENCE_URL", ds.get("url", "http://127.0.0.1:8100"))
    ds["process_timeout_sec"] = float(
        os.environ.get("PROCESS_TIMEOUT_SEC", ds.get("process_timeout_sec", 120))
    )

    continuous = config.setdefault("continuous", {})
    if os.environ.get("CONTINUOUS_ENABLED") is not None:
        continuous["enabled"] = os.environ["CONTINUOUS_ENABLED"].lower() in ("1", "true", "yes")
    if os.environ.get("CONTINUOUS_INTERVAL_SEC"):
        continuous["interval_sec"] = float(os.environ["CONTINUOUS_INTERVAL_SEC"])

    cameras = config.setdefault("cameras", {})
    cameras["vertical_device_id"] = os.environ.get(
        "VERTICAL_DEVICE_ID", cameras.get("vertical_device_id", "vertical")
    )
    cameras["horizontal_device_id"] = os.environ.get(
        "HORIZONTAL_DEVICE_ID", cameras.get("horizontal_device_id", "horizontal")
    )

    return config
