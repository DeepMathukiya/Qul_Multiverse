"""Load the datascience layer's configuration (YAML + .env overrides).

The datascience service is fully self-contained: its .env, config, product
specs and calibration files all live under datascience/.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parent          # datascience/
PROJECT_ROOT = PACKAGE_ROOT.parent
CONFIG_DIR = PACKAGE_ROOT / "config"

# Environment variables from datascience/.env (real environment wins over
# file). Loaded at import time so the CLI pipeline and Sarvam client see the
# variables too.
load_dotenv(PACKAGE_ROOT / ".env")


@lru_cache(maxsize=None)
def load_system_config() -> dict:
    with open(CONFIG_DIR / "processing_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Environment variables override processing_config.yaml values.
    # (SARVAM_API_KEY is read directly from the environment by the OCR
    # client; load_dotenv above makes it available.)
    pothole = config.setdefault("pothole", {})
    if os.environ.get("POTHOLE_WEIGHTS_PATH"):
        pothole["weights_path"] = os.environ["POTHOLE_WEIGHTS_PATH"]

    scale = config.setdefault("scale", {}) or {}
    if os.environ.get("MM_PER_PX"):
        scale["mm_per_px"] = float(os.environ["MM_PER_PX"])
    config["scale"] = scale

    return config


@lru_cache(maxsize=None)
def load_product_specs() -> dict:
    with open(CONFIG_DIR / "product_specs.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calibration_dir() -> Path:
    cfg = load_system_config()
    configured = cfg.get("calibration_dir")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else PACKAGE_ROOT / path
    return CONFIG_DIR / "calibration"
