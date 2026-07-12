"""Pair vertical/horizontal frames from the two phones into one stereo inspection event."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.config_loader import load_backend_config
from backend.services.frame_store import frame_store


@dataclass
class StereoPair:
    vertical_image: np.ndarray
    horizontal_image: np.ndarray
    vertical_device_id: str
    horizontal_device_id: str
    time_delta_ms: float
    synchronized: bool


class PairingError(Exception):
    pass


def resolve_camera_roles() -> tuple[str, str]:
    """Map device ids to vertical/horizontal roles.

    Uses the configured ids when those devices are streaming; otherwise, with
    auto_assign enabled, the first two devices seen become vertical and horizontal.
    """
    cfg = load_backend_config()["cameras"]
    devices = frame_store.devices()

    vertical_id = cfg.get("vertical_device_id")
    horizontal_id = cfg.get("horizontal_device_id")

    if vertical_id in devices and horizontal_id in devices:
        return vertical_id, horizontal_id

    if cfg.get("auto_assign", True) and len(devices) >= 2:
        return devices[0], devices[1]

    raise PairingError(
        f"need two streaming devices, currently connected: {devices or 'none'}"
    )


def get_single_frame() -> tuple[str, np.ndarray]:
    """Return the one device currently streaming, for when a full stereo
    pair isn't available (only one phone connected). Dimension/volume
    checks need the pair, so callers fall back to defect-only judgment."""
    devices = frame_store.devices()
    if len(devices) != 1:
        raise PairingError(
            f"need exactly one streaming device for single-frame mode, "
            f"currently connected: {devices or 'none'}"
        )

    device_id = devices[0]
    stored = frame_store.get(device_id)
    if stored is None:
        raise PairingError(f"no frame yet from '{device_id}'")

    return device_id, stored.image


def get_latest_pair() -> StereoPair:
    """Build a StereoPair from the latest frame of each role."""
    cfg = load_backend_config()["cameras"]
    tolerance_ms = float(cfg.get("pair_tolerance_ms", 750))

    vertical_id, horizontal_id = resolve_camera_roles()

    vertical = frame_store.get(vertical_id)
    horizontal = frame_store.get(horizontal_id)

    if vertical is None or horizontal is None:
        raise PairingError("missing frame from one of the paired devices")

    # Prefer phone-reported timestamps; fall back to server receive time.
    if vertical.device_timestamp_ms > 0 and horizontal.device_timestamp_ms > 0:
        delta_ms = abs(vertical.device_timestamp_ms - horizontal.device_timestamp_ms)
    else:
        delta_ms = abs(vertical.received_at - horizontal.received_at) * 1000.0

    return StereoPair(
        vertical_image=vertical.image,
        horizontal_image=horizontal.image,
        vertical_device_id=vertical_id,
        horizontal_device_id=horizontal_id,
        time_delta_ms=round(delta_ms, 1),
        synchronized=delta_ms <= tolerance_ms,
    )
