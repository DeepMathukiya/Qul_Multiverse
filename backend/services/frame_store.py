"""Thread-safe latest-frame storage per device (adapted from example_receiver.py)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StoredFrame:
    image: np.ndarray
    device_timestamp_ms: float  # timestamp reported by the phone (0 if absent)
    received_at: float = field(default_factory=time.time)


class FrameStore:
    """Keeps the most recent frame for every streaming device."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: dict[str, StoredFrame] = {}
        self._first_seen_order: list[str] = []

    def update(self, device_id: str, image: np.ndarray, device_timestamp_ms: float) -> None:
        with self._lock:
            if device_id not in self._frames:
                self._first_seen_order.append(device_id)
            self._frames[device_id] = StoredFrame(
                image=image.copy(),
                device_timestamp_ms=device_timestamp_ms,
            )

    def get(self, device_id: str) -> StoredFrame | None:
        with self._lock:
            frame = self._frames.get(device_id)
            if frame is None:
                return None
            return StoredFrame(
                image=frame.image.copy(),
                device_timestamp_ms=frame.device_timestamp_ms,
                received_at=frame.received_at,
            )

    def devices(self) -> list[str]:
        with self._lock:
            return list(self._first_seen_order)


frame_store = FrameStore()
