"""Thread-safe latest-frame storage per device (adapted from example_receiver.py)."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

FPS_WINDOW_FRAMES = 30
FPS_STALE_SEC = 3.0  # report 0 fps if no frame arrived for this long


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
        self._recv_times: dict[str, deque[float]] = {}

    def update(self, device_id: str, image: np.ndarray, device_timestamp_ms: float) -> None:
        with self._lock:
            if device_id not in self._frames:
                self._first_seen_order.append(device_id)
                self._recv_times[device_id] = deque(maxlen=FPS_WINDOW_FRAMES)
            self._recv_times[device_id].append(time.time())
            self._frames[device_id] = StoredFrame(
                image=image.copy(),
                device_timestamp_ms=device_timestamp_ms,
            )

    def get_fps(self, device_id: str) -> float:
        """Incoming stream FPS over the recent frame window (0 if stale)."""
        with self._lock:
            times = self._recv_times.get(device_id)
            if not times or len(times) < 2:
                return 0.0
            if time.time() - times[-1] > FPS_STALE_SEC:
                return 0.0
            span = times[-1] - times[0]
            if span <= 0:
                return 0.0
            return round((len(times) - 1) / span, 1)

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
