"""Continuous streaming inspection.

A background worker that keeps inspecting the latest vertical/horizontal pair from
the streaming phones: pair -> datascience /process -> result store, in a
loop, so the dashboard always shows a fresh result without manual clicks.
"""

from __future__ import annotations

import threading
import time

from backend.config_loader import load_backend_config
from backend.services.datascience_client import (
    DatascienceUnavailable,
    run_inspection_remote,
    run_single_frame_inspection_remote,
)
from backend.services.frame_pairing import PairingError, get_latest_pair, get_single_frame
from backend.services.result_store import result_store


class ContinuousInspector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._interval_sec = 5.0
        self.inspection_count = 0
        self.last_inspection_id: str | None = None
        self.last_error: str | None = None
        self.last_run_at: float | None = None
        self.ocr_enabled = True
        self.area_tolerance_ratio: float | None = None

    # ---- control ----

    def start(
        self,
        interval_sec: float | None = None,
        ocr_enabled: bool | None = None,
        area_tolerance_ratio: float | None = None,
    ) -> None:
        """Start the loop, or update its settings if already running."""
        with self._lock:
            if interval_sec is not None:
                self._interval_sec = max(0.1, float(interval_sec))
            if ocr_enabled is not None:
                self.ocr_enabled = bool(ocr_enabled)
            if area_tolerance_ratio is not None:
                self.area_tolerance_ratio = float(area_tolerance_ratio)
            if self.running:
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop, name="continuous-inspection", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict:
        return {
            "running": self.running,
            "interval_sec": self._interval_sec,
            "ocr_enabled": self.ocr_enabled,
            "area_tolerance_ratio": self.area_tolerance_ratio,
            "inspection_count": self.inspection_count,
            "last_inspection_id": self.last_inspection_id,
            "last_error": self.last_error,
            "last_run_at": self.last_run_at,
        }

    # ---- worker ----

    def _loop(self) -> None:
        print(f"[stream] continuous inspection started (every {self._interval_sec}s)")
        while not self._stop_event.is_set():
            started = time.time()
            try:
                try:
                    pair = get_latest_pair()
                    result = run_inspection_remote(
                        pair.vertical_image,
                        pair.horizontal_image,
                        pair.vertical_device_id,
                        pair.horizontal_device_id,
                        ocr_enabled=self.ocr_enabled,
                        area_tolerance_ratio=self.area_tolerance_ratio,
                    )
                except PairingError:
                    # Only one phone connected — no stereo pair to measure
                    # area/volume from. Judge the product from crack/dent
                    # (surface-defect) detection on that single frame instead.
                    device_id, frame = get_single_frame()
                    result = run_single_frame_inspection_remote(
                        frame, device_id, ocr_enabled=self.ocr_enabled,
                    )
                result_store.add(result)
                self.inspection_count += 1
                self.last_inspection_id = result.get("inspection_id")
                self.last_error = None
                self.last_run_at = started
            except PairingError as exc:
                # No/one device streaming yet — keep waiting quietly.
                self.last_error = f"waiting for devices: {exc}"
            except DatascienceUnavailable as exc:
                self.last_error = str(exc)
            except Exception as exc:  # never let the loop die
                self.last_error = f"unexpected error: {exc}"

            # Inspection time counts toward the interval.
            elapsed = time.time() - started
            self._stop_event.wait(max(0.2, self._interval_sec - elapsed))
        print("[stream] continuous inspection stopped")


continuous_inspector = ContinuousInspector()


def autostart_if_configured() -> None:
    """Start the loop at service boot when configured/env-enabled."""
    cfg = load_backend_config().get("continuous", {}) or {}
    if cfg.get("enabled", False):
        continuous_inspector.start(cfg.get("interval_sec"))
