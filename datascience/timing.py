"""Per-stage timing collected into InspectionResult.timings_ms."""

from __future__ import annotations

import time
from contextlib import contextmanager


class StageTimer:
    """Collects wall-clock duration per named pipeline stage."""

    def __init__(self) -> None:
        self.timings_ms: dict[str, float] = {}
        self._start = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timings_ms[name] = round((time.perf_counter() - t0) * 1000.0, 1)

    def record(self, name: str, duration_ms: float) -> None:
        self.timings_ms[name] = round(duration_ms, 1)

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self._start) * 1000.0, 1)
