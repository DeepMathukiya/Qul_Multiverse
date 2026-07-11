"""In-memory store of completed inspection results (opaque JSON dicts)."""

from __future__ import annotations

import threading
from collections import OrderedDict

MAX_RESULTS = 50


class ResultStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._results: OrderedDict[str, dict] = OrderedDict()

    def add(self, result: dict) -> None:
        inspection_id = result.get("inspection_id", "")
        if not inspection_id:
            return
        with self._lock:
            self._results[inspection_id] = result
            while len(self._results) > MAX_RESULTS:
                self._results.popitem(last=False)

    def get(self, inspection_id: str) -> dict | None:
        with self._lock:
            return self._results.get(inspection_id)

    def latest(self) -> dict | None:
        with self._lock:
            if not self._results:
                return None
            return next(reversed(self._results.values()))

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(reversed(self._results.keys()))


result_store = ResultStore()
