"""Structured session memory for Tier 2 reasoning."""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Optional


class SessionMemory:
    """Stores concise, queryable observations across the current session."""

    def __init__(self, max_items: int = 200) -> None:
        self._workers_seen: dict[str, dict[str, Any]] = {}
        self._tasks: deque[dict[str, Any]] = deque(maxlen=max_items)
        self._predictions: deque[dict[str, Any]] = deque(maxlen=max_items)
        self._progress: deque[dict[str, Any]] = deque(maxlen=max_items)
        self._checklist: dict[str, dict[str, Any]] = {}
        self._hazards: deque[dict[str, Any]] = deque(maxlen=max_items)

    def record_worker(self, worker_id: str, track_id: int, zone: Optional[str], timestamp: float) -> None:
        worker = self._workers_seen.setdefault(
            worker_id,
            {
                "worker_id": worker_id,
                "first_seen": timestamp,
                "track_ids": [],
            },
        )
        worker["last_seen"] = timestamp
        worker["zone"] = zone
        if track_id not in worker["track_ids"]:
            worker["track_ids"].append(track_id)

    def record_task(self, payload: dict[str, Any]) -> None:
        self._tasks.append({"timestamp": time.time(), **payload})

    def record_prediction(self, payload: dict[str, Any]) -> None:
        self._predictions.append({"timestamp": time.time(), **payload})

    def record_progress(self, payload: dict[str, Any]) -> None:
        self._progress.append({"timestamp": time.time(), **payload})

    def record_hazard(self, payload: dict[str, Any]) -> None:
        self._hazards.append({"timestamp": time.time(), **payload})

    def set_checklist_item(self, name: str, payload: dict[str, Any]) -> None:
        self._checklist[name] = {"timestamp": time.time(), **payload}

    def export_context(self, limit: int = 5) -> dict[str, Any]:
        return {
            "workers_seen": list(self._workers_seen.values()),
            "recent_tasks": list(self._tasks)[-limit:],
            "recent_predictions": list(self._predictions)[-limit:],
            "recent_progress": list(self._progress)[-limit:],
            "recent_hazards": list(self._hazards)[-limit:],
            "checklist": list(self._checklist.values()),
        }
