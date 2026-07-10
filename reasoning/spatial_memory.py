"""Lightweight 2D spatial memory for a fixed-camera construction scene."""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Optional


class SpatialMemory:
    """Tracks where entities and hazards were last observed in scene zones."""

    def __init__(self, config: Optional[dict] = None) -> None:
        config = config or {}
        self.frame_width = int(config.get("frame_width", 1280))
        self.frame_height = int(config.get("frame_height", 720))
        self.grid_rows = int(config.get("grid_rows", 3))
        self.grid_cols = int(config.get("grid_cols", 3))
        self.history_limit = int(config.get("history_limit", 200))
        self._entity_history: dict[str, deque[dict[str, Any]]] = {}
        self._hazard_history: deque[dict[str, Any]] = deque(maxlen=self.history_limit)

    def observe_entity(
        self,
        entity_key: str,
        center: tuple[float, float],
        timestamp: float,
        zone_name: Optional[str] = None,
    ) -> str:
        zone = zone_name or self._grid_zone(center)
        history = self._entity_history.setdefault(entity_key, deque(maxlen=self.history_limit))
        history.append({"zone": zone, "center": center, "timestamp": timestamp})
        return zone

    def observe_hazard(
        self,
        hazard_id: str,
        center: Optional[tuple[float, float]],
        description: str,
        timestamp: float,
        zone_name: Optional[str] = None,
    ) -> None:
        zone = zone_name or (self._grid_zone(center) if center else "unknown")
        self._hazard_history.append(
            {
                "hazard_id": hazard_id,
                "zone": zone,
                "description": description,
                "timestamp": timestamp,
            }
        )

    def last_known_zone(self, entity_key: str) -> Optional[str]:
        history = self._entity_history.get(entity_key)
        if not history:
            return None
        return history[-1]["zone"]

    def recent_hazards(self, limit: int = 5) -> list[dict[str, Any]]:
        return list(self._hazard_history)[-limit:]

    def export_context(self) -> dict[str, Any]:
        entities = {}
        for key, history in self._entity_history.items():
            if not history:
                continue
            entities[key] = history[-1]
        return {
            "mode": "2d_fixed_camera_zone_memory",
            "entities": entities,
            "recent_hazards": self.recent_hazards(),
        }

    def _grid_zone(self, center: Optional[tuple[float, float]]) -> str:
        if center is None:
            return "unknown"
        x, y = center
        col = min(self.grid_cols - 1, max(0, int((x / max(self.frame_width, 1)) * self.grid_cols)))
        row = min(self.grid_rows - 1, max(0, int((y / max(self.frame_height, 1)) * self.grid_rows)))
        vertical = ("top", "mid", "bottom")[min(row, 2)]
        horizontal = ("left", "center", "right")[min(col, 2)]
        return f"{vertical}-{horizontal}"
