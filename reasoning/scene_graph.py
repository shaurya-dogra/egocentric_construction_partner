"""NetworkX-backed semantic scene graph utilities."""

from __future__ import annotations

import math
from typing import Optional

try:
    import networkx as nx
except Exception:  # pragma: no cover - lightweight fallback before deps are installed
    class _MiniDiGraph:
        def __init__(self) -> None:
            self._nodes: dict[str, dict] = {}
            self._edges: list[tuple[str, str, dict]] = []

        def add_node(self, node_id: str, **attrs) -> None:
            self._nodes[node_id] = attrs

        def add_edge(self, source: str, target: str, **attrs) -> None:
            self._edges.append((source, target, attrs))

        def has_node(self, node_id: str) -> bool:
            return node_id in self._nodes

        def nodes(self, data: bool = False):
            return self._nodes.items() if data else self._nodes.keys()

        def edges(self, data: bool = False):
            return self._edges if data else [(s, t) for s, t, _ in self._edges]

    class _MiniNX:
        DiGraph = _MiniDiGraph

    nx = _MiniNX()

from core.models import (
    FrameResult,
    KEYPOINT_LEFT_WRIST,
    KEYPOINT_RIGHT_WRIST,
    PoseData,
    TrackedObject,
)
from reasoning.models import ToolUsage


def _center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class SceneGraphBuilder:
    """Construct a compact scene graph once per frame for Tier 2 reasoning."""

    def __init__(self, config: Optional[dict] = None) -> None:
        config = config or {}
        self.tool_labels = set(
            config.get(
                "tool_labels",
                [
                    "cell_phone",
                    "drill",
                    "hammer",
                    "saw",
                    "measuring_tape",
                    "tool",
                ],
            )
        )
        self.structure_labels = set(
            config.get(
                "structure_labels",
                ["scaffold", "beam", "column", "wall", "door", "window", "ladder"],
            )
        )
        self.near_distance_px = float(config.get("near_distance_px", 180))
        self.active_motion_px = float(config.get("active_motion_px", 18))

    def infer_tool_usage(self, result: FrameResult, worker_id_map: dict[int, str]) -> list[ToolUsage]:
        """Infer whether detected tools are in active use versus merely present."""
        poses_by_track = {pose.person_track_id: pose for pose in result.poses}
        usages: list[ToolUsage] = []

        workers = {
            tid: obj
            for tid, obj in result.tracked_objects.items()
            if obj.class_name == "person" and obj.is_active
        }
        tool_detections = [
            det for det in result.detections if det.class_name in self.tool_labels
        ]

        for worker_track_id, worker in workers.items():
            worker_id = worker_id_map.get(worker_track_id, f"worker-{worker_track_id}")
            pose = poses_by_track.get(worker_track_id)
            worker_speed = self._worker_speed(worker)
            best_usage: Optional[ToolUsage] = None
            for tool in tool_detections:
                tool_center = _center(tool.bbox)
                dist = _distance(worker.center, tool_center)
                if dist > self.near_distance_px:
                    continue

                active = worker_speed >= self.active_motion_px
                relation = "holding" if self._near_wrist(tool_center, pose) else "nearby"
                confidence = max(0.2, 1.0 - (dist / max(self.near_distance_px, 1.0)))
                if relation == "holding":
                    confidence += 0.2
                    active = active or worker_speed >= self.active_motion_px * 0.6
                usage = ToolUsage(
                    worker_id=worker_id,
                    worker_track_id=worker_track_id,
                    tool_label=tool.class_name,
                    tool_track_id=tool.track_id,
                    relation=relation,
                    confidence=min(confidence, 0.95),
                    active=active,
                )
                if best_usage is None or usage.confidence > best_usage.confidence:
                    best_usage = usage
            if best_usage is not None:
                usages.append(best_usage)

        return usages

    def build(
        self,
        result: FrameResult,
        worker_id_map: dict[int, str],
        tool_usages: list[ToolUsage],
    ) -> nx.DiGraph:
        """Build a per-frame directed scene graph."""
        graph = nx.DiGraph()

        active_tracks = {
            tid: obj
            for tid, obj in result.tracked_objects.items()
            if obj.is_active
        }
        poses_by_track = {pose.person_track_id: pose for pose in result.poses}
        ppe_by_track = result.worker_ppe_states

        for track_id, obj in active_tracks.items():
            node_id = self._track_node_id(obj.class_name, track_id)
            attrs = {
                "kind": "worker" if obj.class_name == "person" else "entity",
                "label": obj.class_name,
                "track_id": track_id,
                "bbox": obj.bbox,
                "center": obj.center,
                "distance_meters": obj.distance_meters,
            }
            if obj.class_name == "person":
                attrs["worker_id"] = worker_id_map.get(track_id, f"worker-{track_id}")
                # PPE attributes omitted to save VLM resources
            graph.add_node(node_id, **attrs)

        for hazard in result.hazards:
            hazard_node = f"hazard:{hazard.hazard_id}"
            graph.add_node(
                hazard_node,
                kind="hazard",
                label=hazard.hazard_type,
                severity=hazard.severity.value,
                description=hazard.description,
                bbox=hazard.hazard_bbox,
            )
            if hazard.worker_track_id is not None:
                worker_node = self._track_node_id("person", hazard.worker_track_id)
                if graph.has_node(worker_node):
                    graph.add_edge(worker_node, hazard_node, relation="affected_by")

        for zone in result.active_zones:
            zone_node = f"zone:{zone.name}"
            graph.add_node(
                zone_node,
                kind="zone",
                label=zone.name,
                zone_type=zone.zone_type,
                polygon=zone.polygon,
            )

        for track_id, obj in active_tracks.items():
            if obj.class_name != "person":
                continue
            worker_node = self._track_node_id("person", track_id)
            for zone in result.active_zones:
                if self._point_in_polygon(obj.center, zone.polygon):
                    graph.add_edge(worker_node, f"zone:{zone.name}", relation="inside")

            pose = poses_by_track.get(track_id)
            for other_track_id, other in active_tracks.items():
                if other_track_id == track_id:
                    continue
                dist = _distance(obj.center, other.center)
                if dist <= self.near_distance_px:
                    graph.add_edge(
                        worker_node,
                        self._track_node_id(other.class_name, other_track_id),
                        relation="near",
                        distance_px=round(dist, 1),
                    )
                if pose and other.class_name in self.structure_labels and self._near_wrist(other.center, pose):
                    graph.add_edge(
                        worker_node,
                        self._track_node_id(other.class_name, other_track_id),
                        relation="touching_structure",
                    )

        for usage in tool_usages:
            worker_node = self._track_node_id("person", usage.worker_track_id)
            tool_node = self._track_node_id(usage.tool_label, usage.tool_track_id or -1)
            if not graph.has_node(tool_node):
                tool_det = next(
                    (det for det in result.detections if det.track_id == usage.tool_track_id and det.class_name == usage.tool_label),
                    None,
                )
                graph.add_node(
                    tool_node,
                    kind="tool",
                    label=usage.tool_label,
                    track_id=usage.tool_track_id,
                    bbox=tool_det.bbox if tool_det else None,
                    center=_center(tool_det.bbox) if tool_det else None,
                )
            graph.add_edge(
                worker_node,
                tool_node,
                relation="using" if usage.active else "holding",
                confidence=usage.confidence,
            )

        return graph

    @staticmethod
    def serialize(graph: nx.DiGraph) -> dict[str, list[dict]]:
        """Convert the graph to a compact JSON-serializable structure."""
        return {
            "nodes": [
                {"id": node_id, **attrs}
                for node_id, attrs in graph.nodes(data=True)
            ],
            "edges": [
                {"source": source, "target": target, **attrs}
                for source, target, attrs in graph.edges(data=True)
            ],
        }

    @staticmethod
    def _track_node_id(class_name: str, track_id: int) -> str:
        return f"{class_name}:{track_id}"

    @staticmethod
    def _worker_speed(worker: TrackedObject) -> float:
        history = worker.position_history[-5:]
        if len(history) < 2:
            return 0.0
        start_x, start_y, _ = history[0]
        end_x, end_y, _ = history[-1]
        return math.hypot(end_x - start_x, end_y - start_y)

    @staticmethod
    def _near_wrist(point: tuple[float, float], pose: Optional[PoseData]) -> bool:
        if pose is None or pose.keypoints is None or len(pose.keypoints) < 11:
            return False
        for index in (KEYPOINT_LEFT_WRIST, KEYPOINT_RIGHT_WRIST):
            wrist = pose.keypoints[index]
            if len(wrist) < 3 or wrist[2] < 0.2:
                continue
            if _distance(point, (float(wrist[0]), float(wrist[1]))) <= 80:
                return True
        return False

    @staticmethod
    def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[int, int]]) -> bool:
        x, y = point
        inside = False
        for idx in range(len(polygon)):
            x1, y1 = polygon[idx]
            x2, y2 = polygon[(idx + 1) % len(polygon)]
            intersects = ((y1 > y) != (y2 > y)) and (
                x < ((x2 - x1) * (y - y1) / ((y2 - y1) or 1e-6) + x1)
            )
            if intersects:
                inside = not inside
        return inside
