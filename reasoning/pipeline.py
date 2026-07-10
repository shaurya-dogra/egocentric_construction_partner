"""Tier 2 reasoning coordinator built on top of Tier 1 scene state."""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any, Optional

import numpy as np

from core.models import FrameResult, TrackedObject
from reasoning.blueprint_store import BlueprintStore
from reasoning.document_store import DocumentStore
from reasoning.models import (
    ChecklistState,
    HazardPrediction,
    ProgressEstimate,
    ReasoningSnapshot,
    TaskEstimate,
    WorkerIdentity,
)
from reasoning.scene_graph import SceneGraphBuilder
from reasoning.session_memory import SessionMemory
from reasoning.spatial_memory import SpatialMemory


class ReasoningCoordinator:
    """Builds structured Tier 2 context from the fast Tier 1 pipeline."""

    def __init__(self, config: Optional[dict] = None, event_logger=None) -> None:
        config = config or {}
        self.config = config
        self.event_logger = event_logger
        self.reason_every_seconds = float(config.get("reason_every_seconds", 4.0))
        self.state_change_cooldown = float(config.get("state_change_cooldown_seconds", 1.5))
        self.checklist_hold_frames = int(config.get("checklist_hold_frames", 6))
        self.expected_items = list(config.get("expected_items", []))
        self.scene_graph_builder = SceneGraphBuilder(config.get("scene_graph", {}))
        self.spatial_memory = SpatialMemory(config.get("spatial_memory", {}))
        self.session_memory = SessionMemory(max_items=int(config.get("memory_items", 200)))
        self.document_store = DocumentStore(config.get("documents", {}))
        self.blueprint_store = BlueprintStore(config.get("blueprints", {}))
        self._worker_profiles: dict[str, WorkerIdentity] = {}
        self._track_to_worker: dict[int, str] = {}
        self._next_worker_id = 1
        self._last_reasoned_at = 0.0
        self._last_state_hash = ""
        self._last_state_change_at = 0.0
        self._checklist_counters: dict[str, int] = {}

    def build_snapshot(self, result: FrameResult, frame_shape: tuple[int, int, int]) -> ReasoningSnapshot:
        frame_h, frame_w = frame_shape[:2]
        self.spatial_memory.frame_width = frame_w
        self.spatial_memory.frame_height = frame_h
        worker_id_map = self._identify_workers(result)
        tool_usages = self.scene_graph_builder.infer_tool_usage(result, worker_id_map)
        graph = self.scene_graph_builder.build(result, worker_id_map, tool_usages)
        graph_payload = self.scene_graph_builder.serialize(graph)
        tasks = self._infer_tasks(result, worker_id_map, tool_usages, graph_payload)
        progress = self._estimate_progress(result, tasks)
        predictions = self._predict_hazards(result, worker_id_map, tasks)
        checklist = self._update_checklist(result, progress, tasks)
        self._update_memories(result, worker_id_map, tasks, progress, predictions, checklist)
        state_hash = self._state_hash(graph_payload, tasks, progress, predictions, checklist)
        if state_hash != self._last_state_hash:
            self._last_state_hash = state_hash
            self._last_state_change_at = time.time()
        return ReasoningSnapshot(
            frame_number=result.frame_number,
            timestamp=result.timestamp,
            state_hash=state_hash,
            worker_id_map=worker_id_map,
            scene_graph=graph_payload,
            tool_usages=[usage.to_dict() for usage in tool_usages],
            task_candidates=[task.to_dict() for task in tasks],
            progress=progress.to_dict(),
            predictions=[prediction.to_dict() for prediction in predictions],
            checklist=[item.to_dict() for item in checklist],
            spatial_memory=self.spatial_memory.export_context(),
            session_memory=self.session_memory.export_context(),
            blueprint_context=self.blueprint_store.prompt_context(),
            document_strategy=self.document_store.strategy(),
        )

    def should_schedule_reasoning(self, snapshot: ReasoningSnapshot, force: bool = False) -> bool:
        now = time.time()
        if force:
            self._last_reasoned_at = now
            return True
        if now - self._last_reasoned_at < self.reason_every_seconds:
            return False
        # If the state hasn't settled, we wait, but we force an update if we've been waiting too long.
        if now - self._last_state_change_at < self.state_change_cooldown:
            if now - self._last_reasoned_at < (self.reason_every_seconds * 1.5):
                return False
        self._last_reasoned_at = now
        return True

    def retrieval_context(self, query: str) -> dict[str, Any]:
        return self.document_store.build_prompt_context(query)

    def blueprint_context(self) -> list[dict]:
        return self.blueprint_store.prompt_context()

    def blueprint_images(self, limit: int = 2) -> list[str]:
        return self.blueprint_store.image_paths(limit=limit)

    def fast_path_answer(self, question: str, snapshot: ReasoningSnapshot) -> Optional[str]:
        lower = question.lower()
        if "last hazard" in lower:
            recent = snapshot.spatial_memory.get("recent_hazards", [])
            if recent:
                last = recent[-1]
                return f"Last hazard was in {last['zone']}: {last['description']}."
            return "No hazards have been logged in this session."
        if "which zone" in lower and "crane" in lower:
            zone = self.spatial_memory.last_known_zone("crane")
            return f"The crane was last seen in {zone}." if zone else "No crane location is stored yet."
        if "near me" in lower or "what's near" in lower:
            workers = snapshot.task_candidates
            if workers:
                top = workers[0]
                return f"{top['worker_id']} appears to be {top['label']}."
            return "I do not have an active task estimate yet."
        return None

    def log_reasoning_event(
        self,
        frame_number: int,
        event_kind: str,
        summary: str,
        confidence: Optional[float] = None,
        worker_id: Optional[str] = None,
        zone_name: Optional[str] = None,
        hazard_type: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        if self.event_logger is None:
            return
        self.event_logger.log_reasoning_event(
            frame_number=frame_number,
            event_kind=event_kind,
            summary=summary,
            confidence=confidence,
            worker_id=worker_id,
            zone_name=zone_name,
            hazard_type=hazard_type,
            payload=payload,
        )

    def _identify_workers(self, result: FrameResult) -> dict[int, str]:
        worker_map: dict[int, str] = {}
        for track_id, obj in result.tracked_objects.items():
            if obj.class_name != "person" or not obj.is_active:
                continue
            worker_id = self._track_to_worker.get(track_id)
            if worker_id is None:
                worker_id = self._match_existing_worker(obj)
            if worker_id is None:
                worker_id = f"worker-{self._next_worker_id}"
                self._next_worker_id += 1
            profile = self._worker_profiles.setdefault(
                worker_id,
                WorkerIdentity(worker_id=worker_id, current_track_id=track_id),
            )
            profile.current_track_id = track_id
            profile.last_seen = result.timestamp
            if track_id not in profile.track_history:
                profile.track_history.append(track_id)
            if obj.appearance_embedding:
                centroid = np.array(profile.embedding_centroid or obj.appearance_embedding, dtype=float)
                vector = np.array(obj.appearance_embedding, dtype=float)
                samples = max(profile.embedding_samples, 1)
                centroid = ((centroid * samples) + vector) / (samples + 1)
                profile.embedding_centroid = centroid.tolist()
                profile.embedding_samples = samples + 1
            self._track_to_worker[track_id] = worker_id
            worker_map[track_id] = worker_id
        return worker_map

    def _match_existing_worker(self, obj: TrackedObject) -> Optional[str]:
        if not obj.appearance_embedding:
            return None
        candidate = np.array(obj.appearance_embedding, dtype=float)
        best_id: Optional[str] = None
        best_score = 0.0
        for worker_id, profile in self._worker_profiles.items():
            if not profile.embedding_centroid:
                continue
            centroid = np.array(profile.embedding_centroid, dtype=float)
            score = self._cosine(candidate, centroid)
            if score > best_score:
                best_score = score
                best_id = worker_id
        return best_id if best_score >= 0.85 else None

    def _infer_tasks(
        self,
        result: FrameResult,
        worker_id_map: dict[int, str],
        tool_usages: list[Any],
        graph_payload: dict[str, Any],
    ) -> list[TaskEstimate]:
        tasks: list[TaskEstimate] = []
        worker_zone = {
            edge["source"]: edge["target"].split("zone:", 1)[1]
            for edge in graph_payload["edges"]
            if edge.get("relation") == "inside" and edge["source"].startswith("person:")
        }
        for track_id, worker_id in worker_id_map.items():
            tool_usage = next((usage for usage in tool_usages if usage.worker_track_id == track_id), None)
            label = "observing"
            confidence = 0.35
            evidence = ["worker tracked in scene"]
            active_tool = None
            if tool_usage:
                active_tool = tool_usage.tool_label
                evidence.append(f"{tool_usage.relation}_{tool_usage.tool_label}")
                label, confidence = self._label_task_from_tool(tool_usage.tool_label, tool_usage.active)
            elif self._worker_motion(track_id, result.tracked_objects) > 60:
                label = "carrying_material"
                confidence = 0.45
                evidence.append("sustained_worker_motion")
            if any(h.hazard_type == "zone_proximity" and h.worker_track_id == track_id for h in result.hazards):
                evidence.append("working_inside_marked_zone")
                confidence = min(0.95, confidence + 0.1)
            tasks.append(
                TaskEstimate(
                    worker_id=worker_id,
                    worker_track_id=track_id,
                    label=label,
                    confidence=confidence,
                    evidence=evidence,
                    active_tool=active_tool,
                    zone_name=worker_zone.get(f"person:{track_id}"),
                )
            )
        return tasks

    def _estimate_progress(self, result: FrameResult, tasks: list[TaskEstimate]) -> ProgressEstimate:
        detected = sorted(
            {
                det.class_name
                for det in result.detections
                if det.class_name not in {"person", "car", "truck", "bus", "cell_phone"}
            }
        )
        expected = set(self.expected_items)
        completed = sorted(item for item in expected if item in detected)
        pending = sorted(item for item in expected if item not in detected)
        if expected:
            confidence = len(completed) / max(len(expected), 1)
            summary = f"{len(completed)} of {len(expected)} expected items are currently visible."
        else:
            confidence = 0.3 if detected else 0.1
            summary = "Progress is based on coarse scene-state presence, not precise schedule completion."
        if any(task.label != "observing" for task in tasks):
            summary += " Active work is in progress."
        return ProgressEstimate(
            summary=summary,
            confidence=round(confidence, 2),
            completed_items=completed,
            pending_items=pending,
            detected_structures=detected,
            basis="presence_absence_proxy",
        )

    def _predict_hazards(
        self,
        result: FrameResult,
        worker_id_map: dict[int, str],
        tasks: list[TaskEstimate],
    ) -> list[HazardPrediction]:
        predictions: list[HazardPrediction] = []
        task_by_track = {task.worker_track_id: task for task in tasks}
        for hazard in result.hazards:
            if hazard.hazard_type == "vehicle_proximity" and hazard.worker_track_id in worker_id_map:
                worker_obj = result.tracked_objects.get(hazard.worker_track_id)
                speed = self._worker_motion(hazard.worker_track_id, result.tracked_objects)
                if worker_obj and speed > 25:
                    predictions.append(
                        HazardPrediction(
                            label="collision_risk_worsening",
                            confidence=0.72,
                            reason="Worker remains close to a vehicle while moving through the approach corridor.",
                            worker_id=worker_id_map[hazard.worker_track_id],
                            severity_hint="danger",
                        )
                    )
            if hazard.hazard_type == "zone_proximity" and hazard.worker_track_id in worker_id_map:
                task = task_by_track.get(hazard.worker_track_id)
                if task and task.label == "carrying_material":
                    predictions.append(
                        HazardPrediction(
                            label="trip_or_load_drop_risk",
                            confidence=0.63,
                            reason="Worker is moving through a marked zone while carrying material.",
                            worker_id=task.worker_id,
                            zone_name=hazard.zone_name,
                            severity_hint="warning",
                        )
                    )
        return predictions

    def _update_checklist(
        self,
        result: FrameResult,
        progress: ProgressEstimate,
        tasks: list[TaskEstimate],
    ) -> list[ChecklistState]:
        states: list[ChecklistState] = []
        visible = set(progress.detected_structures)
        for name in self.expected_items:
            count = self._checklist_counters.get(name, 0)
            count = count + 1 if name in visible else 0
            self._checklist_counters[name] = count
            completed = count >= self.checklist_hold_frames
            states.append(
                ChecklistState(
                    name=name,
                    completed=completed,
                    confidence=min(1.0, count / max(self.checklist_hold_frames, 1)),
                    evidence_frames=count,
                    notes="Auto-completed only after repeated confirmation across frames.",
                )
            )
        return states

    def _update_memories(
        self,
        result: FrameResult,
        worker_id_map: dict[int, str],
        tasks: list[TaskEstimate],
        progress: ProgressEstimate,
        predictions: list[HazardPrediction],
        checklist: list[ChecklistState],
    ) -> None:
        for track_id, worker_id in worker_id_map.items():
            zone = self.spatial_memory.observe_entity(
                entity_key=worker_id,
                center=result.tracked_objects[track_id].center,
                timestamp=result.timestamp,
            )
            self.session_memory.record_worker(worker_id, track_id, zone, result.timestamp)
        for hazard in result.hazards:
            center = None
            if hazard.hazard_bbox:
                x1, y1, x2, y2 = hazard.hazard_bbox
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            self.spatial_memory.observe_hazard(
                hazard_id=hazard.hazard_id,
                center=center,
                description=hazard.description,
                timestamp=result.timestamp,
                zone_name=hazard.zone_name,
            )
            self.session_memory.record_hazard(
                {
                    "hazard_id": hazard.hazard_id,
                    "hazard_type": hazard.hazard_type,
                    "description": hazard.description,
                    "zone_name": hazard.zone_name,
                }
            )
        for task in tasks:
            self.session_memory.record_task(task.to_dict())
            self.log_reasoning_event(
                frame_number=result.frame_number,
                event_kind="task",
                summary=f"{task.worker_id} likely performing {task.label}",
                confidence=task.confidence,
                worker_id=task.worker_id,
                zone_name=task.zone_name,
                payload=task.to_dict(),
            )
        self.session_memory.record_progress(progress.to_dict())
        self.log_reasoning_event(
            frame_number=result.frame_number,
            event_kind="progress",
            summary=progress.summary,
            confidence=progress.confidence,
            payload=progress.to_dict(),
        )
        for prediction in predictions:
            self.session_memory.record_prediction(prediction.to_dict())
            self.log_reasoning_event(
                frame_number=result.frame_number,
                event_kind="prediction",
                summary=prediction.label,
                confidence=prediction.confidence,
                worker_id=prediction.worker_id,
                zone_name=prediction.zone_name,
                payload=prediction.to_dict(),
            )
        for item in checklist:
            self.session_memory.set_checklist_item(item.name, item.to_dict())
            if item.completed:
                self.log_reasoning_event(
                    frame_number=result.frame_number,
                    event_kind="checklist",
                    summary=f"Checklist item completed: {item.name}",
                    confidence=item.confidence,
                    payload=item.to_dict(),
                )

    def _state_hash(
        self,
        graph_payload: dict[str, Any],
        tasks: list[TaskEstimate],
        progress: ProgressEstimate,
        predictions: list[HazardPrediction],
        checklist: list[ChecklistState],
    ) -> str:
        # Strip continuous data that changes every frame to allow stable hashes
        clean_nodes = []
        for node in graph_payload.get("nodes", []):
            clean_nodes.append({k: v for k, v in node.items() if k not in ("bbox", "center")})
        clean_edges = []
        for edge in graph_payload.get("edges", []):
            clean_edges.append({k: v for k, v in edge.items() if k not in ("distance_px",)})
            
        payload = json.dumps(
            {
                "graph": {"nodes": clean_nodes, "edges": clean_edges},
                "tasks": [task.to_dict() for task in tasks],
                "progress": progress.to_dict(),
                "predictions": [prediction.to_dict() for prediction in predictions],
                "checklist": [item.to_dict() for item in checklist],
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _worker_motion(track_id: int, tracked_objects: dict[int, TrackedObject]) -> float:
        worker = tracked_objects.get(track_id)
        if worker is None or len(worker.position_history) < 2:
            return 0.0
        history = worker.position_history[-10:]
        start_x, start_y, _ = history[0]
        end_x, end_y, _ = history[-1]
        return math.hypot(end_x - start_x, end_y - start_y)

    @staticmethod
    def _label_task_from_tool(tool_label: str, active: bool) -> tuple[str, float]:
        if tool_label in {"drill", "hammer", "saw"}:
            return ("tool_operation" if active else "tool_positioning", 0.72 if active else 0.58)
        if tool_label in {"measuring_tape"}:
            return ("measuring", 0.68)
        return ("tool_handling" if active else "tool_carrying", 0.55 if active else 0.48)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
