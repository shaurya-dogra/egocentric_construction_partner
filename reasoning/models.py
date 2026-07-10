"""Shared Tier 2 data models."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class WorkerIdentity:
    """Session-scoped worker identity built on top of tracker continuity."""

    worker_id: str
    current_track_id: int
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    track_history: list[int] = field(default_factory=list)
    embedding_centroid: Optional[list[float]] = None
    embedding_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolUsage:
    """Heuristic tool usage inference for a worker in the current scene."""

    worker_id: str
    worker_track_id: int
    tool_label: str
    tool_track_id: Optional[int]
    relation: str
    confidence: float
    active: bool
    zone_name: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskEstimate:
    """Likely task a worker is performing."""

    worker_id: str
    worker_track_id: int
    label: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    active_tool: Optional[str] = None
    zone_name: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HazardPrediction:
    """Near-future hazard prediction inferred from trajectory and scene state."""

    label: str
    confidence: float
    reason: str
    worker_id: Optional[str] = None
    zone_name: Optional[str] = None
    severity_hint: str = "warning"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProgressEstimate:
    """Coarse construction progress estimate."""

    summary: str
    confidence: float
    completed_items: list[str] = field(default_factory=list)
    pending_items: list[str] = field(default_factory=list)
    detected_structures: list[str] = field(default_factory=list)
    basis: str = "scene_state"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChecklistState:
    """Checklist automation state."""

    name: str
    completed: bool
    confidence: float
    evidence_frames: int = 0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalHit:
    """Chunk returned from the local knowledge store."""

    document_id: str
    title: str
    content: str
    score: float
    chunk_index: int
    source_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BlueprintContext:
    """Summarized blueprint/floor-plan metadata."""

    source_path: str
    asset_type: str
    summary: str
    extracted_text: str = ""
    image_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReasoningSnapshot:
    """Compact scene snapshot shipped to the reasoning model."""

    frame_number: int
    timestamp: float
    state_hash: str
    worker_id_map: dict[int, str]
    scene_graph: dict[str, Any]
    tool_usages: list[dict[str, Any]]
    task_candidates: list[dict[str, Any]]
    progress: dict[str, Any]
    predictions: list[dict[str, Any]]
    checklist: list[dict[str, Any]]
    spatial_memory: dict[str, Any]
    session_memory: dict[str, Any]
    blueprint_context: list[dict[str, Any]]
    document_strategy: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
