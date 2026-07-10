"""YOLO26 detection wrapper with dual-model support (COCO + PPE).

Runs a general COCO-pretrained model and an optional PPE-specific model,
merging results into a unified list of ``Detection`` objects.  Supports
ByteTrack-based tracking via the Ultralytics ``model.track()`` API.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional
import logging

import cv2
import numpy as np
import re

# Suppress the misleading ultralytics warning when ReID initializes
try:
    from ultralytics.utils import LOGGER as ULTRALYTICS_LOGGER
    class _UltralyticsSourceFilter(logging.Filter):
        def filter(self, record):
            return "'source' is missing" not in record.getMessage()
    ULTRALYTICS_LOGGER.addFilter(_UltralyticsSourceFilter())
except ImportError:
    pass

from core.models import Detection

logger = logging.getLogger(__name__)

# ── COCO class filter ─────────────────────────────────────────

# Relevant COCO class IDs for job-site safety
_COCO_FILTER: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    67: "cell_phone",
}

# ── PPE name normalisation ───────────────────────────────────

# Regex patterns → canonical name
_PPE_NAME_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)^no[_\-\s]*hard[_\-\s]*hat"), "no_hardhat"),
    (re.compile(r"(?i)^no[_\-\s]*safety[_\-\s]*vest"), "no_vest"),
    (re.compile(r"(?i)^no[_\-\s]*vest"), "no_vest"),
    (re.compile(r"(?i)^no[_\-\s]*goggle"), "no_goggles"),
    (re.compile(r"(?i)^no[_\-\s]*glove"), "no_gloves"),
    (re.compile(r"(?i)^no[_\-\s]*boot"), "no_boots"),
    (re.compile(r"(?i)^no[_\-\s]*mask"), "no_mask"),
    (re.compile(r"(?i)hard[_\-\s]*hat"), "hardhat"),
    (re.compile(r"(?i)^helmet"), "hardhat"),
    (re.compile(r"(?i)safety[_\-\s]*vest"), "vest"),
    (re.compile(r"(?i)^vest"), "vest"),
    (re.compile(r"(?i)^goggle"), "goggles"),
    (re.compile(r"(?i)^glove"), "gloves"),
    (re.compile(r"(?i)^boot"), "boots"),
    (re.compile(r"(?i)^mask"), "mask"),
]


def _normalise_ppe_name(raw: str) -> str:
    """Map a raw PPE class name to a canonical name."""
    for pattern, canonical in _PPE_NAME_MAP:
        if pattern.search(raw):
            return canonical
    # Fallback: lowercase, strip whitespace
    return raw.strip().lower().replace(" ", "_").replace("-", "_")


class Detector:
    """Dual-model YOLO26 detector (general COCO + optional PPE).

    Args:
        general_model_path: Path or name for the COCO-pretrained model
            (e.g. ``"yolo26n.pt"``).  Ultralytics auto-downloads on first use.
        ppe_model_path: Path to a PPE-specific model, or ``None`` to skip.
        device: Compute device string (``"cpu"``, ``"mps"``, etc.).
        general_conf: Confidence threshold for the general model.
        general_iou: IoU threshold for NMS on the general model.
        ppe_conf: Confidence threshold for the PPE model.
        ppe_iou: IoU threshold for NMS on the PPE model.
    """

    def __init__(
        self,
        general_model_path: str = "yolo26n.pt",
        ppe_model_path: Optional[str] = None,
        device: str = "cpu",
        tracker_config: str = "bytetrack.yaml",
        general_conf: float = 0.35,
        general_iou: float = 0.45,
        ppe_conf: float = 0.30,
        ppe_iou: float = 0.45,
        ppe_api_key: Optional[str] = None,
        tool_model_path: Optional[str] = None,
        tool_conf: float = 0.25,
        tool_iou: float = 0.45,
        tool_classes: Optional[list[str]] = None,
    ) -> None:
        from ultralytics import YOLO

        self._device = device
        self._tracker_config = tracker_config
        self._general_conf = general_conf
        self._general_iou = general_iou
        self._ppe_conf = ppe_conf
        self._ppe_iou = ppe_iou
        self._tool_conf = tool_conf
        self._tool_iou = tool_iou
        self._tool_classes = tool_classes or []

        # ── Load general (COCO) model ─────────────────────────
        self._general_model = None
        if general_model_path is not None and general_model_path.lower() not in ("none", "null", ""):
            logger.info(
                "Loading general model '%s' on device '%s'…",
                general_model_path,
                device,
            )
            self._general_model = YOLO(general_model_path)
            logger.info("General model loaded — classes: %d", len(self._general_model.names))
        else:
            logger.info("General model disabled — relying entirely on consolidated PPE model")

        # ── Load tool model (optional YOLO-World) ─────────────
        self._tool_model = None
        if tool_model_path is not None and self._tool_classes:
            logger.info(
                "Loading YOLO-World tool model '%s' on device '%s'…",
                tool_model_path,
                device,
            )
            from ultralytics import YOLOWorld
            self._tool_model = YOLOWorld(tool_model_path)
            self._tool_model.set_classes(self._tool_classes)
            logger.info("YOLO-World tool model loaded successfully with classes: %s", self._tool_classes)

        # ── Load PPE model (optional) ─────────────────────────
        self._ppe_model = None
        self._ppe_model_is_roboflow = False
        if ppe_model_path is not None:
            logger.info(
                "Loading PPE model '%s' on device '%s'…",
                ppe_model_path,
                device,
            )
            # Check if it looks like a Roboflow model ID (e.g. project/version)
            if "/" in ppe_model_path and not ppe_model_path.endswith(".pt") and not ppe_model_path.startswith("hf:"):
                import os
                import sys
                if ppe_api_key:
                    os.environ["ROBOFLOW_API_KEY"] = ppe_api_key
                try:
                    from inference import get_model
                except (ImportError, ModuleNotFoundError):
                    logger.critical(
                        "\n" + "="*85 + "\n"
                        "❌ ERROR: The 'inference' package is required to load Roboflow models but was not found.\n"
                        "Since 'inference' only supports Python <= 3.12 (and you are currently running Python 3.13),\n"
                        "you must run the safety copilot using Python 3.10:\n\n"
                        f"    .venv/bin/python3.10 main.py --source {sys.argv[sys.argv.index('--source') + 1] if '--source' in sys.argv else 'webcam'}\n"
                        + "="*85 + "\n"
                    )
                    sys.exit(1)
                self._ppe_model = get_model(model_id=ppe_model_path)
                self._ppe_model_is_roboflow = True
                logger.info("PPE model loaded via Roboflow Inference API")
            elif ppe_model_path.startswith("hf:"):
                try:
                    parts = ppe_model_path.split(":", 2)
                    repo_id = parts[1]
                    filename = parts[2]
                    logger.info("Downloading model '%s' from Hugging Face repo '%s'…", filename, repo_id)
                    from huggingface_hub import hf_hub_download
                    downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename)
                    self._ppe_model = YOLO(downloaded_path)
                    logger.info(
                        "PPE model loaded from Hugging Face — classes: %s",
                        {i: n for i, n in self._ppe_model.names.items()},
                    )
                except Exception as e:
                    logger.critical("Failed to download model from Hugging Face: %s", e)
                    import sys
                    sys.exit(1)
            else:
                self._ppe_model = YOLO(ppe_model_path)
                logger.info(
                    "PPE model loaded — classes: %s",
                    {i: n for i, n in self._ppe_model.names.items()},
                )
        else:
            logger.info("No PPE model specified — PPE detection disabled")

    # ── Public API ────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run all active models on *frame* and return merged detections."""
        detections: list[Detection] = []

        if self._general_model is None:
            if self._ppe_model is not None and not self._ppe_model_is_roboflow:
                results = self._ppe_model.predict(
                    source=frame,
                    device=self._device,
                    conf=self._ppe_conf,
                    iou=self._ppe_iou,
                    verbose=False,
                )
                detections.extend(self._parse_consolidated_results(results, with_tracking=False))
            elif self._ppe_model is not None:
                detections.extend(self._run_ppe(frame))
        else:
            detections.extend(self._run_general(frame))
            detections.extend(self._run_ppe(frame))

        # Run tool detection (YOLO-World) if active
        if self._tool_model is not None:
            tool_results = self._tool_model.predict(
                source=frame,
                device=self._device,
                conf=self._tool_conf,
                iou=self._tool_iou,
                verbose=False,
            )
            detections.extend(self._parse_tool_results(tool_results, with_tracking=False))

        return self._deduplicate(detections)

    def detect_and_track(self, frame: np.ndarray) -> list[Detection]:
        """Run models with tracking enabled where supported."""
        detections: list[Detection] = []

        if self._general_model is None:
            if self._ppe_model is not None and not self._ppe_model_is_roboflow:
                results = self._ppe_model.track(
                    source=frame,
                    device=self._device,
                    conf=self._ppe_conf,
                    iou=self._ppe_iou,
                    verbose=False,
                    persist=True,
                    tracker=self._tracker_config,
                )
                detections.extend(self._parse_consolidated_results(results, with_tracking=True))
            elif self._ppe_model is not None:
                detections.extend(self._run_ppe(frame))
        else:
            detections.extend(self._run_general_tracked(frame))
            detections.extend(self._run_ppe(frame))

        # Run tool tracking if active
        if self._tool_model is not None:
            tool_results = self._tool_model.track(
                source=frame,
                device=self._device,
                conf=self._tool_conf,
                iou=self._tool_iou,
                verbose=False,
                persist=True,
                tracker=self._tracker_config,
            )
            detections.extend(self._parse_tool_results(tool_results, with_tracking=True))

        return self._deduplicate(detections)

    def _parse_consolidated_results(
        self, results, *, with_tracking: bool = False
    ) -> list[Detection]:
        """Extract both General (coco) and PPE detections from a single consolidated model run."""
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                raw_name = self._ppe_model.names.get(cls_id, f"class_{cls_id}")
                canonical = _normalise_ppe_name(raw_name)
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                bbox = (int(x1), int(y1), int(x2), int(y2))

                track_id: Optional[int] = None
                if with_tracking and boxes.id is not None:
                    track_id = int(boxes.id[i].item())

                # Determine if this class is a worker/vehicle or a PPE gear item
                raw_lower = raw_name.lower()
                is_person_or_vehicle = "person" in raw_lower or any(
                    v in raw_lower for v in ("vehicle", "machinery", "car", "truck", "bus", "forklift")
                )

                if is_person_or_vehicle:
                    appearance_embedding = self._extract_embedding(result, i)
                    detections.append(
                        Detection(
                            class_name=canonical,
                            confidence=conf,
                            bbox=bbox,
                            track_id=track_id,
                            is_ppe=False,
                            model_source="coco",
                            appearance_embedding=appearance_embedding,
                        )
                    )
                else:
                    detections.append(
                        Detection(
                            class_name=canonical,
                            confidence=conf,
                            bbox=bbox,
                            track_id=None,
                            is_ppe=True,
                            model_source="ppe",
                        )
                    )
        return detections

    def _deduplicate(self, detections: list[Detection]) -> list[Detection]:
        """Remove overlapping duplicate detections across general and PPE models."""
        if not detections:
            return []

        # Sort prioritizing general model (coco) and tool detections (sources of tracking), then confidence
        sorted_dets = sorted(
            detections,
            key=lambda d: (1 if d.model_source in ("coco", "tool") else 0, d.confidence),
            reverse=True
        )
        kept: list[Detection] = []

        for det in sorted_dets:
            overlap = False
            for existing in kept:
                # Check if classes are compatible
                if self._is_compatible(det.class_name, existing.class_name):
                    # Check box overlap (IoU > 0.45)
                    if self._box_iou(det.bbox, existing.bbox) > 0.45:
                        overlap = True
                        break
            if not overlap:
                kept.append(det)

        return kept

    @staticmethod
    def _is_compatible(cls1: str, cls2: str) -> bool:
        if cls1 == cls2:
            return True
        vehicles = {"car", "truck", "bus", "motorcycle", "forklift", "machinery", "vehicle"}
        return cls1 in vehicles and cls2 in vehicles

    @staticmethod
    def _box_iou(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        if union_area <= 0:
            return 0.0
        return float(inter_area) / float(union_area)

    # ── Internal: general model ───────────────────────────────

    def _run_general(self, frame: np.ndarray) -> list[Detection]:
        """Run the general COCO model (predict, no tracking)."""
        results = self._general_model.predict(
            source=frame,
            device=self._device,
            conf=self._general_conf,
            iou=self._general_iou,
            verbose=False,
            classes=list(_COCO_FILTER.keys()),
        )
        return self._parse_general_results(results)

    def _run_general_tracked(self, frame: np.ndarray) -> list[Detection]:
        """Run the general COCO model with configured tracking."""
        results = self._general_model.track(
            source=frame,
            device=self._device,
            conf=self._general_conf,
            iou=self._general_iou,
            verbose=False,
            classes=list(_COCO_FILTER.keys()),
            persist=True,
            tracker=self._tracker_config,
        )
        return self._parse_general_results(results, with_tracking=True)

    def _parse_general_results(
        self, results, *, with_tracking: bool = False
    ) -> list[Detection]:
        """Extract ``Detection`` objects from Ultralytics results."""
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if cls_id not in _COCO_FILTER:
                    continue

                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                bbox = (int(x1), int(y1), int(x2), int(y2))

                track_id: Optional[int] = None
                if with_tracking and boxes.id is not None:
                    track_id = int(boxes.id[i].item())
                appearance_embedding = self._extract_embedding(result, i)

                detections.append(
                    Detection(
                        class_name=_COCO_FILTER[cls_id],
                        confidence=conf,
                        bbox=bbox,
                        track_id=track_id,
                        is_ppe=False,
                        model_source="coco",
                        appearance_embedding=appearance_embedding,
                    )
                )
        return detections

    @staticmethod
    def _extract_embedding(result, index: int) -> Optional[list[float]]:
        """Best-effort extraction of tracker appearance features.

        Ultralytics tracker backends do not expose a stable public accessor for
        ReID features across versions. Tier 2 consumes them opportunistically
        when present and falls back to track continuity when they are absent.
        """
        candidates = (
            getattr(result, "embeddings", None),
            getattr(result, "feats", None),
            getattr(result, "features", None),
            getattr(result, "reid_features", None),
            getattr(getattr(result, "boxes", None), "embeddings", None),
            getattr(getattr(result, "boxes", None), "feats", None),
            getattr(getattr(result, "boxes", None), "features", None),
        )
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                vector = candidate[index]
                if hasattr(vector, "detach"):
                    vector = vector.detach().cpu().float().tolist()
                elif hasattr(vector, "tolist"):
                    vector = vector.tolist()
                return [float(v) for v in vector]
            except Exception:
                continue
        return None

    # ── Internal: PPE model ───────────────────────────────────

    def _run_ppe(self, frame: np.ndarray) -> list[Detection]:
        """Run the PPE model (if loaded) and return normalised detections."""
        if self._ppe_model is None:
            return []

        if self._ppe_model_is_roboflow:
            # Roboflow local inference returns a list of ObjectDetectionInferenceResponse
            results = self._ppe_model.infer(frame)
            detections: list[Detection] = []
            for res in results:
                for pred in res.predictions:
                    if pred.confidence < self._ppe_conf:
                        continue
                    cx, cy = pred.x, pred.y
                    w, h = pred.width, pred.height
                    x1 = int(cx - w / 2)
                    y1 = int(cy - h / 2)
                    x2 = int(cx + w / 2)
                    y2 = int(cy + h / 2)
                    bbox = (x1, y1, x2, y2)
                    canonical = _normalise_ppe_name(pred.class_name)
                    detections.append(
                        Detection(
                            class_name=canonical,
                            confidence=pred.confidence,
                            bbox=bbox,
                            track_id=None,
                            is_ppe=True,
                            model_source="ppe",
                        )
                    )
            return detections

        results = self._ppe_model.predict(
            source=frame,
            device=self._device,
            conf=self._ppe_conf,
            iou=self._ppe_iou,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                raw_name = self._ppe_model.names.get(cls_id, f"class_{cls_id}")
                canonical = _normalise_ppe_name(raw_name)
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                bbox = (int(x1), int(y1), int(x2), int(y2))

                detections.append(
                    Detection(
                        class_name=canonical,
                        confidence=conf,
                        bbox=bbox,
                        track_id=None,
                        is_ppe=True,
                        model_source="ppe",
                    )
                )
        return detections

    def _parse_tool_results(self, results, *, with_tracking: bool = False) -> list[Detection]:
        """Extract tool detections from YOLOWorld results."""
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if cls_id < len(self._tool_classes):
                    raw_name = self._tool_classes[cls_id]
                else:
                    raw_name = "tool"

                canonical = raw_name.lower().replace(" ", "_")
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                bbox = (int(x1), int(y1), int(x2), int(y2))

                track_id: Optional[int] = None
                if with_tracking and boxes.id is not None:
                    track_id = int(boxes.id[i].item())

                detections.append(
                    Detection(
                        class_name=canonical,
                        confidence=conf,
                        bbox=bbox,
                        track_id=track_id,
                        is_ppe=False,
                        model_source="tool",
                    )
                )
        return detections
