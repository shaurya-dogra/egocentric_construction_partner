"""On-device Moondream2 VLM engine for the Safety Copilot.

Wraps `vikhyatk/moondream2` (loaded via HuggingFace `transformers`) and
exposes a clean API for visual question answering, captioning, object
detection, and scene description.  The model is loaded lazily on first
call and is protected by a threading lock so it can be called from the
background reasoning thread safely.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("integration.moondream")


class MoondreamEngine:
    """Thread-safe wrapper around `vikhyatk/moondream2`."""

    def __init__(self, config: Optional[dict] = None, device_override: Optional[str] = None) -> None:
        config = config or {}
        self.model_id: str = config.get("model_id", "vikhyatk/moondream2")
        self.revision: Optional[str] = config.get("revision", None)
        self.max_image_size: int = int(config.get("max_image_size", 448))

        # Resolve dtype string → torch dtype
        dtype_str = config.get("dtype", "float16")
        self.torch_dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }.get(dtype_str, torch.float16)

        # Device selection
        if device_override and device_override != "auto":
            self.device = device_override
        else:
            requested = config.get("device", "auto")
            if requested == "auto":
                if torch.backends.mps.is_available():
                    self.device = "mps"
                elif torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
            else:
                self.device = requested

        # MPS doesn't support bfloat16 — fall back to float16
        if self.device == "mps" and self.torch_dtype == torch.bfloat16:
            logger.info("MPS does not support bfloat16; falling back to float16.")
            self.torch_dtype = torch.float16

        self._model = None
        self._lock = threading.Lock()
        self._load_failed = False

    # ── Lazy Model Loading ───────────────────────────────────

    def _ensure_loaded(self) -> bool:
        """Lazily load the model on first use.  Returns True if the model is ready."""
        if self._model is not None:
            return True
        if self._load_failed:
            return False

        with self._lock:
            # Double-check after acquiring lock
            if self._model is not None:
                return True
            if self._load_failed:
                return False

            logger.info(
                "Loading Moondream2 model '%s' (revision=%s) on device '%s' with dtype %s ...",
                self.model_id,
                self.revision or "latest",
                self.device,
                self.torch_dtype,
            )
            start = time.time()
            try:
                # ── Fix transformers 5.x compatibility issue with all_tied_weights_keys ──
                try:
                    from transformers import PreTrainedModel
                    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
                        PreTrainedModel.all_tied_weights_keys = property(lambda self: {})
                except Exception as e:
                    logger.warning("Could not patch PreTrainedModel: %s", e)

                from transformers import AutoModelForCausalLM

                kwargs: dict[str, Any] = {
                    "trust_remote_code": True,
                    "dtype": self.torch_dtype,
                }

                if self.revision:
                    kwargs["revision"] = self.revision

                # device_map expects a dict or string
                kwargs["device_map"] = {"": self.device}

                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    **kwargs,
                )
                self._model.eval()

                # ── Monkeypatch Moondream2 MPS/CPU device mismatch bug ──
                import sys
                for name, module in list(sys.modules.items()):
                    if "transformers_modules" in name and name.endswith("vision"):
                        def safe_adaptive_avg_pool2d(input, output_size):
                            orig_device = input.device
                            import torch.nn.functional as F
                            return F.adaptive_avg_pool2d(input.to("cpu"), output_size).to(orig_device)
                        module.adaptive_avg_pool2d = safe_adaptive_avg_pool2d
                        logger.info("Successfully patched Moondream2 vision module for device safety: %s", name)

                logger.info(
                    "Moondream2 loaded successfully in %.1fs on '%s'.",
                    time.time() - start,
                    self.device,
                )
                return True
            except Exception as exc:
                import traceback
                import sys
                tb = traceback.format_exc()
                logger.error("Failed to load Moondream2: %s. Traceback:\n%s. Moondream backend disabled.", exc, tb)
                # Print directly to stderr to ensure user sees it in console
                sys.stderr.write(f"\n❌ [ERROR] Moondream2 model loading failed: {exc}\n")
                sys.stderr.write("Traceback:\n" + tb + "\n")
                sys.stderr.flush()
                self._load_failed = True
                return False

    @property
    def is_ready(self) -> bool:
        """True if the model is loaded or can still be loaded."""
        return self._model is not None or not self._load_failed

    # ── Image Conversion ─────────────────────────────────────

    def _prepare_image(self, frame: np.ndarray) -> Image.Image:
        """Convert a BGR OpenCV frame to a resized RGB PIL Image."""
        h, w = frame.shape[:2]
        max_dim = self.max_image_size
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(
                frame,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    # ── Public API ────────────────────────────────────────────

    def query(self, frame: np.ndarray, question: str) -> str:
        """Visual question answering — returns the answer string."""
        if not self._ensure_loaded():
            return "Moondream2 model is not available."
        image = self._prepare_image(frame)
        with self._lock:
            try:
                result = self._model.query(image, question)
                return result.get("answer", str(result)) if isinstance(result, dict) else str(result)
            except Exception as exc:
                logger.error("Moondream2 query failed: %s", exc)
                return f"Error during visual query: {exc}"

    def caption(self, frame: np.ndarray, length: str = "normal") -> str:
        """Generate a caption for the frame.  length: 'short' | 'normal' | 'long'."""
        if not self._ensure_loaded():
            return "Moondream2 model is not available."
        image = self._prepare_image(frame)
        with self._lock:
            try:
                result = self._model.caption(image, length=length)
                return result.get("caption", str(result)) if isinstance(result, dict) else str(result)
            except Exception as exc:
                logger.error("Moondream2 caption failed: %s", exc)
                return f"Error during captioning: {exc}"

    def detect(self, frame: np.ndarray, object_class: str) -> list[dict[str, Any]]:
        """Detect objects of *object_class* and return a list of bounding-box dicts.

        Each dict has ``{"box": [x_min, y_min, x_max, y_max]}`` with
        coordinates normalised to 0–1.
        """
        if not self._ensure_loaded():
            return []
        image = self._prepare_image(frame)
        with self._lock:
            try:
                result = self._model.detect(image, object_class)
                return result.get("objects", []) if isinstance(result, dict) else []
            except Exception as exc:
                logger.error("Moondream2 detect failed: %s", exc)
                return []

    def point(self, frame: np.ndarray, object_class: str) -> list[dict[str, Any]]:
        """Point at objects of *object_class* and return coordinate dicts."""
        if not self._ensure_loaded():
            return []
        image = self._prepare_image(frame)
        with self._lock:
            try:
                result = self._model.point(image, object_class)
                return result.get("points", []) if isinstance(result, dict) else []
            except Exception as exc:
                logger.error("Moondream2 point failed: %s", exc)
                return []

    def describe_scene(self, frame: np.ndarray, context: str = "") -> str:
        """Produce a comprehensive scene description suitable for safety reasoning.

        Combines a detailed caption with a targeted safety-oriented query
        to give the reasoning pipeline rich textual context.
        """
        if not self._ensure_loaded():
            return "Moondream2 model is not available."

        image = self._prepare_image(frame)

        with self._lock:
            try:
                # Get a long caption first
                cap_result = self._model.caption(image, length="long")
                scene_caption = (
                    cap_result.get("caption", str(cap_result))
                    if isinstance(cap_result, dict)
                    else str(cap_result)
                )

                # Follow up with a safety-oriented query
                safety_question = (
                    "Describe any safety hazards, risks, or unsafe conditions you see. "
                    "Include workers without protective equipment, proximity to machinery or vehicles, "
                    "fall risks, and any other construction-site dangers."
                )
                if context:
                    safety_question += f" Additional context: {context}"

                safety_result = self._model.query(image, safety_question)
                safety_analysis = (
                    safety_result.get("answer", str(safety_result))
                    if isinstance(safety_result, dict)
                    else str(safety_result)
                )

                return f"Scene: {scene_caption}\nSafety Analysis: {safety_analysis}"
            except Exception as exc:
                logger.error("Moondream2 describe_scene failed: %s", exc)
                return f"Error during scene description: {exc}"
