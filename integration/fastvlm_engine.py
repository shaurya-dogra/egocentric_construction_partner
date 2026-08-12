"""On-device FastVLM VLM engine for the Safety Copilot.

Wraps `apple/FastVLM-0.5B` (loaded via HuggingFace `transformers`) and
exposes a clean API for visual question answering, captioning, and scene description.
The model is loaded lazily on first call and is protected by a threading lock.
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

logger = logging.getLogger("integration.fastvlm")


class FastVlmEngine:
    """Thread-safe wrapper around `apple/FastVLM-0.5B`."""

    def __init__(self, config: Optional[dict] = None, device_override: Optional[str] = None) -> None:
        config = config or {}
        self.model_id: str = config.get("model_id", "apple/FastVLM-0.5B")
        self.max_image_size: int = int(config.get("max_image_size", 448))
        self.IMAGE_TOKEN_INDEX = -200

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

        # CPU must run in float32
        if self.device == "cpu":
            self.torch_dtype = torch.float32

        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._load_failed = False

    # ── Lazy Model Loading ───────────────────────────────────

    def _ensure_loaded(self) -> bool:
        """Lazily load the model on first use. Returns True if the model is ready."""
        if self._model is not None:
            return True
        if self._load_failed:
            return False

        with self._lock:
            if self._model is not None:
                return True
            if self._load_failed:
                return False

            logger.info(
                "Loading FastVLM model '%s' on device '%s' with dtype %s ...",
                self.model_id,
                self.device,
                self.torch_dtype,
            )
            start = time.time()
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    trust_remote_code=True
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    dtype=self.torch_dtype,
                    trust_remote_code=True,
                    attn_implementation="sdpa",
                ).to(self.device)
                self._model.eval()

                logger.info(
                    "FastVLM loaded successfully in %.1fs on '%s'.",
                    time.time() - start,
                    self.device,
                )
                return True
            except Exception as exc:
                import traceback
                import sys
                tb = traceback.format_exc()
                logger.error("Failed to load FastVLM: %s. Traceback:\n%s. FastVLM backend disabled.", exc, tb)
                sys.stderr.write(f"\n❌ [ERROR] FastVLM model loading failed: {exc}\n")
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

    def query(self, frame: np.ndarray, question: str, max_new_tokens: int = 100) -> str:
        """Visual question answering — returns the answer string."""
        if not self._ensure_loaded():
            return "FastVLM model is not available."

        img = self._prepare_image(frame)

        with self._lock:
            try:
                # Preprocess image into pixel_values tensor
                pixel_values = self._model.get_vision_tower().image_processor(
                    images=img,
                    return_tensors="pt"
                )["pixel_values"].to(self.device, dtype=self.torch_dtype)

                messages = [
                    {"role": "user", "content": f"<image>\n{question}"}
                ]
                
                rendered = self._tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                pre, post = rendered.split("<image>", 1)
                
                pre_ids = self._tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
                post_ids = self._tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids
                img_tok = torch.tensor([[self.IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
                
                input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(self.device)

                attention_mask = torch.ones_like(input_ids).to(self.device)

                with torch.no_grad():
                    output_ids = self._model.generate(
                        input_ids,
                        images=pixel_values,
                        attention_mask=attention_mask,
                        pad_token_id=self._tokenizer.eos_token_id,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.2,
                        repetition_penalty=1.05,
                        eos_token_id=self._tokenizer.eos_token_id,
                    )
                
                return self._tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            except Exception as exc:
                logger.error("FastVLM query failed: %s", exc)
                return f"Error during visual query: {exc}"

    def caption(self, frame: np.ndarray, length: str = "normal") -> str:
        """Generate a caption for the frame."""
        prompt = "Describe this image briefly." if length == "short" else "Describe the given image in detail."
        max_tokens = 50 if length == "short" else 100
        return self.query(frame, prompt, max_new_tokens=max_tokens)

    def describe_scene(self, frame: np.ndarray, context: str = "") -> str:
        """Produce a comprehensive scene description suitable for safety reasoning using a single VLM query."""
        prompt = (
            "Describe the given construction scene in detail, identifying any safety hazards, "
            "risks, or unsafe conditions you see (such as proximity to machinery/vehicles, "
            "lack of protective equipment, or fall risks)."
        )
        if context:
            prompt += f" Additional context: {context}"
            
        combined_analysis = self.query(frame, prompt, max_new_tokens=120)
        return combined_analysis
