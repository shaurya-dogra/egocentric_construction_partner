"""Tier 2 reasoning hook backed by local Ollama Gemma 4."""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

import cv2

from core.models import Detection, FrameResult
from reasoning import ReasoningCoordinator
from reasoning.models import ReasoningSnapshot

logger = logging.getLogger(__name__)


class VLMHook:
    """Structured reasoning layer wired into the existing Tier 1 extension point."""

    def __init__(self, config: Optional[dict] = None, event_logger=None) -> None:
        config = config or {}
        self.config = config
        self._available = bool(config.get("enabled", False))
        self.api_url = config.get("api_url", "http://localhost:11434/api/chat")
        self._chat_urls = self._build_chat_url_candidates(self.api_url)
        self._active_chat_url = self._chat_urls[0]
        self.model = config.get("model", "gemma4:cloud")
        self.api_key = config.get("api_key") or os.environ.get("VLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("OLLAMA_API_KEY")
        self.system_prompt = config.get(
            "system_prompt",
            (
                "You are a construction-site reasoning assistant. "
                "Prefer the structured scene graph and session memory over free-form guessing. "
                "If retrieved documents are provided, ground answers in them and explicitly label any inference beyond them."
            ),
        )
        self.reasoning = ReasoningCoordinator(config.get("reasoning", {}), event_logger=event_logger)
        self.background_polling = bool(config.get("background_polling", True))
        self.frame_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self.insight_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._last_snapshot: Optional[ReasoningSnapshot] = None
        self._hazard_cache: dict[str, float] = {}
        self._disabled_until = 0.0
        self._disable_seconds = float(config.get("error_backoff_seconds", 30.0))

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "query_yolo",
                    "description": "Query the live detection state for counts and approximate locations of detected objects.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_object": {
                                "type": "string",
                                "description": "The object class to look up, for example person, truck, hardhat, drill.",
                            }
                        },
                        "required": ["target_object"],
                    },
                },
            }
        ]

        if self._available and self.background_polling:
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()
            logger.info("Started Tier 2 reasoning thread.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)

    def is_available(self) -> bool:
        return self._available and time.time() >= self._disabled_until

    def submit_scene(self, frame: np.ndarray, frame_result: FrameResult) -> None:
        """Submit structured scene state for low-rate asynchronous reasoning."""
        if not self._available or not self.background_polling:
            return
        snapshot = self.reasoning.build_snapshot(frame_result, frame.shape)
        self._last_snapshot = snapshot
        if not self.reasoning.should_schedule_reasoning(snapshot):
            return
        payload = {
            "snapshot": snapshot.to_dict(),
            "detections": frame_result.detections,
            "frame": frame.copy(),
        }
        try:
            self.frame_queue.put_nowait(payload)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait(payload)
            except queue.Full:
                return

    def submit_frame(self, frame: np.ndarray, detections: list[Any]) -> None:
        """Legacy Tier 1 API; kept for compatibility."""
        if not self._available or not self.background_polling:
            return
        payload = {
            "snapshot": None,
            "detections": detections,
        }
        try:
            self.frame_queue.put_nowait(payload)
        except queue.Full:
            pass

    def get_alerts(self) -> list[dict[str, Any]]:
        insights: list[dict[str, Any]] = []
        while not self.insight_queue.empty():
            insights.append(self.insight_queue.get())
        return insights

    def ask_question(
        self,
        prompt: str,
        scene_state: Optional[FrameResult | list[Detection]] = None,
        frame: Optional[np.ndarray] = None,
    ) -> str:
        if not self._available:
            return "Tier 2 reasoning is disabled in configuration."

        snapshot = self._snapshot_for(scene_state, frame)
        if snapshot:
            fast_answer = self.reasoning.fast_path_answer(prompt, snapshot)
            if fast_answer:
                return fast_answer

        retrieval_context = self.reasoning.retrieval_context(prompt)
        blueprint_context = self.reasoning.blueprint_context()
        images: list[str] = []
        prompt_lower = prompt.lower()
        if any(token in prompt_lower for token in ("blueprint", "plan", "floor", "room", "zone")):
            images.extend(self.reasoning.blueprint_images(limit=2))

        user_payload = {
            "question": prompt,
            "scene_snapshot": snapshot.to_dict() if snapshot else None,
            "retrieved_documents": retrieval_context,
            "blueprint_context": blueprint_context,
            "instructions": {
                "format": "answer in 3 short parts: Answer, Grounded in docs, Inference beyond docs",
                "grounding": "If retrieved text is absent, say that clearly.",
            },
        }
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
        ]
        encoded_images = self._encode_images(frame=frame, file_paths=images)
        if encoded_images:
            messages[1]["images"] = encoded_images
        try:
            return self._run_chat_loop(messages, self._detections_for(scene_state))
        except Exception as exc:
            logger.error("VLM question handling failed: %s", exc)
            return "I could not complete the reasoning request."

    def escalate_to_reasoning(
        self,
        frame: np.ndarray,
        detections: list[Any],
        hazard_context: dict[str, Any],
        frame_result: Optional[FrameResult] = None,
    ) -> Optional[str]:
        if not self._available:
            return None

        hazard = hazard_context["hazard"]
        now = time.time()
        last_run = self._hazard_cache.get(hazard.hazard_id)
        if last_run and (now - last_run) < float(self.config.get("escalation_cooldown_seconds", 5.0)):
            return None
        self._hazard_cache[hazard.hazard_id] = now

        snapshot = self._snapshot_for(frame_result, frame)
        retrieval_context = self.reasoning.retrieval_context(hazard.description)
        crop = self._crop_hazard(frame, getattr(hazard, "hazard_bbox", None))
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "mode": "hazard_escalation",
                        "hazard": {
                            "description": hazard.description,
                            "type": hazard.hazard_type,
                            "severity": hazard.severity.value,
                            "worker_track_id": hazard.worker_track_id,
                            "zone_name": hazard.zone_name,
                        },
                        "scene_snapshot": snapshot.to_dict() if snapshot else None,
                        "retrieved_documents": retrieval_context,
                        "instructions": "Verify the risk using structured state first, then return one concise spoken sentence.",
                    },
                    ensure_ascii=True,
                ),
            },
        ]
        if crop is not None:
            messages[1]["images"] = [self._encode_frame(crop)]
        try:
            return self._run_chat_loop(messages, detections)
        except Exception as exc:
            logger.error("VLM escalation failed: %s", exc)
            return None

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            snapshot = payload.get("snapshot")
            detections = payload.get("detections", [])
            frame = payload.get("frame")
            if not snapshot:
                continue
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "mode": "background_reasoning",
                            "scene_snapshot": snapshot,
                            "instructions": {
                                "output": {
                                    "tasks": [
                                        {
                                            "worker_id": "string",
                                            "label": "string",
                                            "confidence": 0.0,
                                        }
                                    ],
                                    "predictions": [
                                        {
                                            "label": "string",
                                            "confidence": 0.0,
                                            "reason": "string",
                                        }
                                    ],
                                    "guidance": [
                                        {
                                            "worker_id": "string",
                                            "next_step": "string",
                                            "grounded": True,
                                        }
                                    ],
                                },
                                "constraints": [
                                    "Use the scene graph and memory instead of restating all detections.",
                                    "Do not emit hazards as certain unless they are detections; predictions must be marked as predictions.",
                                    "Do NOT evaluate or predict PPE compliance (e.g., whether someone is wearing a hardhat or vest). Ignore PPE completely.",
                                    "Focus entirely on understanding worker tasks, intent, and progress relative to blueprint goals.",
                                    "Keep the JSON compact.",
                                ],
                            },
                        },
                        ensure_ascii=True,
                    ),
                },
            ]
            if frame is not None:
                try:
                    messages[1]["images"] = [self._encode_frame(frame)]
                except Exception as e:
                    logger.warning("Failed to encode frame for background reasoning: %s", e)
            try:
                logger.info("ENGAGING VLM: Sending scene snapshot for background reasoning...")
                response = self._run_chat_loop(messages, detections)
                logger.info("================ VLM RAW RESPONSE ================\n%s\n==================================================", response)
                
                parsed = self._extract_json(response)
                if not parsed:
                    logger.warning("Tier 2 reasoning returned invalid JSON")
                    continue
                
                tasks = parsed.get("tasks", [])
                predictions = parsed.get("predictions", [])
                logger.info(
                    "VLM Reasoning Complete -> Tasks: %s | Predictions: %s", 
                    json.dumps(tasks, indent=2), 
                    json.dumps(predictions, indent=2)
                )
                
                for prediction in parsed.get("predictions", []):
                    confidence = float(prediction.get("confidence", 0.0))
                    if confidence >= float(self.config.get("prediction_alert_threshold", 0.65)):
                        self.insight_queue.put(
                            {
                                "kind": "prediction",
                                "label": prediction.get("label", "predicted_hazard"),
                                "confidence": confidence,
                                "reason": prediction.get("reason", ""),
                            }
                        )
                self._disabled_until = 0.0
            except Exception as exc:
                self._pause_after_error(exc, "Tier 2 background reasoning error")

    def _run_chat_loop(self, messages: list[dict], detections: Optional[list[Any]] = None) -> str:
        for _ in range(5):
            result, mode_used = self._post_chat_request(messages, detections or [])
            if mode_used == "generate":
                return str(result.get("response", "")).strip()
            if mode_used == "openai_chat":
                choice = (result.get("choices") or [{}])[0]
                message = choice.get("message", {})
                messages.append(message)
                if message.get("tool_calls"):
                    for tool_call in message["tool_calls"]:
                        function = tool_call.get("function", {})
                        if function.get("name") != "query_yolo":
                            continue
                        arguments = function.get("arguments", {})
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                arguments = {}
                        target = arguments.get("target_object", "person")
                        tool_result = self._execute_query_yolo(target, detections or [])
                        messages.append(
                            {
                                "role": "tool",
                                "content": tool_result,
                                "name": "query_yolo",
                                "tool_call_id": tool_call.get("id"),
                            }
                        )
                    continue
                return str(message.get("content", "")).strip()

            message = result.get("message", {})
            messages.append(message)

            if message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    if tool_call["function"]["name"] != "query_yolo":
                        continue
                    args = tool_call["function"].get("arguments", {})
                    target = args.get("target_object", "person")
                    tool_result = self._execute_query_yolo(target, detections or [])
                    messages.append(
                        {
                            "role": "tool",
                            "content": tool_result,
                            "name": "query_yolo",
                        }
                    )
                continue
            return message.get("content", "").strip()
        return "Reasoning loop limit reached."

    def _post_chat_request(
        self,
        messages: list[dict[str, Any]],
        detections: list[Any],
    ) -> tuple[dict[str, Any], str]:
        last_error: Optional[Exception] = None
        for url in self._chat_urls:
            try:
                result = self._send_chat_request(url, messages, detections)
                self._active_chat_url = url
                return result
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code == 404:
                    logger.warning("Ollama endpoint %s returned 404; trying fallback.", url)
                    continue
                raise
            except urllib.error.URLError as exc:
                last_error = exc
                if url != self._active_chat_url:
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("No chat endpoints configured.")

    def _send_chat_request(
        self,
        url: str,
        messages: list[dict[str, Any]],
        detections: list[Any],
    ) -> tuple[dict[str, Any], str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if "chat/completions" in url:
            # Map the configured model to the native target model for cloud APIs
            model_name = self.model
            if "googleapis.com" in url:
                model_name = "gemini-1.5-flash"
            elif "groq.com" in url:
                model_name = "llama-3.2-11b-vision-preview"

            payload = {
                "model": model_name,
                "messages": self._messages_to_openai_chat(messages),
                "stream": False,
                "max_tokens": 400,
                "temperature": 0.1,
            }
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8")), "openai_chat"

        if url.endswith("/api/generate"):
            payload = {
                "model": self.model,
                "prompt": self._messages_to_prompt(messages, detections),
                "stream": False,
                "options": {
                    "num_ctx": 4096,
                    "num_predict": 400,
                    "temperature": 0.1,
                }
            }
            images = self._collect_images(messages)
            if images:
                payload["images"] = images
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8")), "generate"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "tools": self.tools,
            "options": {
                "num_ctx": 4096,
                "num_predict": 400,
                "temperature": 0.1,
            }
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.loads(response.read().decode("utf-8")), "chat"

    @staticmethod
    def _execute_query_yolo(target_object: str, detections: list[Any]) -> str:
        target = target_object.lower()
        matches = []
        for detection in detections:
            name = getattr(detection, "class_name", "").lower()
            if target in name or (target == "worker" and name == "person"):
                bbox = getattr(detection, "bbox", None)
                if bbox:
                    cx = int((bbox[0] + bbox[2]) / 2)
                    cy = int((bbox[1] + bbox[3]) / 2)
                    matches.append(f"{name} at ({cx},{cy})")
                else:
                    matches.append(name)
        if not matches:
            return f"No {target_object} found in the current detection state."
        return f"Found {len(matches)} {target_object}(s): " + ", ".join(matches)

    def _pause_after_error(self, exc: Exception, prefix: str) -> None:
        self._disabled_until = time.time() + self._disable_seconds
        logger.error(
            "%s: %s. Pausing Tier 2 reasoning for %.0f seconds.",
            prefix,
            exc,
            self._disable_seconds,
        )

    def _snapshot_for(
        self,
        scene_state: Optional[FrameResult | list[Detection]],
        frame: Optional[np.ndarray],
    ) -> Optional[ReasoningSnapshot]:
        if isinstance(scene_state, FrameResult) and frame is not None:
            snapshot = self.reasoning.build_snapshot(scene_state, frame.shape)
            self._last_snapshot = snapshot
            return snapshot
        if isinstance(scene_state, FrameResult):
            return self._last_snapshot
        return self._last_snapshot

    @staticmethod
    def _detections_for(scene_state: Optional[FrameResult | list[Detection]]) -> list[Any]:
        if isinstance(scene_state, FrameResult):
            return scene_state.detections
        return scene_state or []

    @staticmethod
    def _crop_hazard(
        frame: Optional[np.ndarray],
        bbox: Optional[tuple[int, int, int, int]],
        padding: int = 40,
    ) -> Optional[np.ndarray]:
        if frame is None or bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        return frame[
            max(0, y1 - padding):min(h, y2 + padding),
            max(0, x1 - padding):min(w, x2 + padding),
        ]

    @staticmethod
    def _encode_frame(frame: np.ndarray) -> str:
        # Resize to 448px (native resolution for most VLM vision encoders) to minimize server-side scaling
        h, w = frame.shape[:2]
        max_size = 448
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Compress heavily to quality 35 to minimize upload payload size over network
        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 35])
        if not success:
            raise ValueError("Failed to encode frame for Gemma 4.")
        return base64.b64encode(buffer).decode("utf-8")

    def _encode_images(
        self,
        frame: Optional[np.ndarray] = None,
        file_paths: Optional[list[str]] = None,
    ) -> list[str]:
        images: list[str] = []
        if frame is not None:
            images.append(self._encode_frame(frame))
        for path_str in file_paths or []:
            path = Path(path_str)
            if not path.exists():
                continue
            try:
                images.append(base64.b64encode(path.read_bytes()).decode("utf-8"))
            except OSError:
                continue
        return images

    @staticmethod
    def _collect_images(messages: list[dict[str, Any]]) -> list[str]:
        images: list[str] = []
        for message in messages:
            for image in message.get("images", []) or []:
                images.append(image)
        return images

    def _messages_to_prompt(self, messages: list[dict[str, Any]], detections: list[Any]) -> str:
        parts = []
        for message in messages:
            role = message.get("role", "user").upper()
            content = message.get("content", "")
            if content:
                parts.append(f"{role}:\n{content}")
        if detections:
            parts.append("DETECTIONS:\n" + self._serialize_detections(detections))
        parts.append(
            "Return only the final answer. If you need to reason about detections, use the structured detection summary above."
        )
        return "\n\n".join(parts)

    @staticmethod
    def _serialize_detections(detections: list[Any]) -> str:
        rows = []
        for detection in detections:
            class_name = getattr(detection, "class_name", "unknown")
            bbox = getattr(detection, "bbox", None)
            track_id = getattr(detection, "track_id", None)
            rows.append(
                json.dumps(
                    {
                        "class_name": class_name,
                        "track_id": track_id,
                        "bbox": bbox,
                    },
                    ensure_ascii=True,
                )
            )
        return "\n".join(rows)

    @staticmethod
    def _messages_to_openai_chat(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            images = message.get("images") or []
            payload: dict[str, Any] = {"role": role}
            if images:
                parts: list[dict[str, Any]] = []
                if content:
                    parts.append({"type": "text", "text": content})
                for image_b64 in images:
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        }
                    )
                payload["content"] = parts
            else:
                payload["content"] = content
            if message.get("name"):
                payload["name"] = message["name"]
            if message.get("tool_call_id"):
                payload["tool_call_id"] = message["tool_call_id"]
            converted.append(payload)
        return converted

    @staticmethod
    def _build_chat_url_candidates(configured_url: str) -> list[str]:
        parsed = urllib.parse.urlparse(configured_url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        path = parsed.path.rstrip("/")
        candidates: list[str] = []
        if configured_url:
            candidates.append(configured_url)
        if path.endswith("/api/chat"):
            candidates.extend([base + "/api/generate", base + "/v1/chat/completions"])
        elif path.endswith("/api/generate"):
            candidates.extend([base + "/api/chat", base + "/v1/chat/completions"])
        elif path.endswith("/v1/chat/completions"):
            candidates.extend([base + "/api/chat", base + "/api/generate"])
        else:
            candidates.extend([base + "/api/chat", base + "/api/generate", base + "/v1/chat/completions"])
        deduped: list[str] = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped

    @staticmethod
    def _extract_json(response: str) -> Optional[dict[str, Any]]:
        import re
        response = response.strip()
        if not response:
            return None

        # Clean markdown code block wrappers
        start_idx = response.find("```")
        if start_idx >= 0:
            content_start = start_idx + 3
            if response[content_start : content_start + 4].lower() == "json":
                content_start += 4
            
            end_idx = response.rfind("```")
            if end_idx > start_idx:
                response = response[content_start:end_idx].strip()
            else:
                response = response[content_start:].strip()
        else:
            start = response.find("{")
            end = response.rfind("}")
            if start >= 0:
                if end > start:
                    response = response[start : end + 1].strip()
                else:
                    response = response[start:].strip()

        # Repair truncated JSON by closing open braces and brackets
        stack = []
        in_string = False
        escaped = False
        for char in response:
            if in_string:
                if char == '"' and not escaped:
                    in_string = False
                elif char == '\\':
                    escaped = not escaped
                else:
                    escaped = False
            else:
                if char == '"':
                    in_string = True
                elif char in ('{', '['):
                    stack.append(char)
                elif char == '}':
                    if stack and stack[-1] == '{':
                        stack.pop()
                elif char == ']':
                    if stack and stack[-1] == '[':
                        stack.pop()

        if in_string:
            response += '"'

        while stack:
            top = stack.pop()
            if top == '{':
                response += '}'
            elif top == '[':
                response += ']'

        # Robust cleanup of trailing commas (e.g., ", }" -> " }" and ", ]" -> " ]")
        response = re.sub(r",\s*([}\]])", r"\1", response)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # If still failing, try a final attempt at loading by replacing unescaped newlines in JSON strings
            try:
                cleaned = re.sub(r'(?<=[:\s])"(.*?)"', lambda m: m.group(0).replace("\n", "\\n"), response)
                return json.loads(cleaned)
            except Exception:
                return None
        return None
