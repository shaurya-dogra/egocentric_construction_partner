"""Google Gemini Multimodal Vision Reasoner Provider."""

import base64
import io
from typing import Any, Dict, List, Optional, Tuple, Union
import httpx
from PIL import Image

from app.interfaces import VisionReasoner

KAYA_SYSTEM_PROMPT = (
    "You are Kaya, a visual voice assistant.\n\n"
    "Answer the user's question using the provided camera image or chronological sequence of camera frames.\n\n"
    "Only claim things that are reasonably supported by the visual input.\n"
    "If the image is unclear or you cannot determine something, say so.\n"
    "Do not invent details.\n\n"
    "Keep responses concise and natural because the response will be spoken aloud.\n\n"
    "The user may ask about objects, people, scenes, text, locations, actions, motion, or changes visible across the frames."
)


class GeminiVisionReasoner(VisionReasoner):
    """Multimodal reasoning using Google Gemini API."""

    def __init__(self, api_key: str, model: str = "gemini-3.6-flash"):
        if not api_key:
            raise ValueError("Gemini API key is required for GeminiVisionReasoner.")
        self.api_key = api_key
        self.model = model
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self.model

    def _prepare_image(self, image_bytes: bytes, max_dim: int = 1280) -> tuple[bytes, str]:
        """Optionally resize image if too large, while retaining high visual clarity for text and objects."""
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                format_name = img.format or "JPEG"
                mime = f"image/{format_name.lower()}"
                if format_name.upper() not in ["JPEG", "PNG", "WEBP"]:
                    # Convert to JPEG for standard Gemini compatibility
                    img = img.convert("RGB")
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    return buf.getvalue(), "image/jpeg"

                if max(img.width, img.height) > max_dim:
                    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                    buf = io.BytesIO()
                    img.save(buf, format=format_name, quality=85)
                    return buf.getvalue(), mime
                return image_bytes, mime
        except Exception:
            return image_bytes, "image/jpeg"

    def _normalize_images(self, images: Any, default_mime: str = "image/jpeg") -> List[Tuple[bytes, str]]:
        """Normalize input images to a list of (bytes, mime_type) tuples."""
        if isinstance(images, bytes):
            return [(images, default_mime)]
        if isinstance(images, list):
            norm = []
            for item in images:
                if isinstance(item, tuple) and len(item) == 2:
                    norm.append((item[0], item[1]))
                elif isinstance(item, bytes):
                    norm.append((item, default_mime))
            return norm
        return []

    async def answer(
        self,
        question: str,
        images: Any,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        mime_type: str = "image/jpeg",
    ) -> str:
        """Answer question based on captured image(s) and context."""
        if not question or not question.strip():
            question = "What am I looking at?"

        image_list = self._normalize_images(images, mime_type)
        if not image_list:
            raise ValueError("No valid image frames provided to GeminiVisionReasoner.")

        # Build contents structure with conversation history
        contents = []

        if conversation_history:
            for turn in conversation_history:
                role = "user" if turn.get("role") == "user" else "model"
                text = turn.get("content", "")
                if text:
                    contents.append({
                        "role": role,
                        "parts": [{"text": text}]
                    })

        # Build user turn parts with image frame(s)
        user_parts = []

        # If multiple frames are present, add a temporal context note
        if len(image_list) > 1:
            user_parts.append({
                "text": f"[Context: The user provided a chronological sequence of {len(image_list)} camera frames captured at 1 FPS leading up to now.]"
            })

        for idx, (img_bytes, mime) in enumerate(image_list):
            processed_bytes, resolved_mime = self._prepare_image(img_bytes)
            b64_image = base64.b64encode(processed_bytes).decode("utf-8")
            user_parts.append({
                "inline_data": {
                    "mime_type": resolved_mime,
                    "data": b64_image
                }
            })

        user_parts.append({"text": question})

        contents.append({
            "role": "user",
            "parts": user_parts
        })

        payload = {
            "system_instruction": {
                "parts": [{"text": KAYA_SYSTEM_PROMPT}]
            },
            "contents": contents,
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 350
            }
        }

        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        custom_timeout = httpx.Timeout(connect=15.0, read=60.0, write=30.0, pool=10.0)
        async with httpx.AsyncClient(timeout=custom_timeout) as client:
            response = await client.post(
                self.api_url,
                headers=headers,
                json=payload
            )

        if response.status_code != 200:
            error_detail = response.text
            raise RuntimeError(f"Gemini API error (status {response.status_code}): {error_detail}")


        result = response.json()
        try:
            candidates = result.get("candidates", [])
            if not candidates:
                raise RuntimeError("No answer candidate returned by Gemini.")
            text_parts = candidates[0].get("content", {}).get("parts", [])
            answer_text = "".join(p.get("text", "") for p in text_parts).strip()
            return answer_text or "I was unable to see anything clearly in the frames."
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Failed to parse Gemini response: {e}")
