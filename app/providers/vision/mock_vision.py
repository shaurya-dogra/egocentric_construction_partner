"""Mock Multimodal Vision Reasoner for offline testing."""

from typing import Any, Dict, List, Optional, Tuple
from app.interfaces import VisionReasoner


class MockVisionReasoner(VisionReasoner):
    """Mock vision reasoner for local tests and offline development."""

    @property
    def name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-vlm-v1"

    def _normalize_images(self, images: Any, default_mime: str = "image/jpeg") -> List[Tuple[bytes, str]]:
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
        """Return a structured mock response referencing question and frame count."""
        image_list = self._normalize_images(images, mime_type)
        frame_count = len(image_list)
        q = question.strip() if question else "What is this?"

        if frame_count > 1:
            return f"I analyzed a temporal sequence of {frame_count} frames. In response to '{q}': The scene shows consistent lighting with subtle movement across the frames."
        return f"I can see the camera frame clearly. In response to '{q}': This appears to be an object in front of the lens."
