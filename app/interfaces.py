"""Abstract base classes and interfaces for Kaya Voice + Vision Assistant."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class STTProvider(ABC):
    """Abstract interface for Speech-to-Text transcription providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identification name."""
        pass

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
        """Transcribe speech audio bytes to text string.

        Args:
            audio_bytes: Raw audio byte buffer.
            mime_type: MIME type of audio (e.g. 'audio/wav', 'audio/webm').

        Returns:
            Transcribed text.
        """
        pass


class VisionReasoner(ABC):
    """Abstract interface for Multimodal Vision + LLM reasoning providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identification name."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the underlying multimodal model."""
        pass

    @abstractmethod
    async def answer(
        self,
        question: str,
        images: Any,  # Union[List[Tuple[bytes, str]], bytes, List[bytes]]
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        mime_type: str = "image/jpeg",
    ) -> str:
        """Answer user's question using the captured image(s) and conversation context.

        Args:
            question: The user's transcribed or typed question.
            images: Single image bytes, or list of (image_bytes, mime_type) in chronological order.
            conversation_history: List of prior turns [{'role': 'user'|'assistant', 'content': '...'}].
            mime_type: Default MIME type if raw bytes passed.

        Returns:
            Natural spoken response string.
        """
        pass



class TTSProvider(ABC):
    """Abstract interface for Text-to-Speech synthesis providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identification name."""
        pass

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Synthesize text into speech audio bytes.

        Args:
            text: Text to synthesize.

        Returns:
            Audio bytes (e.g. WAV / MP3 format).
        """
        pass
