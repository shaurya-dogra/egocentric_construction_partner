"""Factory methods for instantiating STT, Vision, and TTS providers."""

import logging
from app.config import Settings
from app.interfaces import STTProvider, VisionReasoner, TTSProvider

logger = logging.getLogger("kaya.factory")


def get_stt_provider(settings: Settings) -> STTProvider:
    """Resolve and return the configured STT provider."""
    provider_type = settings.stt_provider.lower()

    if provider_type == "mock":
        from app.providers.stt.mock_stt import MockSTTProvider
        logger.info("Using Mock STT Provider.")
        return MockSTTProvider()

    if provider_type == "gemini":
        from app.providers.stt.gemini_stt import GeminiSTTProvider
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when STT_PROVIDER='gemini'.")
        logger.info(f"Using Gemini STT Provider ({settings.gemini_model}).")
        return GeminiSTTProvider(api_key=settings.gemini_api_key, model=settings.gemini_model)

    if provider_type == "sarvam":
        from app.providers.stt.sarvam_stt import SarvamSTTProvider
        if not settings.sarvam_api_key:
            raise ValueError("SARVAM_API_KEY is required when STT_PROVIDER='sarvam'.")
        logger.info(f"Using Sarvam STT Provider ({settings.sarvam_stt_model}).")
        return SarvamSTTProvider(
            api_key=settings.sarvam_api_key,
            model=settings.sarvam_stt_model,
            language_code=settings.sarvam_stt_language_code,
            gemini_api_key=settings.gemini_api_key,
        )

    raise ValueError(f"Unsupported STT provider: '{settings.stt_provider}'")


def get_vision_reasoner(settings: Settings) -> VisionReasoner:
    """Resolve and return the configured Multimodal Vision reasoner."""
    provider_type = settings.vision_provider.lower()

    if provider_type == "mock":
        from app.providers.vision.mock_vision import MockVisionReasoner
        logger.info("Using Mock Vision Reasoner.")
        return MockVisionReasoner()

    if provider_type == "gemini":
        from app.providers.vision.gemini_vision import GeminiVisionReasoner
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when VISION_PROVIDER='gemini'.")
        logger.info(f"Using Gemini Vision Reasoner ({settings.gemini_model}).")
        return GeminiVisionReasoner(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model
        )

    if provider_type == "nvidia":
        from app.providers.vision.nvidia_vision import NvidiaVisionReasoner
        if not settings.nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY is required when VISION_PROVIDER='nvidia'.")
        logger.info(f"Using NVIDIA Vision Reasoner ({settings.nvidia_model}).")
        return NvidiaVisionReasoner(
            api_key=settings.nvidia_api_key,
            model=settings.nvidia_model,
            base_url=settings.nvidia_base_url
        )

    if provider_type == "ollama":
        from app.providers.vision.ollama_vision import OllamaVisionReasoner
        logger.info(f"Using Ollama Vision Reasoner ({settings.ollama_model} @ {settings.ollama_base_url}).")
        return OllamaVisionReasoner(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model
        )

    raise ValueError(f"Unsupported Vision provider: '{settings.vision_provider}'")


def get_tts_provider(settings: Settings) -> TTSProvider:
    """Resolve and return the configured TTS provider."""
    provider_type = settings.tts_provider.lower()

    if provider_type == "mock":
        from app.providers.tts.mock_tts import MockTTSProvider
        logger.info("Using Mock TTS Provider.")
        return MockTTSProvider()

    if provider_type == "mac":
        from app.providers.tts.mac_tts import MacNativeTTSProvider
        logger.info("Using macOS Native TTS Provider (Samantha).")
        return MacNativeTTSProvider()

    if provider_type == "sarvam":
        from app.providers.tts.sarvam_tts import SarvamTTSProvider
        if not settings.sarvam_api_key:
            raise ValueError("SARVAM_API_KEY is required when TTS_PROVIDER='sarvam'.")
        logger.info(f"Using Sarvam TTS Provider ({settings.sarvam_tts_model}/{settings.sarvam_tts_speaker}).")
        return SarvamTTSProvider(
            api_key=settings.sarvam_api_key,
            model=settings.sarvam_tts_model,
            speaker=settings.sarvam_tts_speaker,
            language_code=settings.sarvam_tts_language_code
        )

    raise ValueError(f"Unsupported TTS provider: '{settings.tts_provider}'")

