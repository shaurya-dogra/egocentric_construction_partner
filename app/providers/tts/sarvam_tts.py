"""Sarvam AI Text-to-Speech (Bulbul) Provider with macOS Native Failover."""

import base64
import logging
from typing import Optional
import httpx

from app.interfaces import TTSProvider
from app.providers.tts.mac_tts import MacNativeTTSProvider

logger = logging.getLogger("kaya.tts.sarvam")


class SarvamTTSProvider(TTSProvider):
    """TTS Provider using Sarvam AI Bulbul API with automatic local failover."""

    def __init__(
        self,
        api_key: str,
        model: str = "bulbul:v3",
        speaker: str = "shubh",
        language_code: str = "en-IN",
    ):
        if not api_key:
            raise ValueError("Sarvam API key is required for SarvamTTSProvider.")
        self.api_key = api_key
        self.model = model
        self.speaker = speaker
        self.language_code = language_code
        self.api_url = "https://api.sarvam.ai/text-to-speech"
        self.mac_fallback = MacNativeTTSProvider()

    @property
    def name(self) -> str:
        return f"sarvam/{self.model}/{self.speaker}"

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text into WAV audio bytes, with automatic failover."""
        if not text or not text.strip():
            return b""

        payload = {
            "inputs": [text.strip()],
            "target_language_code": self.language_code,
            "speaker": self.speaker,
            "model": self.model,
            "enable_preprocessing": True
        }

        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json"
        }

        # Fast connection timeout (2.5s) to allow seamless fallback to macOS native speech if network blocks Sarvam
        custom_timeout = httpx.Timeout(connect=2.5, read=6.0, write=3.0, pool=2.5)
        try:
            async with httpx.AsyncClient(timeout=custom_timeout) as client:
                response = await client.post(
                    self.api_url,
                    headers=headers,
                    json=payload
                )


            if response.status_code == 200:
                result = response.json()
                audios = result.get("audios", [])
                if audios and audios[0]:
                    return base64.b64decode(audios[0])
            else:
                logger.warning(f"Sarvam TTS error (status {response.status_code}): {response.text}")

        except (httpx.ConnectTimeout, httpx.ConnectError, httpx.TimeoutException) as net_err:
            logger.warning(f"Sarvam TTS network timeout: {net_err}. Using macOS native speech failover...")
            return await self.mac_fallback.synthesize(text)
        except Exception as e:
            logger.warning(f"Sarvam TTS error: {e}. Using macOS native speech failover...")
            return await self.mac_fallback.synthesize(text)

        # Fallback if Sarvam returned non-200 or empty audio
        return await self.mac_fallback.synthesize(text)
