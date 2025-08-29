"""Composite TTS service that chooses between Piper and pyttsx3.

This service provides automatic fallback from Piper to pyttsx3,
ensuring robust TTS functionality while preferring higher quality
Piper synthesis when available.
"""

import logging
from pathlib import Path

from ..models.schemas import VoiceInfo
from .tts_base import BaseTTSService
from .tts_piper import PiperTTSService
from .tts_pyttsx3 import Pyttsx3TTSService

logger = logging.getLogger(__name__)


class CompositeTTSService(BaseTTSService):
    """Composite TTS service that prefers Piper with pyttsx3 fallback."""

    def __init__(
        self,
        cache_dir: str | None = None,
        # Piper settings
        piper_voice_path: str | None = None,
        piper_length_scale: float = 1.08,
        piper_noise_scale: float = 0.65,
        piper_noise_w: float = 0.80,
        piper_sentence_silence: float = 0.25,
    ):
        """Initialize composite TTS service.

        Args:
            cache_dir: Directory for caching synthesized audio
            piper_voice_path: Path to Piper voice model
            piper_length_scale: Piper speech speed multiplier
            piper_noise_scale: Piper speech variability
            piper_noise_w: Piper variance in speech timing
            piper_sentence_silence: Piper pause between sentences
        """
        self.cache_dir = cache_dir

        # Initialize both services
        self.piper_service = PiperTTSService(
            cache_dir=cache_dir,
            voice_path=piper_voice_path,
            length_scale=piper_length_scale,
            noise_scale=piper_noise_scale,
            noise_w=piper_noise_w,
            sentence_silence=piper_sentence_silence,
        )

        self.pyttsx3_service = Pyttsx3TTSService(cache_dir=cache_dir)

        # Log which services are available
        if self.piper_service.is_available:
            logger.info(f"Piper TTS available with voice: {piper_voice_path}")
        else:
            logger.info("Piper TTS not available, will use pyttsx3 fallback")

        if self.pyttsx3_service.is_available:
            logger.info("pyttsx3 TTS available as fallback")
        else:
            logger.warning("pyttsx3 TTS not available - no TTS fallback!")

    @property
    def is_available(self) -> bool:
        """Check if any TTS service is available."""
        return self.piper_service.is_available or self.pyttsx3_service.is_available

    @property
    def preferred_service_name(self) -> str:
        """Get name of preferred/active TTS service."""
        if self.piper_service.is_available:
            return "Piper"
        elif self.pyttsx3_service.is_available:
            return "pyttsx3"
        else:
            return "None"

    async def synthesize(self, text: str, voice_id: str | None = None, rate: int = 180) -> Path:
        """Synthesize text using preferred service with fallback.

        Args:
            text: Text to synthesize
            voice_id: Voice ID (used for pyttsx3 fallback)
            rate: Speech rate (used for pyttsx3 fallback)

        Returns:
            Path to synthesized WAV file

        Raises:
            RuntimeError: If no TTS service is available
            ValueError: If text is empty
        """
        if not text.strip():
            raise ValueError("Empty text")

        # Try Piper first if available
        if self.piper_service.is_available:
            try:
                logger.debug("Attempting synthesis with Piper TTS")
                result = await self.piper_service.synthesize(text, voice_id, rate)
                logger.info(f"Successfully synthesized with Piper: {result}")
                return result
            except Exception as e:
                logger.warning(f"Piper TTS failed, falling back to pyttsx3: {e}")

        # Fall back to pyttsx3
        if self.pyttsx3_service.is_available:
            try:
                logger.debug("Using pyttsx3 TTS fallback")
                result = await self.pyttsx3_service.synthesize(text, voice_id, rate)
                logger.info(f"Successfully synthesized with pyttsx3: {result}")
                return result
            except Exception as e:
                logger.error(f"pyttsx3 TTS fallback failed: {e}")
                raise

        # No TTS service available
        raise RuntimeError("No TTS service available (neither Piper nor pyttsx3)")

    async def list_voices(self) -> list[VoiceInfo]:
        """List available voices from all TTS services.

        Returns:
            Combined list of voices from Piper and pyttsx3
        """
        voices = []

        # Add Piper voices if available
        if self.piper_service.is_available:
            try:
                piper_voices = await self.piper_service.list_voices()
                voices.extend(piper_voices)
            except Exception as e:
                logger.warning(f"Failed to get Piper voices: {e}")

        # Add pyttsx3 voices if available
        if self.pyttsx3_service.is_available:
            try:
                pyttsx3_voices = await self.pyttsx3_service.list_voices()
                voices.extend(pyttsx3_voices)
            except Exception as e:
                logger.warning(f"Failed to get pyttsx3 voices: {e}")

        return voices
