"""Text-to-Speech base interface definition."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol

from ..models.schemas import VoiceInfo


class TTSService(Protocol):
    """Protocol for Text-to-Speech services."""

    @property
    def is_available(self) -> bool:
        """Check if TTS service is available."""
        ...

    async def synthesize(self, text: str, voice_id: str | None = None, rate: int = 180) -> Path:
        """Synthesize text to speech and return path to WAV file."""
        ...

    async def list_voices(self) -> list[VoiceInfo]:
        """List available voices."""
        ...


class BaseTTSService(ABC):
    """Base class for TTS service implementations."""

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if TTS service is available."""
        raise NotImplementedError

    @abstractmethod
    async def synthesize(self, text: str, voice_id: str | None = None, rate: int = 180) -> Path:
        """Synthesize text to speech."""
        raise NotImplementedError

    @abstractmethod
    async def list_voices(self) -> list[VoiceInfo]:
        """List available voices."""
        raise NotImplementedError
