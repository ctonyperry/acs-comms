"""Text-to-Speech base interface definition."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Protocol

from ..models.schemas import VoiceInfo


class TTSService(Protocol):
    """Protocol for Text-to-Speech services."""
    
    @property
    def is_available(self) -> bool:
        """Check if TTS service is available."""
        ...
        
    async def synthesize(self, text: str, voice_id: Optional[str] = None, rate: int = 180) -> Path:
        """Synthesize text to speech and return path to WAV file."""
        ...
        
    async def list_voices(self) -> List[VoiceInfo]:
        """List available voices."""
        ...


class BaseTTSService(ABC):
    """Base class for TTS service implementations."""
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if TTS service is available."""
        pass
        
    @abstractmethod
    async def synthesize(self, text: str, voice_id: Optional[str] = None, rate: int = 180) -> Path:
        """Synthesize text to speech."""
        pass
        
    @abstractmethod
    async def list_voices(self) -> List[VoiceInfo]:
        """List available voices."""
        pass