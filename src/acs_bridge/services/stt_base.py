"""Speech-to-Text base interface definition."""

import asyncio
from abc import ABC, abstractmethod
from typing import Protocol


class STTService(Protocol):
    """Protocol for Speech-to-Text services."""

    @property
    def is_available(self) -> bool:
        """Check if STT service is available."""
        ...

    async def start_processing(self, transcript_queue: asyncio.Queue[str]) -> None:
        """Start STT processing with output queue for transcripts."""
        ...

    async def process_audio_chunk(self, audio_data: bytes) -> None:
        """Process an audio chunk for speech recognition."""
        ...

    async def stop_processing(self) -> None:
        """Stop STT processing and cleanup resources."""
        ...


class BaseSTTService(ABC):
    """Base class for STT service implementations."""

    def __init__(self):
        self._is_running = False
        self._transcript_queue: asyncio.Queue[str] | None = None

    @property
    def is_running(self) -> bool:
        """Check if STT service is currently running."""
        return self._is_running

    @abstractmethod
    async def start_processing(self, transcript_queue: asyncio.Queue[str]) -> None:
        """Start STT processing."""
        pass

    @abstractmethod
    async def process_audio_chunk(self, audio_data: bytes) -> None:
        """Process audio chunk."""
        pass

    @abstractmethod
    async def stop_processing(self) -> None:
        """Stop processing."""
        pass
