"""Services package for business logic components."""

from .acs_client import ACSClient
from .guardrails import GuardrailsService
from .llm_base import BaseLLMService, LLMMessage, LLMResponse, LLMService
from .llm_ollama import OllamaLLMService
from .stt_base import BaseSTTService, STTService
from .stt_vosk import VoskSTTService
from .tts_base import BaseTTSService, TTSService
from .tts_composite import CompositeTTSService
from .tts_piper import PiperTTSService
from .tts_pyttsx3 import Pyttsx3TTSService

# Optional imports that may fail in test environments
MediaStreamer = None
MEDIA_STREAMER_AVAILABLE = False

try:
    from .media_streamer import MediaStreamer
    MEDIA_STREAMER_AVAILABLE = True
except (ImportError, OSError):
    # OSError for PortAudio library not found, ImportError for missing modules
    pass

__all__ = [
    "ACSClient",
    "GuardrailsService",
    "LLMService",
    "BaseLLMService", 
    "LLMMessage",
    "LLMResponse",
    "OllamaLLMService",
    "STTService",
    "BaseSTTService",
    "VoskSTTService",
    "TTSService",
    "BaseTTSService",
    "CompositeTTSService",
    "PiperTTSService",
    "Pyttsx3TTSService",
]

# Add MediaStreamer to exports if available
if MEDIA_STREAMER_AVAILABLE:
    __all__.append("MediaStreamer")
