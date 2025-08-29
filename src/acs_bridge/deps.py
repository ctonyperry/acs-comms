"""Dependency injection and service wiring."""

import logging
from functools import lru_cache

from .models.state import CallState
from .services import ACSClient, MediaStreamer, VoskSTTService, CompositeTTSService
from .settings import Settings, get_settings

logger = logging.getLogger(__name__)


@lru_cache()
def get_call_state() -> CallState:
    """Get singleton call state instance."""
    return CallState()


@lru_cache()
def get_acs_client(settings: Settings = None) -> ACSClient:
    """Get ACS client instance.

    Args:
        settings: Application settings (injected)

    Returns:
        ACS client instance
    """
    if settings is None:
        settings = get_settings()

    return ACSClient(settings.acs_connection_string)


@lru_cache()
def get_stt_service(settings: Settings = None) -> VoskSTTService:
    """Get STT service instance.

    Args:
        settings: Application settings (injected)

    Returns:
        STT service instance
    """
    if settings is None:
        settings = get_settings()

    return VoskSTTService(model_path=settings.stt_model_path)


@lru_cache()
def get_tts_service(settings: Settings = None) -> CompositeTTSService:
    """Get TTS service instance.

    Args:
        settings: Application settings (injected)

    Returns:
        Composite TTS service instance (Piper + pyttsx3 fallback)
    """
    if settings is None:
        settings = get_settings()

    return CompositeTTSService(
        piper_voice_path=settings.piper_voice_path,
        piper_length_scale=settings.piper_length_scale,
        piper_noise_scale=settings.piper_noise_scale,
        piper_noise_w=settings.piper_noise_w,
        piper_sentence_silence=settings.piper_sentence_silence,
    )


@lru_cache()
def get_media_streamer(
    stt_service: VoskSTTService = None, tts_service: CompositeTTSService = None
) -> MediaStreamer:
    """Get media streamer instance.

    Args:
        stt_service: STT service (injected)
        tts_service: TTS service (injected)

    Returns:
        Media streamer instance
    """
    if stt_service is None:
        stt_service = get_stt_service()
    if tts_service is None:
        tts_service = get_tts_service()

    return MediaStreamer(stt_service, tts_service)


# Dependency factories for FastAPI
def get_settings_dependency() -> Settings:
    """FastAPI dependency for settings."""
    return get_settings()


def get_call_state_dependency() -> CallState:
    """FastAPI dependency for call state."""
    return get_call_state()


def get_acs_client_dependency(settings: Settings = None) -> ACSClient:
    """FastAPI dependency for ACS client."""
    return get_acs_client(settings)


def get_media_streamer_dependency() -> MediaStreamer:
    """FastAPI dependency for media streamer."""
    return get_media_streamer()


def get_tts_service_dependency() -> CompositeTTSService:
    """FastAPI dependency for TTS service."""
    return get_tts_service()
