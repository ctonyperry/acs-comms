"""Dependency injection and service wiring."""

import logging
from functools import lru_cache

from .models.state import CallState
from .services import ACSClient, MediaStreamer, VoskSTTService, Pyttsx3TTSService
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
def get_tts_service() -> Pyttsx3TTSService:
    """Get TTS service instance.
    
    Returns:
        TTS service instance
    """
    return Pyttsx3TTSService()


@lru_cache()
def get_media_streamer(
    stt_service: VoskSTTService = None,
    tts_service: Pyttsx3TTSService = None
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


def get_tts_service_dependency() -> Pyttsx3TTSService:
    """FastAPI dependency for TTS service."""
    return get_tts_service()