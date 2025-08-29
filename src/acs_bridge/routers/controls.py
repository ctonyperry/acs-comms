"""Control API router for call management operations."""

import logging
import os
from typing import Dict, Any

from fastapi import APIRouter, Body, Depends, HTTPException

from ..audio.utils import ensure_16k_mono_wav
from ..deps import (
    get_call_state_dependency,
    get_acs_client_dependency,
    get_media_streamer_dependency,
    get_tts_service_dependency,
)
from ..models.schemas import (
    MuteResponse,
    PlayResponse,
    SayResponse,
    HangupResponse,
    VoicesResponse,
    HealthResponse,
)
from ..models.state import CallState
from ..services.acs_client import ACSClient
from ..services.media_streamer import MediaStreamer
from ..services.tts_composite import CompositeTTSService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


@router.post("/mute", response_model=MuteResponse)
async def api_mute(
    on: bool = Body(embed=True),
    call_state: CallState = Depends(get_call_state_dependency),
) -> MuteResponse:
    """Mute or unmute the microphone.

    Args:
        on: True to mute, False to unmute
        call_state: Current call state

    Returns:
        Current mute status

    Raises:
        HTTPException: If no active call
    """
    if not call_state.has_active_call:
        raise HTTPException(status_code=409, detail="No active call")

    call_state.muted = bool(on)
    logger.info(f"Microphone {'muted' if call_state.muted else 'unmuted'}")

    return MuteResponse(muted=call_state.muted)


@router.post("/play", response_model=PlayResponse)
async def api_play(
    file: str = Body(embed=True),
    call_state: CallState = Depends(get_call_state_dependency),
    media_streamer: MediaStreamer = Depends(get_media_streamer_dependency),
) -> PlayResponse:
    """Play an audio file to the caller.

    Args:
        file: Path to audio file to play
        call_state: Current call state
        media_streamer: Media streaming service

    Returns:
        Playback result with file path and sequence number

    Raises:
        HTTPException: If no active call or file issues
    """
    if not call_state.has_active_call:
        raise HTTPException(status_code=409, detail="No active call")

    if not file or not os.path.isfile(file):
        raise HTTPException(status_code=400, detail=f"File not found: {file}")

    try:
        # Normalize audio file
        normalized_file = ensure_16k_mono_wav(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to prepare WAV: {e}")

    try:
        # Play file through media streamer
        await media_streamer.play_audio_file(normalized_file, call_state.ws, call_state)
        logger.info(f"Successfully played file: {normalized_file}")

    except Exception as e:
        logger.error(f"Failed to play file {normalized_file}: {e}")
        raise HTTPException(status_code=500, detail=f"Playback failed: {e}")

    return PlayResponse(played=normalized_file, seq=call_state.seq)


@router.get("/voices", response_model=VoicesResponse)
async def api_voices(
    tts_service: CompositeTTSService = Depends(get_tts_service_dependency),
) -> VoicesResponse:
    """List available TTS voices.

    Args:
        tts_service: TTS service instance

    Returns:
        List of available voices

    Raises:
        HTTPException: If TTS not available or voice enumeration fails
    """
    if not tts_service.is_available:
        raise HTTPException(
            status_code=501, detail="No TTS service available (install piper and/or pyttsx3)"
        )

    try:
        voices = await tts_service.list_voices()
        logger.info(f"Listed {len(voices)} available voices")
        return VoicesResponse(voices=voices)

    except Exception as e:
        logger.error(f"Voice enumeration failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS voice enumeration failed: {e}")


@router.post("/say", response_model=SayResponse)
async def api_say(
    payload: Dict[str, Any] = Body(...),
    call_state: CallState = Depends(get_call_state_dependency),
    media_streamer: MediaStreamer = Depends(get_media_streamer_dependency),
    tts_service: CompositeTTSService = Depends(get_tts_service_dependency),
) -> SayResponse:
    """Synthesize text to speech and play to caller.

    Args:
        payload: Request payload with text, voice, and rate
        call_state: Current call state
        media_streamer: Media streaming service
        tts_service: TTS service instance

    Returns:
        TTS result with file path and sequence number

    Raises:
        HTTPException: If no active call, TTS not available, or synthesis fails
    """
    if not call_state.has_active_call:
        raise HTTPException(status_code=409, detail="No active call")

    if not tts_service.is_available:
        raise HTTPException(
            status_code=501, detail="No TTS service available (install piper and/or pyttsx3)"
        )

    # Extract parameters
    text = (payload.get("text") or "").strip()
    voice = payload.get("voice")  # voice ID from /api/voices
    rate = int(payload.get("rate", 180))

    if not text:
        raise HTTPException(status_code=400, detail="field 'text' is required")

    try:
        # Synthesize speech
        wav_path = await tts_service.synthesize(text, voice, rate)

        # Normalize audio
        normalized_path = ensure_16k_mono_wav(str(wav_path))

    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")

    try:
        # Play synthesized audio
        await media_streamer.play_audio_file(normalized_path, call_state.ws, call_state)
        logger.info(f"Successfully played TTS: {normalized_path}")

    except Exception as e:
        logger.error(f"TTS playback failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS playback failed: {e}")

    return SayResponse(ok=True, played=normalized_path, seq=call_state.seq)


@router.post("/hangup", response_model=HangupResponse)
async def api_hangup(
    call_state: CallState = Depends(get_call_state_dependency),
    acs_client: ACSClient = Depends(get_acs_client_dependency),
) -> HangupResponse:
    """Hang up the active call.

    Args:
        call_state: Current call state
        acs_client: ACS client instance

    Returns:
        Hangup confirmation

    Raises:
        HTTPException: If no active call or hangup fails
    """
    call_connection_id = call_state.call_connection_id
    if not call_connection_id:
        raise HTTPException(status_code=409, detail="No active call")

    try:
        acs_client.hang_up_call(call_connection_id)
        logger.info(f"Call {call_connection_id} hung up successfully")

    except Exception as e:
        logger.error(f"Hangup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Hangup failed: {e}")

    return HangupResponse(hung_up=True)


@router.get("/health", response_model=HealthResponse)
async def api_health(
    call_state: CallState = Depends(get_call_state_dependency),
) -> HealthResponse:
    """Get application health status.

    Args:
        call_state: Current call state

    Returns:
        Health status information
    """
    return HealthResponse(
        ok=True,
        ws_active=call_state.has_active_call,
        muted=call_state.muted,
        seq=call_state.seq,
    )
