"""Media WebSocket router for bidirectional audio streaming."""

import base64
import json
import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from ..deps import get_call_state_dependency, get_media_streamer_dependency
from ..models.state import CallState
from ..services.media_streamer import MediaStreamer

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    call_state: CallState = Depends(get_call_state_dependency),
    media_streamer: MediaStreamer = Depends(get_media_streamer_dependency),
) -> None:
    """WebSocket endpoint for bidirectional audio streaming.

    Handles:
    - WebSocket connection lifecycle
    - Bidirectional audio streaming (mic <-> phone)
    - Audio recording
    - STT processing and TTS responses
    """
    await websocket.accept()
    call_state.ws = websocket
    logger.info("WebSocket connection established")

    try:
        # Start media streaming
        await media_streamer.start_streaming(websocket, call_state)

        # Main message loop
        frames_received = 0
        async for message in websocket.iter_text():
            await _process_websocket_message(message, call_state, media_streamer, frames_received)
            frames_received += 1

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await _cleanup_websocket_connection(call_state, media_streamer)


async def _process_websocket_message(
    message: str,
    call_state: CallState,
    media_streamer: MediaStreamer,
    frames_received: int,
) -> None:
    """Process a single WebSocket message.

    Args:
        message: Raw WebSocket message
        call_state: Current call state
        media_streamer: Media streaming service
        frames_received: Number of frames received so far
    """
    try:
        data = json.loads(message)
        message_kind = (data.get("kind") or "").lower()

        if message_kind == "audiometadata":
            logger.info(f"Audio metadata: {data}")

        elif message_kind == "audiodata":
            await _process_audio_data(data, media_streamer, frames_received)

        else:
            logger.debug(f"Unknown message kind: {message_kind}")

    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in WebSocket message: {e}")
    except Exception as e:
        logger.error(f"Error processing WebSocket message: {e}")


async def _process_audio_data(
    data: Dict[str, Any],
    media_streamer: MediaStreamer,
    frames_received: int,
) -> None:
    """Process audio data from WebSocket.

    Args:
        data: Audio data message
        media_streamer: Media streaming service
        frames_received: Number of frames received so far
    """
    audio_data = data.get("audioData") or data.get("audiodata") or {}
    b64_data = audio_data.get("data")

    if not b64_data:
        return

    try:
        # Decode base64 audio data
        audio_bytes = base64.b64decode(b64_data)

        # Process through media streamer
        await media_streamer.process_incoming_audio(audio_bytes)

        # Log progress periodically
        if frames_received % 50 == 0:
            logger.debug(f"Processed {frames_received} audio frames")

    except Exception as e:
        logger.error(f"Error processing audio data: {e}")


async def _cleanup_websocket_connection(
    call_state: CallState,
    media_streamer: MediaStreamer,
) -> None:
    """Cleanup WebSocket connection and associated resources.

    Args:
        call_state: Current call state
        media_streamer: Media streaming service
    """
    logger.info("Cleaning up WebSocket connection")

    try:
        # Stop media streaming
        await media_streamer.stop_streaming()

        # Reset call state
        call_state.reset()

        logger.info("WebSocket cleanup completed")

    except Exception as e:
        logger.error(f"Error during WebSocket cleanup: {e}")
