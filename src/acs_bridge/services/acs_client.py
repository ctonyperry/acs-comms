"""Azure Communication Services client wrapper."""

import logging
from typing import Any

from azure.communication.callautomation import (
    CallAutomationClient,
    MediaStreamingOptions,
    StreamingTransportType,
    MediaStreamingContentType,
    MediaStreamingAudioChannelType,
    AudioFormat,
)

logger = logging.getLogger(__name__)


class ACSClient:
    """Wrapper for Azure Communication Services Call Automation client."""

    def __init__(self, connection_string: str):
        """Initialize ACS client.

        Args:
            connection_string: ACS connection string
        """
        self.connection_string = connection_string
        self._client = CallAutomationClient.from_connection_string(connection_string)
        logger.info("ACS client initialized")

    def answer_call(self, incoming_call_context: str, callback_url: str, websocket_url: str) -> Any:
        """Answer an incoming call with media streaming.

        Args:
            incoming_call_context: ACS incoming call context
            callback_url: Callback URL for call events
            websocket_url: WebSocket URL for media streaming

        Returns:
            Call connection response
        """
        logger.info(f"Answering call with WebSocket: {websocket_url}")

        media_options = MediaStreamingOptions(
            transport_url=websocket_url,
            transport_type=StreamingTransportType.WEBSOCKET,
            content_type=MediaStreamingContentType.AUDIO,
            audio_channel_type=MediaStreamingAudioChannelType.MIXED,
            start_media_streaming=True,
            enable_bidirectional=True,
            audio_format=AudioFormat.PCM16_K_MONO,
        )

        try:
            response = self._client.answer_call(
                incoming_call_context=incoming_call_context,
                callback_url=callback_url,
                media_streaming=media_options,
            )

            call_connection_id = response.call_connection.call_connection_id
            logger.info(f"Call answered successfully, connection ID: {call_connection_id}")
            return response

        except Exception as e:
            logger.error(f"Failed to answer call: {e}")
            raise

    def hang_up_call(self, call_connection_id: str) -> None:
        """Hang up an active call.

        Args:
            call_connection_id: Call connection ID to hang up
        """
        try:
            call_connection = self._client.get_call_connection(call_connection_id)
            call_connection.hang_up(is_for_everyone=True)
            logger.info(f"Call {call_connection_id} hung up successfully")

        except Exception as e:
            logger.error(f"Failed to hang up call {call_connection_id}: {e}")
            raise
