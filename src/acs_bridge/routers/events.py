"""Event handling router for Azure Communication Services webhooks."""

import logging
import traceback
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from ..deps import get_acs_client_dependency, get_call_state_dependency, get_settings_dependency
from ..models.state import CallState
from ..services.acs_client import ACSClient
from ..settings import Settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/events")
async def handle_events(
    request: Request,
    call_state: CallState = Depends(get_call_state_dependency),
    acs_client: ACSClient = Depends(get_acs_client_dependency),
    settings: Settings = Depends(get_settings_dependency),
) -> JSONResponse:
    """Handle Azure Communication Services events.

    Processes Event Grid subscription validation and ACS call events.
    """
    body = await request.json()

    # Handle Event Grid subscription validation
    if (
        isinstance(body, list)
        and body
        and body[0].get("eventType") == "Microsoft.EventGrid.SubscriptionValidationEvent"
    ):
        validation_code = body[0]["data"]["validationCode"]
        logger.info(f"Event Grid validation: {validation_code}")
        return JSONResponse({"validationResponse": validation_code})

    # Handle ACS events
    if isinstance(body, list):
        for event in body:
            await _process_acs_event(event, call_state, acs_client, settings)

    return JSONResponse({"ok": True})


async def _process_acs_event(
    event: Dict[str, Any],
    call_state: CallState,
    acs_client: ACSClient,
    settings: Settings,
) -> None:
    """Process a single ACS event.

    Args:
        event: ACS event data
        call_state: Current call state
        acs_client: ACS client instance
        settings: Application settings
    """
    event_type = event.get("eventType")
    event_data = event.get("data", {})

    if event_type == "Microsoft.Communication.IncomingCall":
        await _handle_incoming_call(event_data, call_state, acs_client, settings)
    else:
        logger.info(f"Received event: {event_type}")


async def _handle_incoming_call(
    event_data: Dict[str, Any],
    call_state: CallState,
    acs_client: ACSClient,
    settings: Settings,
) -> None:
    """Handle incoming call event.

    Args:
        event_data: Incoming call event data
        call_state: Current call state
        acs_client: ACS client instance
        settings: Application settings
    """
    incoming_call_context = event_data["incomingCallContext"]

    # Construct URLs
    ws_url = f"{settings.public_base.replace('https://', 'wss://')}/ws"
    callback_url = f"{settings.public_base}/events"

    logger.info(f"Incoming call - WebSocket URL: {ws_url}")
    logger.info(f"Incoming call - Callback URL: {callback_url}")

    try:
        # Answer the call
        response = acs_client.answer_call(
            incoming_call_context=incoming_call_context,
            callback_url=callback_url,
            websocket_url=ws_url,
        )

        # Store call connection ID
        call_state.call_connection_id = response.call_connection.call_connection_id
        logger.info(f"Call answered successfully, connection ID: {call_state.call_connection_id}")

    except Exception as e:
        logger.error(f"Failed to answer call: {e}")
        logger.error(traceback.format_exc())
