"""Pydantic schemas for API requests and responses."""

from typing import Optional, List
from pydantic import BaseModel


class VoiceInfo(BaseModel):
    """Information about a TTS voice."""
    id: str
    name: str
    lang: Optional[str] = None


class PlayResponse(BaseModel):
    """Response from play endpoint."""
    played: str
    seq: int


class SayResponse(BaseModel):
    """Response from say endpoint.""" 
    text: str
    voice_id: Optional[str]
    seq: int


class HangupResponse(BaseModel):
    """Response from hangup endpoint."""
    hung_up: bool


class MuteResponse(BaseModel):
    """Response from mute endpoint."""
    muted: bool


class HealthResponse(BaseModel):
    """Response from health endpoint."""
    ok: bool
    ws_active: bool
    muted: bool
    seq: int


class VoicesResponse(BaseModel):
    """Response from voices endpoint."""
    voices: List[VoiceInfo]
    preferred_service: str