"""Call state management."""

from typing import Optional
from fastapi import WebSocket


class CallState:
    """Manages the state of an active call."""
    
    def __init__(self):
        self.ws: Optional[WebSocket] = None
        self.seq: int = 0
        self.muted: bool = False
        self.call_connection_id: Optional[str] = None
        
    @property
    def has_active_call(self) -> bool:
        """Check if there's an active call."""
        return self.ws is not None and self.call_connection_id is not None
        
    def start_call(self, websocket: WebSocket, call_connection_id: str) -> None:
        """Start a new call."""
        self.ws = websocket
        self.call_connection_id = call_connection_id
        self.seq = 0
        self.muted = False
        
    def end_call(self) -> None:
        """End the current call."""
        self.ws = None
        self.call_connection_id = None
        self.seq = 0
        self.muted = False