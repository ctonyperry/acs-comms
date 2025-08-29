"""Routers package for FastAPI route handlers."""

from .events import router as events_router
from .media_ws import router as media_ws_router  
from .controls import router as controls_router

__all__ = [
    "events_router",
    "media_ws_router", 
    "controls_router",
]