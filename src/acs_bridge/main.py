"""FastAPI application factory and main entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .logging_config import setup_logging
from .routers import events_router, media_ws_router, controls_router
from .settings import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting ACS Bridge application")
    settings = get_settings()
    logger.info(f"Loaded settings - Public base: {settings.public_base}")
    logger.info(f"STT model path: {settings.stt_model_path or 'Not configured'}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ACS Bridge application")


def create_app() -> FastAPI:
    """Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    # Setup logging first
    setup_logging()
    
    # Create FastAPI app
    app = FastAPI(
        title="ACS Bridge",
        description="Azure Communication Services bidirectional audio bridge",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # Mount routers
    app.include_router(events_router, tags=["events"])
    app.include_router(media_ws_router, tags=["media"])
    app.include_router(controls_router, tags=["controls"])
    
    logger.info("FastAPI application created and configured")
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "acs_bridge.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
    )