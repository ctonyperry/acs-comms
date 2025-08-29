#!/usr/bin/env python3
"""Entry point for running the ACS Bridge application locally."""

import uvicorn

from src.acs_bridge.main import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
    )
