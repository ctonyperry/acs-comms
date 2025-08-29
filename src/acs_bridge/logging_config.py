"""Logging configuration for ACS Bridge application."""

import logging.config
from typing import Dict, Any


def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration dictionary."""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "default",
                "stream": "ext://sys.stdout",
            },
            "detailed_console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "acs_bridge": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "acs_bridge.services.stt_vosk": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "acs_bridge.services.tts_pyttsx3": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "acs_bridge.services.media_streamer": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
        },
        "root": {
            "level": "WARNING",
            "handlers": ["console"],
        },
    }


def setup_logging() -> None:
    """Configure logging for the application."""
    logging.config.dictConfig(get_logging_config())

    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.info("Logging configuration initialized")
