"""Test configuration for pytest."""

import pytest
import asyncio
from pathlib import Path
import tempfile
import os


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_wav_file(temp_dir):
    """Create a test WAV file."""
    import wave

    wav_path = temp_dir / "test.wav"

    # Create a simple 16kHz mono WAV file
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # Write 1 second of silence
        wf.writeframes(b"\x00\x00" * 16000)

    return wav_path


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    from src.acs_bridge.settings import Settings

    # Mock environment variables
    test_env = {
        "ACS_CONNECTION_STRING": "endpoint=https://test.communication.azure.com/;accesskey=test123",
        "PUBLIC_BASE": "https://test.ngrok-free.app",
        "STT_MODEL_PATH": "",
    }

    # Patch environment
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        settings = Settings()
        yield settings
    finally:
        # Restore environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
