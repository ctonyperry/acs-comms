"""Tests for Piper TTS service functionality."""

from unittest.mock import patch

import pytest

from src.acs_bridge.services.tts_piper import PiperTTSService, check_piper_available


class TestPiperTTSService:
    """Test cases for PiperTTSService."""

    def test_piper_availability_without_executable(self):
        """Test Piper availability when executable is not installed."""
        with patch("shutil.which", return_value=None):
            assert not check_piper_available()
            service = PiperTTSService(voice_path="./voices/test.onnx")
            assert not service.is_available

    def test_piper_availability_without_voice_file(self):
        """Test Piper availability when voice file doesn't exist."""
        with patch("shutil.which", return_value="/usr/bin/piper"):
            with patch("os.path.isfile", return_value=False):
                service = PiperTTSService(voice_path="./voices/nonexistent.onnx")
                assert not service.is_available

    def test_piper_availability_with_valid_setup(self):
        """Test Piper availability when properly configured."""
        with patch("shutil.which", return_value="/usr/bin/piper"):
            with patch("os.path.isfile", return_value=True):
                service = PiperTTSService(voice_path="./voices/test.onnx")
                assert service.is_available

    @pytest.mark.asyncio
    async def test_synthesize_without_piper(self, temp_dir):
        """Test synthesis fails when Piper is not available."""
        service = PiperTTSService(cache_dir=str(temp_dir))
        # Service should not be available by default (no Piper installed)
        assert not service.is_available

        with pytest.raises(RuntimeError, match="Piper TTS not available"):
            await service.synthesize("Hello world")

    @pytest.mark.asyncio
    async def test_synthesize_empty_text(self, temp_dir):
        """Test synthesis fails with empty text."""
        with patch("shutil.which", return_value="/usr/bin/piper"):
            with patch("os.path.isfile", return_value=True):
                service = PiperTTSService(
                    cache_dir=str(temp_dir),
                    voice_path="./voices/test.onnx"
                )

                with pytest.raises(ValueError, match="Empty text"):
                    await service.synthesize("")

    @pytest.mark.asyncio
    async def test_list_voices_without_piper(self, temp_dir):
        """Test voice listing returns empty when Piper not available."""
        service = PiperTTSService(cache_dir=str(temp_dir))
        voices = await service.list_voices()
        assert voices == []

    @pytest.mark.asyncio
    async def test_list_voices_with_piper(self, temp_dir):
        """Test voice listing with Piper available."""
        with patch("shutil.which", return_value="/usr/bin/piper"):
            with patch("os.path.isfile", return_value=True):
                service = PiperTTSService(
                    cache_dir=str(temp_dir),
                    voice_path="./voices/en-us-high.onnx"
                )

                voices = await service.list_voices()
                assert len(voices) == 1
                assert voices[0].id == "piper-voice"
                assert "en-us-high" in voices[0].name
                assert voices[0].lang == "en-US"


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for testing."""
    return tmp_path
