"""Tests for TTS service functionality."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.acs_bridge.services.tts_pyttsx3 import Pyttsx3TTSService
from src.acs_bridge.models.schemas import VoiceInfo


class TestPyttsx3TTSService:
    """Test cases for Pyttsx3TTSService."""
    
    def test_availability_without_pyttsx3(self):
        """Test service availability when pyttsx3 is not installed."""
        with patch("src.acs_bridge.services.tts_pyttsx3.TTS_AVAILABLE", False):
            service = Pyttsx3TTSService()
            assert not service.is_available
            
    def test_availability_with_pyttsx3(self):
        """Test service availability when pyttsx3 is available."""
        with patch("src.acs_bridge.services.tts_pyttsx3.TTS_AVAILABLE", True):
            service = Pyttsx3TTSService()
            assert service.is_available
            
    @pytest.mark.asyncio
    async def test_synthesize_without_pyttsx3(self):
        """Test synthesis fails when pyttsx3 is not available."""
        with patch("src.acs_bridge.services.tts_pyttsx3.TTS_AVAILABLE", False):
            service = Pyttsx3TTSService()
            
            with pytest.raises(RuntimeError, match="pyttsx3 not installed"):
                await service.synthesize("test text")
                
    @pytest.mark.asyncio
    async def test_synthesize_empty_text(self):
        """Test synthesis fails with empty text."""
        with patch("src.acs_bridge.services.tts_pyttsx3.TTS_AVAILABLE", True):
            service = Pyttsx3TTSService()
            
            with pytest.raises(ValueError, match="Empty text"):
                await service.synthesize("")
                
            with pytest.raises(ValueError, match="Empty text"):
                await service.synthesize("   ")
                
    @pytest.mark.asyncio
    @patch("src.acs_bridge.services.tts_pyttsx3.TTS_AVAILABLE", True)
    @patch("asyncio.to_thread")
    @patch("src.acs_bridge.audio.utils.ensure_16k_mono_wav")
    async def test_synthesize_success(self, mock_ensure_wav, mock_to_thread, temp_dir):
        """Test successful synthesis."""
        service = Pyttsx3TTSService(cache_dir=str(temp_dir))
        
        # Mock the sync synthesis
        mock_to_thread.return_value = None
        
        # Mock the WAV normalization
        expected_path = temp_dir / "normalized.wav"
        mock_ensure_wav.return_value = str(expected_path)
        
        result = await service.synthesize("Hello world", voice_id="test_voice", rate=200)
        
        # Verify calls
        mock_to_thread.assert_called_once()
        mock_ensure_wav.assert_called_once()
        
        assert result == expected_path
        
    @pytest.mark.asyncio
    async def test_list_voices_without_pyttsx3(self):
        """Test voice listing fails when pyttsx3 is not available."""
        with patch("src.acs_bridge.services.tts_pyttsx3.TTS_AVAILABLE", False):
            service = Pyttsx3TTSService()
            
            with pytest.raises(RuntimeError, match="pyttsx3 not installed"):
                await service.list_voices()
                
    @pytest.mark.asyncio
    @patch("src.acs_bridge.services.tts_pyttsx3.TTS_AVAILABLE", True)
    @patch("asyncio.to_thread")
    async def test_list_voices_success(self, mock_to_thread):
        """Test successful voice listing."""
        service = Pyttsx3TTSService()
        
        # Mock voice data
        mock_voice_data = [
            {"id": "voice1", "name": "Voice 1", "lang": ["en"]},
            {"id": "voice2", "name": "Voice 2", "lang": ["en-US"]},
        ]
        mock_to_thread.return_value = mock_voice_data
        
        result = await service.list_voices()
        
        # Verify result
        assert len(result) == 2
        assert all(isinstance(voice, VoiceInfo) for voice in result)
        assert result[0].id == "voice1"
        assert result[0].name == "Voice 1"
        assert result[1].id == "voice2"
        
    def test_synthesize_sync_mocked(self, temp_dir):
        """Test synchronous synthesis with mocked pyttsx3."""
        service = Pyttsx3TTSService(cache_dir=str(temp_dir))
        
        # Mock pyttsx3 engine
        mock_engine = MagicMock()
        
        with patch("src.acs_bridge.services.tts_pyttsx3.pyttsx3") as mock_pyttsx3:
            mock_pyttsx3.init.return_value = mock_engine
            
            output_path = temp_dir / "test.wav"
            service._synthesize_sync("test text", "voice1", 180, output_path)
            
            # Verify engine calls
            mock_pyttsx3.init.assert_called_once()
            mock_engine.setProperty.assert_any_call("voice", "voice1")
            mock_engine.setProperty.assert_any_call("rate", 180)
            mock_engine.save_to_file.assert_called_once_with("test text", str(output_path))
            mock_engine.runAndWait.assert_called_once()
            
    def test_list_voices_sync_mocked(self):
        """Test synchronous voice listing with mocked pyttsx3."""
        service = Pyttsx3TTSService()
        
        # Mock voice objects
        mock_voice1 = MagicMock()
        mock_voice1.id = "voice1"
        mock_voice1.name = "Voice 1"
        mock_voice1.languages = ["en"]
        
        mock_voice2 = MagicMock()
        mock_voice2.id = "voice2"
        mock_voice2.name = "Voice 2"
        mock_voice2.languages = None
        
        mock_engine = MagicMock()
        mock_engine.getProperty.return_value = [mock_voice1, mock_voice2]
        
        with patch("src.acs_bridge.services.tts_pyttsx3.pyttsx3") as mock_pyttsx3:
            mock_pyttsx3.init.return_value = mock_engine
            
            result = service._list_voices_sync()
            
            # Verify result
            assert len(result) == 2
            assert result[0]["id"] == "voice1"
            assert result[0]["name"] == "Voice 1"
            assert result[0]["lang"] == ["en"]
            assert result[1]["id"] == "voice2"
            assert result[1]["name"] == "Voice 2"
            assert result[1]["lang"] is None