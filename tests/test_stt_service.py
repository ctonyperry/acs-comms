"""Tests for STT service functionality."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.acs_bridge.services.stt_vosk import VoskSTTService


class TestVoskSTTService:
    """Test cases for VoskSTTService."""

    def test_availability_without_vosk(self):
        """Test service availability when Vosk is not installed."""
        with patch("src.acs_bridge.services.stt_vosk.VOSK_AVAILABLE", False):
            service = VoskSTTService(model_path="/fake/path")
            assert not service.is_available

    def test_availability_without_model_path(self):
        """Test service availability without model path."""
        with patch("src.acs_bridge.services.stt_vosk.VOSK_AVAILABLE", True):
            service = VoskSTTService(model_path=None)
            assert not service.is_available

    def test_availability_with_invalid_model(self):
        """Test service availability with invalid model."""
        with (
            patch("src.acs_bridge.services.stt_vosk.VOSK_AVAILABLE", True),
            patch(
                "src.acs_bridge.services.stt_vosk.VoskModel", side_effect=Exception("Invalid model")
            ),
        ):
            service = VoskSTTService(model_path="/fake/path")
            assert not service.is_available

    def test_availability_with_valid_model(self):
        """Test service availability with valid model."""
        mock_model = MagicMock()
        with (
            patch("src.acs_bridge.services.stt_vosk.VOSK_AVAILABLE", True),
            patch("src.acs_bridge.services.stt_vosk.VoskModel", return_value=mock_model),
        ):
            service = VoskSTTService(model_path="/fake/path")
            assert service.is_available

    @pytest.mark.asyncio
    async def test_start_processing_unavailable(self):
        """Test starting processing when service is unavailable."""
        with patch("src.acs_bridge.services.stt_vosk.VOSK_AVAILABLE", False):
            service = VoskSTTService(model_path="/fake/path")
            transcript_queue = asyncio.Queue()

            await service.start_processing(transcript_queue)

            assert not service.is_running

    @pytest.mark.asyncio
    async def test_start_processing_already_running(self):
        """Test starting processing when already running."""
        mock_model = MagicMock()
        with (
            patch("src.acs_bridge.services.stt_vosk.VOSK_AVAILABLE", True),
            patch("src.acs_bridge.services.stt_vosk.VoskModel", return_value=mock_model),
        ):
            service = VoskSTTService(model_path="/fake/path")
            service._is_running = True

            transcript_queue = asyncio.Queue()
            await service.start_processing(transcript_queue)

            # Should not start a new thread
            assert service._worker_thread is None

    @pytest.mark.asyncio
    async def test_process_audio_chunk_not_running(self):
        """Test processing audio when not running."""
        service = VoskSTTService(model_path="/fake/path")

        # Should not raise any exceptions
        await service.process_audio_chunk(b"audio_data")

    @pytest.mark.asyncio
    async def test_stop_processing_not_running(self):
        """Test stopping processing when not running."""
        service = VoskSTTService(model_path="/fake/path")

        # Should not raise any exceptions
        await service.stop_processing()

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Test full STT lifecycle with mocked Vosk."""
        mock_model = MagicMock()
        mock_recognizer = MagicMock()

        with (
            patch("src.acs_bridge.services.stt_vosk.VOSK_AVAILABLE", True),
            patch("src.acs_bridge.services.stt_vosk.VoskModel", return_value=mock_model),
            patch("src.acs_bridge.services.stt_vosk.KaldiRecognizer", return_value=mock_recognizer),
            patch("threading.Thread") as mock_thread_class,
        ):

            # Mock thread
            mock_thread = MagicMock()
            mock_thread_class.return_value = mock_thread

            service = VoskSTTService(model_path="/fake/path")
            transcript_queue = asyncio.Queue()

            # Start processing
            await service.start_processing(transcript_queue)

            assert service.is_running
            assert service._transcript_queue is transcript_queue
            mock_thread.start.assert_called_once()

            # Process audio chunk
            await service.process_audio_chunk(b"audio_data")

            # Stop processing
            await service.stop_processing()

            assert not service.is_running
            mock_thread.join.assert_called_once()

    def test_stt_worker_with_final_result(self):
        """Test STT worker thread with final recognition result."""
        import json
        import queue

        mock_model = MagicMock()
        mock_recognizer = MagicMock()

        # Mock successful recognition
        mock_recognizer.AcceptWaveform.return_value = True
        mock_recognizer.Result.return_value = json.dumps({"text": "hello world"})

        with (
            patch("src.acs_bridge.services.stt_vosk.VoskModel", return_value=mock_model),
            patch("src.acs_bridge.services.stt_vosk.KaldiRecognizer", return_value=mock_recognizer),
        ):

            service = VoskSTTService(model_path="/fake/path")
            service._transcript_queue = MagicMock()
            service._stt_queue = queue.Queue()
            service._should_stop = MagicMock()
            service._should_stop.is_set.side_effect = [False, True]  # Run once, then stop

            # Add test data
            service._stt_queue.put(b"audio_data")
            service._stt_queue.put(None)  # Stop signal

            # Run worker
            service._stt_worker()

            # Verify calls
            mock_recognizer.AcceptWaveform.assert_called_once_with(b"audio_data")
            mock_recognizer.Result.assert_called_once()
            service._transcript_queue.put_nowait.assert_called_once_with("hello world")

    def test_stt_worker_with_partial_result(self):
        """Test STT worker thread with partial recognition result."""
        import json
        import queue

        mock_model = MagicMock()
        mock_recognizer = MagicMock()

        # Mock partial recognition
        mock_recognizer.AcceptWaveform.return_value = False
        mock_recognizer.PartialResult.return_value = json.dumps({"partial": "hello"})

        with (
            patch("src.acs_bridge.services.stt_vosk.VoskModel", return_value=mock_model),
            patch("src.acs_bridge.services.stt_vosk.KaldiRecognizer", return_value=mock_recognizer),
        ):

            service = VoskSTTService(model_path="/fake/path")
            service._transcript_queue = MagicMock()
            service._stt_queue = queue.Queue()
            service._should_stop = MagicMock()
            service._should_stop.is_set.side_effect = [False, True]  # Run once, then stop

            # Add test data
            service._stt_queue.put(b"audio_data")
            service._stt_queue.put(None)  # Stop signal

            # Run worker
            service._stt_worker()

            # Verify calls
            mock_recognizer.AcceptWaveform.assert_called_once_with(b"audio_data")
            mock_recognizer.PartialResult.assert_called_once()
            # Should not emit partial results
            service._transcript_queue.put_nowait.assert_not_called()
