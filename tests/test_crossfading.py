"""Tests for audio crossfading functionality."""

import pytest
import wave
import tempfile
import os
from pathlib import Path

from src.acs_bridge.audio.utils import crossfade_wav_files, get_wav_duration


class TestAudioCrossfading:
    """Test cases for audio crossfading functions."""

    def create_test_wav(self, duration_sec: float, frequency: int = 440) -> str:
        """Create a test WAV file with specified duration."""
        # Create a simple sine wave
        import numpy as np

        sample_rate = 16000
        frames = int(sample_rate * duration_sec)

        # Generate sine wave
        t = np.linspace(0, duration_sec, frames, False)
        audio = np.sin(2 * np.pi * frequency * t)

        # Convert to 16-bit PCM
        audio = (audio * 32767).astype(np.int16)

        # Create temporary file
        fd, path = tempfile.mkstemp(suffix=".wav")
        with os.fdopen(fd, "wb"):
            pass  # Close file descriptor

        # Write WAV file
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio.tobytes())

        return path

    def test_single_file_crossfade(self):
        """Test crossfading with single file returns the same file."""
        test_wav = self.create_test_wav(1.0)
        try:
            result = crossfade_wav_files([test_wav])
            assert result == test_wav
        finally:
            os.unlink(test_wav)

    def test_empty_file_list_raises_error(self):
        """Test that empty file list raises ValueError."""
        with pytest.raises(ValueError, match="At least one WAV file required"):
            crossfade_wav_files([])

    def test_get_wav_duration(self):
        """Test WAV duration calculation."""
        test_wav = self.create_test_wav(2.5)  # 2.5 seconds
        try:
            duration = get_wav_duration(test_wav)
            assert abs(duration - 2.5) < 0.1  # Allow small tolerance
        finally:
            os.unlink(test_wav)

    @pytest.mark.skipif(
        not os.system("which ffmpeg > /dev/null 2>&1") == 0, reason="ffmpeg not available"
    )
    def test_two_file_crossfade(self):
        """Test crossfading two WAV files."""
        # Create two test files
        wav1 = self.create_test_wav(1.0, frequency=440)  # A4
        wav2 = self.create_test_wav(1.0, frequency=880)  # A5

        try:
            # Create output file
            fd, output_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

            # Crossfade the files
            result = crossfade_wav_files([wav1, wav2], crossfade_ms=50, output_path=output_path)

            # Verify result
            assert result == output_path
            assert os.path.exists(output_path)

            # Check that output duration is approximately sum minus crossfade
            # Original: 1.0 + 1.0 = 2.0 seconds
            # Crossfade: 0.05 seconds overlap
            # Expected: ~1.95 seconds
            duration = get_wav_duration(output_path)
            assert 1.9 < duration < 2.0

            # Cleanup
            os.unlink(output_path)

        finally:
            os.unlink(wav1)
            os.unlink(wav2)

    def test_crossfade_with_missing_ffmpeg(self):
        """Test that crossfade fails gracefully without ffmpeg."""
        from unittest.mock import patch

        with patch("src.acs_bridge.audio.utils.shutil.which", return_value=None):
            wav1 = self.create_test_wav(0.5)
            wav2 = self.create_test_wav(0.5)

            try:
                with pytest.raises(RuntimeError, match="ffmpeg is required"):
                    crossfade_wav_files([wav1, wav2])
            finally:
                os.unlink(wav1)
                os.unlink(wav2)
